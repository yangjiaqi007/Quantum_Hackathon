"""SQLite persistence for the local, single-worker job service."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from uuid import uuid4


TERMINAL = {"completed", "failed", "cancelled"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def cache_key(request: dict, namespace: str) -> str:
    payload = {k: v for k, v in request.items() if k != "use_cache"}
    canonical = json.dumps({"namespace": namespace, "request": payload},
                           sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class QueueFull(Exception):
    pass


class Store:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.data_dir / "jobs.sqlite3"
        with self.connection() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, status TEXT NOT NULL,
                    request_json TEXT NOT NULL, cache_key TEXT NOT NULL,
                    result_json TEXT, progress_json TEXT NOT NULL,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    started_at TEXT, finished_at TEXT,
                    error_code TEXT, error_message TEXT,
                    cache_hit INTEGER NOT NULL DEFAULT 0, source_job_id TEXT
                );
                CREATE INDEX IF NOT EXISTS jobs_status_created ON jobs(status, created_at);
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL REFERENCES jobs(id)
                );
            """)

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def decode(row):
        if row is None:
            return None
        job = dict(row)
        for field in ("request", "result", "progress"):
            raw = job.pop(field + "_json")
            job[field] = json.loads(raw) if raw is not None else None
        job["cache_hit"] = bool(job["cache_hit"])
        return job

    def get(self, job_id: str):
        with self.connection() as db:
            return self.decode(db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone())

    def has_cached(self, request: dict, namespace: str) -> bool:
        if not request.get('use_cache', True):
            return False
        with self.connection() as db:
            return db.execute("""SELECT 1 FROM cache JOIN jobs ON cache.job_id=jobs.id
                WHERE cache.key=? AND jobs.status='completed'""",
                (cache_key(request, namespace),)).fetchone() is not None

    def create(self, request: dict, namespace: str, limit: int):
        key = cache_key(request, namespace)
        job_id, now = str(uuid4()), utc_now()
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            source = None
            if request.get("use_cache", True):
                source = db.execute("""SELECT jobs.* FROM cache JOIN jobs ON cache.job_id=jobs.id
                    WHERE cache.key=? AND jobs.status='completed'""", (key,)).fetchone()
            if source is None:
                pending = db.execute("SELECT count(*) FROM jobs WHERE status IN ('queued','running')").fetchone()[0]
                if pending >= limit:
                    raise QueueFull("The job queue is full. Try again shortly.")
            status = "completed" if source else "queued"
            progress = {"stage": "Loaded saved result" if source else "Waiting to start",
                        "completed_windows": 0, "total_windows": None}
            db.execute("""INSERT INTO jobs
                (id,status,request_json,cache_key,result_json,progress_json,created_at,updated_at,
                 started_at,finished_at,cache_hit,source_job_id)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (job_id, status, json.dumps(request, allow_nan=False), key,
                 source["result_json"] if source else None, json.dumps(progress), now, now,
                 source["started_at"] if source else None, now if source else None,
                 int(source is not None), source["id"] if source else None))
        return self.get(job_id)

    def recover(self, namespace=None):
        now = utc_now()
        with self.connection() as db:
            db.execute("""UPDATE jobs SET status='failed',error_code='interrupted',
                error_message='The server stopped during this run. Submit a new job.',
                updated_at=?,finished_at=? WHERE status='running'""", (now, now))
            if namespace is not None:
                # A queued request belongs to the pipeline configuration at
                # submission. Do not execute it with new rules under an old key.
                for row in db.execute("SELECT id,request_json,cache_key FROM jobs WHERE status='queued'").fetchall():
                    if cache_key(json.loads(row['request_json']), namespace) != row['cache_key']:
                        db.execute("""UPDATE jobs SET status='failed',error_code='configuration_changed',
                            error_message='Pipeline settings changed. Submit this melody again.',
                            updated_at=?,finished_at=? WHERE id=?""", (now, now, row['id']))

    def claim(self):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM jobs WHERE status='queued' ORDER BY created_at,id LIMIT 1").fetchone()
            if row is None:
                return None
            now = utc_now()
            db.execute("""UPDATE jobs SET status='running',started_at=?,updated_at=?,progress_json=?
                WHERE id=?""", (now, now, json.dumps({"stage": "Starting", "completed_windows": 0,
                                                     "total_windows": None}), row["id"]))
        return self.get(row["id"])

    def progress(self, job_id: str, update: dict):
        with self.connection() as db:
            db.execute("UPDATE jobs SET progress_json=?,updated_at=? WHERE id=? AND status='running'",
                       (json.dumps(update, allow_nan=False), utc_now(), job_id))

    def complete(self, job_id: str, result: dict):
        encoded = json.dumps(result, allow_nan=False)
        now = utc_now()
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            changed = db.execute("""UPDATE jobs SET status='completed',result_json=?,updated_at=?,finished_at=?
                WHERE id=? AND status='running'""", (encoded, now, now, job_id)).rowcount
            if changed:
                row = db.execute("SELECT cache_key,progress_json FROM jobs WHERE id=?", (job_id,)).fetchone()
                progress = json.loads(row["progress_json"])
                progress["stage"] = "Completed"
                db.execute("UPDATE jobs SET progress_json=? WHERE id=?", (json.dumps(progress), job_id))
                db.execute("INSERT OR REPLACE INTO cache(key,job_id) VALUES (?,?)", (row["cache_key"], job_id))

    def fail(self, job_id: str, code: str, message: str):
        now = utc_now()
        with self.connection() as db:
            db.execute("""UPDATE jobs SET status='failed',error_code=?,error_message=?,updated_at=?,finished_at=?
                WHERE id=? AND status='running'""", (code, message, now, now, job_id))

    def cancel(self, job_id: str):
        now = utc_now()
        with self.connection() as db:
            db.execute("""UPDATE jobs SET status='cancelled',updated_at=?,finished_at=?,progress_json=?
                WHERE id=? AND status IN ('queued','running')""",
                (now, now, json.dumps({"stage": "Cancelled", "completed_windows": 0,
                                     "total_windows": None}), job_id))
        return self.get(job_id)

    def counts(self):
        with self.connection() as db:
            return dict(db.execute("SELECT status,count(*) FROM jobs GROUP BY status").fetchall())
