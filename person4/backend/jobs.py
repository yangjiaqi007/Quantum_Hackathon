"""One durable queue consumer; each calculation runs in a killable child process."""
from __future__ import annotations

import logging
import multiprocessing
from pathlib import Path
import queue
import threading
import time

from .storage import Store

logger = logging.getLogger(__name__)


class InstanceLock:
    """Prevent two servers recovering or processing the same local queue."""
    def __init__(self, directory: Path):
        self.path = Path(directory) / "worker.lock"
        self.handle = None

    def acquire(self):
        import os
        self.handle = self.path.open("a+b")
        self.handle.seek(0)
        if not self.handle.read(1):
            self.handle.write(b"0")
            self.handle.flush()
        self.handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.handle.close()
            self.handle = None
            raise RuntimeError("Another backend is using this data folder. Run one server worker.") from exc

    def release(self):
        if self.handle is not None:
            self.handle.close()
            self.handle = None


def execute_job(payload, settings, events, runner=None):
    """Top-level function is importable by Windows' spawn process model."""
    from .models import HarmonizeRequest
    try:
        if runner is None:
            from .pipeline import run_pipeline
            runner = run_pipeline
        request = HarmonizeRequest.model_validate(payload)
        result = runner(request, settings, lambda update: events.put(("progress", update)))
        events.put(("result", result))
    except Exception as exc:
        logger.exception("Harmonization worker failed")
        # Provider messages can contain configuration or credentials: keep them out of responses.
        from .contracts import PipelineError
        message = (exc.public_message if isinstance(exc, PipelineError)
                   else "Harmonization failed. Check the server log for details.")
        events.put(("error", {"code": "pipeline_error", "message": message}))


class JobManager:
    def __init__(self, settings, runner=None):
        self.settings = settings
        self.store = Store(settings.data_dir)
        self.runner = runner
        self.stop_event = threading.Event()
        self.wake_event = threading.Event()
        self.lock = InstanceLock(self.store.data_dir)
        self.thread = None

    def start(self):
        self.lock.acquire()
        try:
            self.store.recover(self.settings.cache_namespace)
            self.thread = threading.Thread(target=self._loop, name="harmony-worker", daemon=True)
            self.thread.start()
        except BaseException:
            self.lock.release()
            raise

    def stop(self):
        self.stop_event.set()
        self.wake_event.set()
        if self.thread:
            self.thread.join(timeout=10)
            if self.thread.is_alive():
                raise RuntimeError("Worker did not stop; keep its data folder exclusive until shutdown.")
        self.lock.release()

    def submit(self, request):
        job = self.store.create(request.model_dump(mode="json"),
                                self.settings.cache_namespace, self.settings.max_pending_jobs)
        self.wake_event.set()
        return job

    def _loop(self):
        while not self.stop_event.is_set():
            try:
                job = self.store.claim()
                if job:
                    self._run(job)
                    continue
            except Exception:
                logger.exception("Job queue worker error")
            self.wake_event.wait(0.25)
            self.wake_event.clear()

    def _run(self, job):
        context = multiprocessing.get_context("spawn")
        events = context.Queue()
        process = context.Process(target=execute_job,
                                  args=(job["request"], self.settings, events, self.runner),
                                  name="harmony-calculation")
        process.daemon = True
        started = time.monotonic()
        terminal = False
        try:
            process.start()
            while not terminal:
                if self.stop_event.is_set():
                    self.store.fail(job["id"], "interrupted", "The server stopped during this run. Submit a new job.")
                    break
                if self.store.get(job["id"])["status"] == "cancelled":
                    break
                if time.monotonic() - started > self.settings.job_timeout_seconds:
                    self.store.fail(job["id"], "timeout", "The run exceeded the configured time limit.")
                    break
                try:
                    kind, payload = events.get(timeout=0.1)
                except queue.Empty:
                    if not process.is_alive():
                        self.store.fail(job["id"], "worker_exit", "The calculation process stopped unexpectedly.")
                        break
                    continue
                if kind == "progress":
                    self.store.progress(job["id"], payload)
                elif kind == "result":
                    self.store.complete(job["id"], payload)
                    terminal = True
                elif kind == "error":
                    self.store.fail(job["id"], payload["code"], payload["message"])
                    terminal = True
        except Exception:
            logger.exception("Could not execute job %s", job["id"])
            self.store.fail(job["id"], "worker_error", "The backend could not execute this job.")
        finally:
            if process.pid is not None:
                # A completed run should exit normally; failed, cancelled and timed-out work is stopped.
                process.join(timeout=0.5 if terminal else 0)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=2)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=2)
                process.close()
            events.cancel_join_thread()
            events.close()
