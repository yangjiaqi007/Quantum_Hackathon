"""FastAPI entry point: uvicorn person4.backend.main:app --host 127.0.0.1 --port 8000."""
from __future__ import annotations

from contextlib import asynccontextmanager
from uuid import UUID
from importlib.util import find_spec

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware

from .config import Settings
from .jobs import JobManager
from .models import HarmonizeRequest
from .storage import QueueFull


def public_job(job: dict) -> dict:
    return {
        "job_id": job["id"], "status": job["status"], "progress": job["progress"],
        "created_at": job["created_at"], "updated_at": job["updated_at"],
        "started_at": job["started_at"], "finished_at": job["finished_at"],
        "error": ({"code": job["error_code"], "message": job["error_message"]}
                  if job["error_code"] else None),
        "provenance": {"cache_hit": job["cache_hit"], "source_job_id": job["source_job_id"],
                       "execution": job["request"]["run"]["backend"]},
        "status_url": f"/jobs/{job['id']}", "result_url": f"/jobs/{job['id']}/result",
    }


def create_app(settings=None, runner=None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app):
        manager = JobManager(settings, runner)
        manager.start()
        app.state.jobs = manager
        try:
            yield
        finally:
            manager.stop()

    app = FastAPI(title="Quantum Harmony Backend", version="1.0.0", lifespan=lifespan,
                  description="Local job service. Mock runs are development fixtures, not quantum results.")
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins,
                       allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["Content-Type"])

    def find_job(job_id: str):
        try:
            key = str(UUID(job_id))
        except ValueError:
            raise HTTPException(404, "Job not found") from None
        job = app.state.jobs.store.get(key)
        if job is None:
            raise HTTPException(404, "Job not found")
        return job

    @app.get("/health")
    def health():
        return {"status": "ok", "queue": app.state.jobs.store.counts()}

    @app.get("/capabilities")
    def capabilities():
        team_ready = (settings.pipeline_mode == 'team' and bool(settings.compiler_callable)
                      and bool(settings.solver_callable))
        quantum_installed = all(find_spec(name) is not None for name in ('qiskit', 'qiskit_aer'))
        runtime_installed = find_spec('qiskit_ibm_runtime') is not None
        ready = ['mock']
        if team_ready and quantum_installed:
            ready.append('aer')
            if runtime_installed:
                ready.append('noisy')
                if settings.ibm_enabled and all((settings.ibm_token, settings.ibm_instance, settings.ibm_backend)):
                    ready.append('ibm')
        return {
            "voice_order": ["bass", "tenor", "alto", "soprano"],
            "pitch_format": "MIDI numbers", "duration_unit": "quarter-note beats",
            "backends": ["mock", "aer", "noisy"] + (["ibm"] if settings.ibm_enabled else []),
            "configured_backends": ready,
            "team_adapters_configured": team_ready,
            "ibm_enabled": settings.ibm_enabled,
            "note": "Aer, noisy and IBM modes require the team's compiler and solver adapters and optional quantum packages.",
            "job_timeout_seconds": settings.job_timeout_seconds,
            "max_pending_jobs": settings.max_pending_jobs,
        }

    @app.post("/jobs", status_code=202)
    def submit(request: HarmonizeRequest, response: Response):
        cached = app.state.jobs.store.has_cached(request.model_dump(mode='json'), settings.cache_namespace)
        if not cached and request.run.backend == "ibm" and not settings.ibm_enabled:
            raise HTTPException(422, "IBM hardware is not enabled on this server.")
        if not cached and request.run.backend != 'mock' and request.run.backend not in capabilities()['configured_backends']:
            raise HTTPException(422, "This backend is not configured. Connect team adapters and install quantum packages first.")
        try:
            job = app.state.jobs.submit(request)
        except QueueFull as exc:
            raise HTTPException(429, str(exc), headers={"Retry-After": "5"}) from None
        response.headers["Location"] = f"/jobs/{job['id']}"
        return public_job(job)

    @app.get("/jobs/{job_id}")
    def status(job_id: str):
        return public_job(find_job(job_id))

    @app.get("/jobs/{job_id}/result")
    def result(job_id: str):
        job = find_job(job_id)
        if job["status"] != "completed":
            raise HTTPException(409, {"message": "Result is not available.", "job": public_job(job)})
        return {"job_id": job["id"], "status": "completed", "result": job["result"],
                "provenance": public_job(job)["provenance"]}

    @app.post("/jobs/{job_id}/cancel")
    def cancel(job_id: str):
        job = find_job(job_id)
        if job["status"] in ("completed", "failed"):
            raise HTTPException(409, "This job has already finished.")
        return public_job(app.state.jobs.store.cancel(job["id"]))

    return app


app = create_app()
