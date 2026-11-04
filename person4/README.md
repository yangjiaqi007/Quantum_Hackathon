# Person 4: Backend, hardware, and resource reporting

The FastAPI service and related backend modules are in `backend/`. From the repository root, launch it with `python -m uvicorn person4.backend.main:app --host 127.0.0.1 --port 8000 --workers 1`, or run `person4/start.ps1` on Windows. The service defaults to a development mock unless configured for the team pipeline. Example API input is in `person4/examples/`; tests are in `person4/tests/`.
