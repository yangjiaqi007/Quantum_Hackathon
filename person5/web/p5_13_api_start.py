# %% P5-14 Start and verify the real local quantum API
# The notebook is intended to run from the repository root. This starts the team's
# compiler + QAOA API on the local simulator, so the notebook never silently falls
# back to the development mock.
import atexit as p5_atexit
import subprocess as p5_subprocess
import sys as p5_sys
from pathlib import Path as p5_PathlibPath
from urllib.parse import urlparse as p5_urlparse

P5_API_PROCESS = None
P5_API_LOG = None


def p5_api_capabilities(timeout=2):
    request = p5_urllib_request.Request(P5_API_BASE_URL + "/capabilities")
    with p5_urllib_request.urlopen(request, timeout=timeout) as response:
        return p5_json.loads(response.read().decode("utf-8"))


def p5_stop_local_api():
    global P5_API_PROCESS, P5_API_LOG
    if P5_API_PROCESS is not None and P5_API_PROCESS.poll() is None:
        P5_API_PROCESS.terminate()
        try:
            P5_API_PROCESS.wait(timeout=5)
        except p5_subprocess.TimeoutExpired:
            P5_API_PROCESS.kill()
    if P5_API_LOG is not None:
        P5_API_LOG.close()


def p5_start_real_api():
    global P5_API_PROCESS, P5_API_LOG
    root = next((candidate for candidate in (p5_PathlibPath.cwd(), *p5_PathlibPath.cwd().parents)
                 if (candidate / "person4" / "backend" / "main.py").exists()), None)
    if root is None:
        raise RuntimeError("Start Jupyter from the project repository root so the real backend can be loaded.")
    p5_os.chdir(root)
    address = p5_urlparse(P5_API_BASE_URL)
    if address.hostname not in ("127.0.0.1", "localhost"):
        raise RuntimeError("P5_API_URL must point to a local API for this notebook's automatic setup.")

    try:
        capabilities = p5_api_capabilities()
    except Exception:
        capabilities = None
    if capabilities is None:
        env = p5_os.environ.copy()
        env.update({
            "PIPELINE_MODE": "team",
            "PIPELINE_VERSION": "hackathon-submission-v1",
            "COMPILER_CALLABLE": "compiler.adapter:compile_melody",
            "SOLVER_CALLABLE": "person2.quantum.adapter:solve",
            "XY_SOLVER_CALLABLE": "person3.quantum.adapter:solve",
        })
        log_path = p5_tmp.gettempdir() + "/quantum_harmony_api.log"
        P5_API_LOG = open(log_path, "w", encoding="utf-8")
        P5_API_PROCESS = p5_subprocess.Popen(
            [p5_sys.executable, "-m", "uvicorn", "person4.backend.main:app",
             "--host", address.hostname, "--port", str(address.port or 8000), "--workers", "1"],
            cwd=root, env=env, stdout=P5_API_LOG, stderr=p5_subprocess.STDOUT,
        )
        p5_atexit.register(p5_stop_local_api)
        deadline = p5_time.monotonic() + 45
        last_error = None
        while p5_time.monotonic() < deadline:
            if P5_API_PROCESS.poll() is not None:
                break
            try:
                capabilities = p5_api_capabilities()
                break
            except Exception as exc:
                last_error = exc
                p5_time.sleep(0.5)
        if capabilities is None:
            p5_stop_local_api()
            detail = f" ({last_error})" if last_error else ""
            raise RuntimeError(f"Could not start the real QAOA API. See {log_path}{detail}")

    if not capabilities.get("team_adapters_configured") or "aer" not in capabilities.get("configured_backends", []):
        raise RuntimeError("The API is not configured for real QAOA on Aer. Stop any existing API and rerun this cell.")
    if not capabilities.get("xy_solver_configured"):
        raise RuntimeError("The API is missing the Person 3 XY-QAOA adapter. Restart this notebook kernel to apply its configuration.")
    print("Real team API ready:", ", ".join(capabilities["configured_backends"]))
    print("Solvers ready: standard QAOA and constraint-preserving XY-QAOA")
    return capabilities


P5_API_CAPABILITIES = p5_start_real_api()
