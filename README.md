# Quantum Harmony: Person 4 backend

This is the Python service connecting the music compiler, Qiskit solver, and website. It accepts melody jobs, runs one calculation at a time in a separate process, stores progress/results in SQLite, and reuses completed demo results. The existing `Schema.py` is the team contract and has not been edited.

**The default mode is a development mock.** It creates synthetic accompanying pitches to test integration. It does not perform QAOA or promise valid harmony. The real music compiler and QAOA/XY-QAOA functions belong to Persons 1–3 and must be connected using the adapter contract below.

## Run on Kevin's laptop

Open PowerShell in this project folder. The local `.venv` has already been prepared. Start the service:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --workers 1
```

Alternatively run `.\start.ps1` if PowerShell permits local scripts. Open **http://127.0.0.1:8000/docs** for the interactive API. Use **POST /jobs**, click **Try it out**, paste `examples/ode_to_joy.json`, then execute. Copy its job ID into **GET /jobs/{job_id}** and **GET /jobs/{job_id}/result**. Press Ctrl+C in the server terminal to stop it.

In another PowerShell window, submit and cache the example automatically:

```powershell
.\.venv\Scripts\python.exe -m backend.demo examples/ode_to_joy.json --output work/ode-result.json
```

Run that command again to see `Saved result reused: True`. `--fresh` bypasses reuse and saves a new result. To demo O Canada, serialize the compiler preset in the same request format and pass it to this command. No hardware results are fabricated.

The compiler package includes Ode to Joy and O Canada presets. O Canada follows the official Canadian Heritage score; the melody is public domain.

## Person 1 compiler API

The compiler is available as `compiler/` while the original root modules remain import-compatible. It supports note-name strings (`melody_from_text`), monophonic MIDI files (`melody_from_midi`, requires `mido`), and `get_preset("ode_to_joy")` / `get_preset("o_canada")`. MIDI input selects the track with the most notes unless a track index is supplied. Rests are represented as a hold of the preceding pitch because the shared melody schema has no rest event; polyphonic tracks are rejected. `compiler.adapter:compile_melody` returns the backend `Compilation` envelope with slot durations, and `problem.h` / `problem.W` are the normalized cost arrays for quantum solvers. Harmony slots split at every melody-note boundary to preserve the supplied soprano.

## Setup on another computer

Use Python 3.11 or 3.12, from a terminal in the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

For actual simulator/noisy/hardware adapters also install:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-quantum.txt
```

Copy `.env.example` to `.env` when changing settings. Credentials and `.local-data` are ignored by Git. The service is designed for one local server with one Uvicorn worker. Do not use multiple workers or multiple servers against the same data folder. The exclusive worker lock prevents duplicate processing.

## Files and ownership

| File | Purpose |
| --- | --- |
| `backend/main.py` | FastAPI endpoints, CORS, capability information |
| `backend/models.py` | Public request validation |
| `backend/jobs.py` | Background process, progress, timeout, cancellation, single-instance lock |
| `backend/storage.py` | Persistent jobs and completed-result cache |
| `backend/pipeline.py` | Connect teammates, preserve timing, solve bounded windows, decode results |
| `backend/contracts.py` | Compiler envelope with slot durations |
| `backend/mock_solver.py` | Clearly labeled development fixtures |
| `backend/hardware.py` | Aer, fake-device noise, IBM Runtime V2 sampler, transpilation, resources |
| `backend/reporting.py` | Exact small correctness benchmark and quality report |
| `backend/config.py` | Server settings and private credentials |
| `backend/demo.py` | Submit and save/cache demo runs |
| `tests/` | API lifecycle, cache, failure, timing, encoding interface, and hardware-adapter checks |

Person 5 owns score rendering, playback, MIDI/MusicXML exports, and melody parsing in the website. The API supplies timed notes for all four voices so that work can start now. It accepts structured MIDI-number JSON; text notation/MIDI upload parsing is not implemented by this service.

## API contract for Person 5

| Method/path | Response |
| --- | --- |
| `GET /health` | Service state and queue counts |
| `GET /capabilities` | Supported modes, configured modes, units and voice order |
| `POST /jobs` | HTTP 202; job ID, status URL and result URL |
| `GET /jobs/{id}` | Status, progress, timestamps, safe error and cache provenance |
| `GET /jobs/{id}/result` | Completed harmony and reports; HTTP 409 until available |
| `POST /jobs/{id}/cancel` | Stop queued/local running work; HTTP 409 for finished jobs |

Statuses: `queued`, `running`, `completed`, `failed`, `cancelled`. Poll the status about once per second. Handle HTTP 422 for invalid/unconfigured requests, 429 for a full queue, and 404 for unknown IDs. `use_cache: false` forces fresh execution; success still populates the cache.

Minimal request:

```json
{
  "melody": {
    "title": "My melody",
    "tonic": 0,
    "mode": "major",
    "time_signature": [4, 4],
    "harmonic_rhythm": 1,
    "notes": [{"midi": 64, "duration": 1}, {"midi": 65, "duration": 1}]
  },
  "tempo": 100,
  "style": "chorale",
  "run": {"backend": "mock"},
  "use_cache": true
}
```

`tonic` is a pitch class 0–11; MIDI middle C is 60. Durations and onsets are quarter-note beats. The voice order is **bass, tenor, alto, soprano**, matching `Schema.py`. Do not use the reverse order from the earlier sample cost function. First version: monophonic soprano without rests, up to 128 notes, 256 beats and 256 harmony slots. Invalid notes, nonpositive durations and excess requests are rejected.

The completed-result endpoint returns `{job_id, status, provenance, result}`. `result` contains:

- `voices`: `{bass, tenor, alto, soprano}`, each a list of `{midi, onset, duration}`; soprano preserves original articulation.
- `slots`: harmony slot timing, pitches in the shared order, chord and selected candidate index.
- `tempo`, `time_signature`, `tonic`, `mode`, and `title` for playback and notation.
- `execution`: actual mode, `is_mock`, pipeline version, elapsed seconds; unknown runtime estimates are null.
- `quantum_report`: per-window resources, energies, feasibility and timing.
- `quality_report`: whole-melody cost, exact reference where tractable, energy gap and optional rule counts.
- `warnings`: compiler warnings and missing-resource notices.

Always show `execution.is_mock` and `provenance.cache_hit` honestly. A cached hardware result retains its original run details and source job ID; it is not a live run. Resource reports refer to per-window circuits; do not add their qubits and call that the melody's circuit size. Final sampling shots do not include all optimizer evaluations.

Default CORS origins permit localhost ports 3000 and 5173. If the website uses another port, add its full origin to `CORS_ORIGINS`. This local backend has no public-hosting authentication; keep the bind address at `127.0.0.1` for laptop development.

## Connect Persons 1–3

These signatures are proposed adapters, not assumptions that teammates have already written them. Their existing functions can be wrapped without modifying `Schema.py`.

Compiler adapter:

```python
from backend.contracts import Compilation

def compile_melody(melody, *, style) -> Compilation:
    # melody is Schema.Melody. Build the team's CompiledProblem.
    return Compilation(problem=problem, slot_durations=durations, warnings=[])
```

Supply one positive duration per slot. Slots must partition the input notes in order; do not cross a note boundary. Every candidate's soprano must match the input. Costs must be finite and normalized according to `Schema.py`; no one-hot penalty is included in the compiler costs. Variable candidate counts are supported.

Single-window solver adapter:

```python
def solve_window(problem, *, run, execution, initial_params=None):
    # problem is Schema.CompiledProblem for THIS window.
    # run has solver, shots, reps, seed and max_iterations.
    # Build QAOA/XY-QAOA, then:
    compiled = execution.transpile(circuit)
    job = execution.sampler.run([compiled], shots=run.shots)
    # Decode job.result() using measured classical-bit order.
    # Return Schema.SolverResult with:
    #   solver=run.solver, best=<candidate indices>,
    #   best_energy=problem.energy(best), feasible_fraction=...,
    #   optimal_params=..., resources=execution.resources(compiled, run.reps, run.shots)
```

Person 2 implements circuit construction, optimizer loop, sample decoding and feasibility filtering. No feasible sample should be reported as a failure, never silently replaced with a classical answer. Qiskit's displayed bitstrings put qubit 0 at the right; use the schema's decoding convention and maintain classical measurement ordering after transpilation. Reuse suggested angles only when their layout matches your circuit.

The service already partitions windows to fit `window_size` and `max_qubits`, folds previous boundary and fixed long-range costs through `CompiledProblem.window`, and reports window progress. If Person 2 already implements full-melody windowing, agree on a single owner rather than nesting two windowing loops. The current greedy strategy has no look-ahead guarantee; full-problem energy is recomputed after stitching.

Optional independent rule checker:

```python
def count_violations(melody, voicings, slot_durations):
    return {"parallel_fifths": 0, "voice_crossing": 0}
```

Configure actual import paths in `.env`, for example:

```dotenv
PIPELINE_MODE=team
PIPELINE_VERSION=v2
COMPILER_CALLABLE=compiler.adapter:compile_melody
SOLVER_CALLABLE=quantum.adapter:solve_window
CHECKER_CALLABLE=compiler.checker:count_violations
```

Restart the server, select `run.backend: "aer"`, and check `/capabilities`. Package presence and configured adapter paths are prerequisites; import compatibility is checked during a job. Set `run.solver: "xy_qaoa"` only after Person 3 supports it. Increment `PIPELINE_VERSION` whenever compiler rules, solver behavior, or adapter code changes so cached results cannot hide changes.

## Hardware and resource reporting

`aer` uses the local ideal V2 sampler. `noisy` uses packaged fake-device calibration snapshots; it does not contact IBM. Default `FakeGuadalupeV2` supports 16-qubit windows. The five-qubit `FakeManilaV2` is useful for tiny adapter tests. Capacity is checked during transpilation.

IBM hardware requires all of these server settings:

```dotenv
IBM_ENABLED=true
IBM_TOKEN=your-private-token
IBM_INSTANCE=your-instance-CRN
IBM_BACKEND=your-accessible-backend
```

Keep these values local; never put them in frontend code or request JSON. Hardware is disabled by default, and no IBM hardware job was run as part of this build. The adapter creates an IBM Runtime SamplerV2 in job mode; optimizer iteration/shot budgets remain Person 2's responsibility. Actual account access, device compatibility, queueing and results require a later end-to-end run with the real solver.

Local cancellation/timeouts stop the calculation process. They cannot guarantee cancellation of a job already submitted to IBM. Check IBM workloads separately for such a job. Do not depend on a live hardware queue during judging.

`context.resources()` measures a transpiled circuit and records logical width, physical register width, active physical qubits, depth and two-qubit gate count. Elapsed times are measured; runtime estimates stay null because queue and optimization duration are unknown. The small exact checker measures quality within the same candidate set and cost model; it does not establish musical correctness or quantum advantage. Weighted rule costs are not violation counts. Independent counts remain null until a checker is connected.

## Persistence, tests and sources

Local state lives in `.local-data/jobs.sqlite3`. Completed results and caches survive restart; interrupted running jobs become failed, and queued jobs resume. Mock, ideal, noisy and hardware requests have separate cache keys, including melody, style, run settings, configured adapters and pipeline version. Fake-device snapshots may change with package versions, so bump the pipeline version when updating quantum packages too.

Previously saved hardware demos can be loaded from cache even when hardware execution is disabled or the token has been removed. Keep the same pipeline version and instance/backend identity settings to address the same cache entry. No fresh hardware run is permitted without configuration. Queued jobs from a different pipeline configuration fail with a request to resubmit.

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q --basetemp=work/test-temp
```

The dependency ranges allow compatible updates; `requirements-lock.txt` records the versions used in this build. To reproduce the tested laptop environment, install that file instead of the ranged requirement files.

Official integration references: [FastAPI CORS](https://fastapi.tiangolo.com/tutorial/cors/), [Aer SamplerV2](https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.primitives.SamplerV2.html), [IBM Runtime SamplerV2](https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/sampler-v2).
