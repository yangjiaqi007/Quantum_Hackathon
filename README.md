# Quantum Four-Part Harmony

A local web-style music tool: enter or upload a soprano melody, then use a Qiskit QAOA pipeline to generate alto, tenor, and bass. The Jupyter interface shows the score, audio playback, up to three sampled chord progressions, exports, and a quantum resource report.

The notebook is the interface; FastAPI is its local service. The default UI solver is **standard QAOA with an X mixer**. The XY-QAOA research code and experiments are in `person3/`, but XY-QAOA is not offered by the website. The project makes no quantum-advantage claim.

## Setup on Windows

These instructions use Windows PowerShell and Python 3.14.7, the version used for the current development setup. Install Git and Python 3.14 (64-bit) first. During Python installation, enable the option to add Python to PATH.

Open PowerShell in the folder where you want the project, then run:

```powershell
git clone https://github.com/yangjiaqi007/Quantum_Hackathon.git
cd Quantum_Hackathon
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-web.txt -r requirements-dev.txt
.\.venv\Scripts\python.exe -m ipykernel install --user --name quantum-harmony --display-name "Python (Quantum Harmony)"
```

If `py -3.14` is not recognized, install Python 3.14 and reopen PowerShell. The `requirements-web.txt` file installs the API, Qiskit/Aer, and notebook interface dependencies. IBM credentials are not required for the local Aer simulator.

## Run the app

Open **two PowerShell windows**, both in the cloned `Quantum_Hackathon` folder.

### 1. Start the local QAOA API

In the first window, configure the project adapters for this server session and start the API:

```powershell
$env:PIPELINE_MODE = "team"
$env:COMPILER_CALLABLE = "compiler.adapter:compile_melody"
$env:SOLVER_CALLABLE = "person2.quantum.adapter:solve"
$env:PIPELINE_VERSION = "p5-api-v2"
.\.venv\Scripts\python.exe -m uvicorn person4.backend.main:app --host 127.0.0.1 --port 8000 --workers 1
```

Keep this window open. Visit **http://127.0.0.1:8000/capabilities**. The service should report the team adapters and Aer as configured. **http://127.0.0.1:8000/docs** provides the API documentation.

### 2. Open the notebook interface

In the second PowerShell window:

```powershell
.\.venv\Scripts\python.exe -m jupyter lab person5/project.ipynb
```

In JupyterLab, select the **Python (Quantum Harmony)** kernel if prompted, then choose **Kernel → Restart Kernel and Run All Cells**. In the panel, choose a preset melody or enter/upload one, keep **Use local quantum API** checked, select **Aer simulator**, and click **Run harmonization**. The web UI exposes standard QAOA only.

The result appears in **Score & audio**, **Quantum report**, and **Export**. If QAOA sampling produces them, the Harmony choice menu offers up to three low-energy solutions with different chord progressions; fewer than three may be available. Exported MIDI, MusicXML, WAV, and PNG files go into a local `p5_output/` folder.

Stop the API with **Ctrl+C** in its PowerShell window. It must be running before you submit a harmonization job.

## IBM Quantum hardware (optional)

The local Aer simulator is the recommended first run. To use IBM hardware, configure these variables in the **API server's** PowerShell window before starting Uvicorn:

```powershell
$env:IBM_ENABLED = "true"
$env:IBM_TOKEN = "your IBM Quantum API key"
$env:IBM_INSTANCE = "your IBM Quantum instance CRN"
$env:IBM_BACKEND = "your accessible backend name"
```

Keep the token private. Do not put it in the notebook, source code, or a committed file. Confirm that `ibm` appears in `/capabilities` before selecting IBM hardware in the notebook. Hardware availability, queue time, and account access depend on IBM Quantum.

## Offline interface and mock mode

Unchecking **Use local quantum API** runs a local classical fixture for interface work. It is not a QAOA result. The API's default configuration is also a mock; the PowerShell environment settings above select the team's compiler and QAOA solver. The resource report identifies mock results honestly.

## Project map

| Path | Contents |
| --- | --- |
| `person1/compiler/` and `compiler/` | Melody presets, music-rule costs, candidate voicings, and compiler adapter |
| `person2/quantum/` | Standard QAOA, QUBO/Ising conversion, and window solving |
| `person3/quantum/`, `person3/experiments/` | XY-mixer research solver and experiments; not connected to the website solver menu |
| `person4/backend/` | FastAPI service, jobs, hardware backends, and resource reports |
| `person5/project.ipynb` | Jupyter interface for melody entry, score, audio, reports, and export |
| `person5/web/` | Notebook source cells and notebook builder |
| `Schema.py` | Shared melody, compiled problem, solver, and resource contracts |

The notebook is generated from the source files in `person5/web/`. After editing those source files, rebuild it from the repository root:

```powershell
.\.venv\Scripts\python.exe person5/web/build_notebook.py --out ..\project.ipynb
```

The builder writes the output relative to its own folder, so `..\project.ipynb` targets `person5/project.ipynb`.

## Development checks

Run the project's test suite from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pytest tests person3/tests person4/tests person2/tests -q
```

Run the Part 5 UI and export self-check:

```powershell
.\.venv\Scripts\python.exe person5/dev/p5_selftest.py
```

## Notes

- Harmonic results depend on the compiler's candidates and rule weights; the program is a demonstrator, not a substitute for a music teacher or a claim of musical correctness.
- Candidate chord progressions are ranked from feasible QAOA samples collected from independently solved windows. They are not guaranteed to be the global top three.
- The score renderer is intentionally simple. Dotted durations, soprano rests, and ties across barlines are supported; the synthesized audio uses a basic timbre.
- `.env`, `.venv`, `.local-data`, generated exports, and local scratch files should remain local and are excluded from Git.

For backend-specific API contracts and hardware details, see [`person4/README.md`](person4/README.md). For the notebook workflow and controls, see [`person5/README.md`](person5/README.md).
