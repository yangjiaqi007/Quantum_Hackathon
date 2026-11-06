# Quantum Four-Part Harmony

A local web-style music tool: enter or upload a soprano melody, then use a Qiskit QAOA pipeline to generate alto, tenor, and bass. The Jupyter interface shows the score, audio playback, up to three sampled chord progressions, exports, and a quantum resource report.

The notebook is the interface; its setup cell starts the local FastAPI service with the team's compiler, standard QAOA, and XY-QAOA adapters. Both solvers run on the Aer simulator by default. The project makes no quantum-advantage claim.

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

Open PowerShell in the repository root and run:

```powershell
.\.venv\Scripts\python.exe -m jupyter lab person5/project.ipynb
```

In JupyterLab, select the **Python (Quantum Harmony)** kernel if prompted, then choose **Kernel → Restart Kernel and Run All Cells**. The setup cell starts the local API in team mode and confirms that Aer and both solvers are connected. The self-check runs a short real QAOA and XY-QAOA example; the interactive panel lets you choose either solver. No IBM account or hardware access is needed.

The result appears in **Score & audio**, **Quantum report**, and **Export**. If QAOA sampling produces them, the Harmony choice menu offers up to three low-energy solutions with different chord progressions; fewer than three may be available. Exported MIDI, MusicXML, WAV, and PNG files go into a local `p5_output/` folder.

The API process is stopped when the notebook kernel shuts down. The notebook refuses to continue if it sees a mock or incomplete API.

## IBM Quantum hardware (optional)

The local Aer simulator is the recommended first run. To use IBM hardware, configure these variables in the **API server's** PowerShell window before starting Uvicorn:

```powershell
$env:IBM_ENABLED = "true"
$env:IBM_TOKEN = "your IBM Quantum API key"
$env:IBM_INSTANCE = "your IBM Quantum instance CRN"
$env:IBM_BACKEND = "your accessible backend name"
```

Keep the token private. Do not put it in the notebook, source code, or a committed file. Confirm that `ibm` appears in `/capabilities` before selecting IBM hardware in the notebook. Hardware availability, queue time, and account access depend on IBM Quantum.

## Development mock

The development mock is excluded from `person5/project.ipynb`. Its notebook setup requires the real team adapters and Aer simulator, and the self-check confirms that both QAOA variants return non-mock results.

## Project map

| Path | Contents |
| --- | --- |
| `person1/compiler/` and `compiler/` | Melody presets, music-rule costs, candidate voicings, and compiler adapter |
| `person2/quantum/` | Standard QAOA, QUBO/Ising conversion, and window solving |
| `person3/quantum/`, `person3/experiments/` | Constraint-preserving XY-QAOA solver and experiments |
| `person4/backend/` | FastAPI service, jobs, hardware backends, and resource reports |
| `person5/project.ipynb` | Jupyter interface for melody entry, score, audio, reports, and export |
| `person5/web/` | Notebook source cells and notebook builder |
| `Schema.py` | Shared melody, compiled problem, solver, and resource contracts |

The notebook is generated from the source files in `person5/web/`. After editing those source files, rebuild it from the repository root:

```powershell
.\.venv\Scripts\python.exe person5/web/build_notebook.py --no-mock --out project.ipynb
```

The builder writes the output relative to its own folder, so `project.ipynb` targets `person5/project.ipynb`. The submission notebook excludes the development mock.

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
