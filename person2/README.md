# Person 2: Quantum core (plain QAOA + sliding windows)

Drop the `quantum/` folder at the repo root (next to Person 1's `compiler/`). The deliverables named in the plan are `quantum/qaoa.py` and `quantum/windowing.py`. The other files support them.

| File | Purpose |
|---|---|
| `quantum/problem.py` | `HarmonyProblem`: the `h` / `W` contract and an exact energy function |
| `quantum/qubo.py` | One-hot QUBO per window, fixed-boundary folding, provably safe penalty, Ising conversion |
| `quantum/qaoa.py` | `solve_window(qubo, ...)` (alias `solve`): optimise angles, sample, filter infeasible, rank |
| `quantum/windowing.py` | `solve_melody(problem, ...)`: sliding windows, stitching, warm-started angles |
| `quantum/qiskit_backend.py` | Qiskit V2 primitives backend (statevector, Aer, fake backends, IBM Runtime) |
| `quantum/simulator.py` | Dependency-free statevector simulator, used for tests and as a reference |
| `quantum/optimizers.py` | SciPy COBYLA if available, built-in Nelder-Mead otherwise |

## Usage

```python
from quantum import HarmonyProblem, solve_melody

problem = HarmonyProblem(h=h, W=W)            # from Person 1's compiler
result = solve_melody(problem, window=3, stride=3, p=2, backend="qiskit", shots=1024)
result.assignment     # candidate index chosen for every slot
result.energy         # exact full-problem energy of the stitched answer
result.windows        # per-window: qubits, evals, feasible_fraction, params, samples, ...
```

## The `h` / `W` contract (confirm with Person 1)

I could not read the repo when writing this, so this contract comes from the plan. If Person 1 emits something else, only `HarmonyProblem` needs to change.

* `h[i][k]`: cost of candidate `k` in slot `i`.
* `W[(i, k, j, l)]` with `i < j`: cost of picking `k` in slot `i` and `l` in slot `j`. `j` need not be `i + 1`.
* Lower energy is better. Slots may have different candidate counts.
* A dense `W[i][k][j][l]` goes through `HarmonyProblem.from_dense(h, W)`. It raises if the lower `i >= j` block is non-zero, because reading a symmetric W twice would silently double every penalty. Pass `symmetric=True` to ignore the lower block.

## Design notes

* **Encoding:** one qubit per (slot, candidate). A 3-slot window with K=4 uses 12 qubits and a 4-slot window uses 16.
* **Penalty:** `auto_penalty` is larger than the biggest possible energy change from flipping one bit. Every ground state of the penalised QUBO is therefore one-hot (tested exhaustively). `penalty_factor=` scales it. Larger values make the landscape flatter for QAOA.
* **Boundary:** every already-committed slot is folded into the window's linear terms, not just the last one, so non-adjacent couplings are handled.
* **Windows:** `stride < window` gives overlapping windows with lookahead. `stride == window` (default) never revisits a slot.
* **Warm start:** the Ising coefficients are normalised to max 1 in every window, so angles transfer. On a repeated-structure test, warm starts cut optimizer evaluations from 433 to 370 (about 15%).
* **Sampling:** samples that are not one-hot are dropped. Survivors are ranked by harmony energy. If a window has no feasible sample, a greedy repair runs and the result is flagged `repaired=True`. Report this in the honesty report rather than hiding it.
* **Person 3:** an XY-mixer solver only has to match `solve(qubo, **kw) -> WindowResult`. Pass it as `solve_melody(..., solver=my_solver)`.
* **Person 4:**
  * `QiskitBackend(...).circuit` and `.measured_circuit` are what you transpile.
  * `WindowResult` carries `n_qubits`, `n_evals`, `seconds`, `feasible_fraction`, `params`, and `trace`.
  * Hardware: pass `backend_options={"estimator": ..., "sampler": ..., "pass_manager": ...}`.
  * `progress=` gives a per-window callback for the progress bar.
* **Python simulator:** it is exact but slow. A 12-qubit window takes about 2 s. Use `backend="qiskit"` for 16 qubits.

## Running the tests

```bash
pip install -r requirements.txt
pytest -v                          # everything
pytest tests/test_qiskit_backend.py -v   # Qiskit cross-checks only
python examples/demo.py [python|qiskit]
```

**Test status:**

* In the sandbox where this was written (no PyPI access, so no Qiskit), 44 tests passed and the Qiskit cross-check file (`test_qiskit_backend.py`) was skipped as a whole.
* `tests/test_qiskit_backend.py` has never been run. It checks the Qiskit circuit against the simulator on expectation values, parameter ordering (p >= 10), sampled bit order, and the Pauli operator. Run it first. If it fails, the bug is most likely in `qiskit_backend.py`, not the core.
* Injecting three bugs on purpose (a flipped ZZ sign, no boundary folding, a too-small penalty) made the suite fail each time, so the tests do detect real errors.
