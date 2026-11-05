# Person 2: Quantum core (plain QAOA + sliding windows)

This folder lives at `person2/` in the team repo. The deliverables named in the plan are `quantum/qaoa.py` and `quantum/windowing.py`. `quantum/adapter.py` connects them to the team's `Schema.py` contract and Person 4's pipeline. The other files support them.

| File | Purpose |
|---|---|
| `quantum/problem.py` | `HarmonyProblem`: the `h` / `W` contract and an exact energy function |
| `quantum/qubo.py` | One-hot QUBO per window, fixed-boundary folding, provably safe penalty, Ising conversion |
| `quantum/qaoa.py` | `solve_window(qubo, ...)` (alias `solve`): optimise angles, sample, filter infeasible, rank |
| `quantum/adapter.py` | `solve(window, *, run, execution, initial_params) -> SolverResult`, the entry point Person 4's pipeline calls |
| `quantum/windowing.py` | `solve_melody(problem, ...)`: sliding windows, stitching, warm-started angles |
| `quantum/qiskit_backend.py` | Qiskit V2 primitives backend (statevector, Aer, fake backends, IBM Runtime) |
| `quantum/simulator.py` | Dependency-free statevector simulator, used for tests and as a reference |
| `quantum/optimizers.py` | SciPy COBYLA if available, built-in Nelder-Mead otherwise |

## Usage

```python
from person2.quantum import HarmonyProblem, solve_melody

problem = HarmonyProblem(h=h, W=W)            # from Person 1's compiler
result = solve_melody(problem, window=3, stride=3, p=2, backend="qiskit", shots=1024)
result.assignment     # candidate index chosen for every slot
result.energy         # exact full-problem energy of the stitched answer
result.windows        # per-window: qubits, evals, feasible_fraction, params, samples, ...
```

## Team contract and the adapter

The repo's `Schema.py` is the contract: `CompiledProblem` (`h[t]` arrays, adjacent `W[t]` matrices, a `long_range` list) in, `SolverResult` out. Person 4's `run_pipeline` cuts the melody with `CompiledProblem.window(...)` (boundary already folded into `h`) and calls `solve(window, *, run, execution, initial_params)` once per window.

`quantum/adapter.py` implements exactly that signature. To switch the backend from the mock to this solver:

```
PIPELINE_MODE=team
COMPILER_CALLABLE=compiler.adapter:compile_melody
SOLVER_CALLABLE=person2.quantum.adapter:solve
XY_SOLVER_CALLABLE=<module:function>   # Person 3's xy_qaoa adapter, once it exists
```

**Solver routing.** The pipeline has one `SOLVER_CALLABLE`, but a request can ask for `qaoa` or `xy_qaoa`. `solve` runs `qaoa` itself and forwards `xy_qaoa` to `XY_SOLVER_CALLABLE`. That callable has the same signature as `solve`, must return `Schema.SolverResult(solver="xy_qaoa", ...)`, and is **Person 3's to write**: `person3/quantum/xy_solve.py` returns its own `SolveResult` and samples with its own Aer calls, so it can't be used as is. If the variable is unset, an `xy_qaoa` request fails with a message that says what to set.

What it does per window:

1. Converts `CompiledProblem` to `HarmonyProblem` (`h`, adjacent `W`, `long_range` terms; duplicate keys add up).
2. Optimises the angles on an exact local simulator (pure Python up to 12 qubits, Qiskit statevector above that). No shot noise reaches the optimiser.
3. Samples the final circuit. With an `execution` context (Aer, fake noisy device, IBM) it transpiles via `execution.transpile`, samples with `execution.sampler`, and takes depth and two-qubit gate counts from `execution.resources`. Without one it samples locally and returns `resources=None`.
4. Drops samples that are not one-hot. If none are left, a greedy classical repair runs and `resources.extra["repaired"]` is `1.0`.
5. Uses `initial_params` as the warm start. If their depth differs from `run.reps` they are interpolated. If they are unusable (empty, NaN, odd length) they are ignored.

`max_iterations` is a budget of objective evaluations for both optimizers (SciPy COBYLA and the built-in Nelder-Mead), so the same setting limits the work identically on every machine.

`HarmonyProblem` is the solver's internal sparse form: `h[i][k]` and `W[(i, k, j, l)]` with `i < j`. `HarmonyProblem.from_dense` raises if the lower block is non-zero, because reading a symmetric W twice would silently double every penalty.

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

Run from the **repository root**. Dependencies come from the root `requirements-*.txt` files (this folder has none of its own). Imports use `person2.quantum`, never bare `quantum`, so they cannot collide with `person3/quantum`.

```bash
pytest person2/tests -v                       # everything; numpy/qiskit tests skip if missing
pytest person2/tests/test_qiskit_backend.py -v
python person2/examples/demo.py [python|qiskit]
```

`pytest.ini` lists `testpaths` without `person2/tests`, so a bare `pytest` skips this folder. Add `person2/tests` to it if you want it in the default run.

**Test status** (written in a sandbox with no numpy, pydantic, or Qiskit):

* 47 core tests passed. They need no dependencies.
* `test_adapter.py` (19 tests) passed against the real `Schema.py` and the pipeline's real `validate_result`, but with a throwaway numpy stand-in. Re-run it with real numpy.
* **Never run:** `test_qiskit_backend.py`, and the `execution=` path in `adapter.py` (transpile, `execution.sampler`, `execution.resources`). Both need Qiskit and Aer. Run them first.
* Injecting deliberate bugs (flipped ZZ sign, no boundary folding, too-small penalty, dropped long-range terms, inflated probabilities) made the suite fail each time.

## Findings on real data (Person 1's Ode to Joy example, 32 slots, K=4)

Using the pipeline's non-overlapping windows of 3 slots (12 qubits), `max_iterations=50`:

| Method | Energy gap to the true optimum |
|---|---|
| Exact solver per window (the best any solver can do with these windows) | 9.2% |
| This QAOA, `reps=1` | 9.2% |
| This QAOA, `reps=2` | 15.4% |
| Exact solver, solving 6 slots and committing 3 | 1.5% |

* Every window reached its exact window optimum at `reps=1`, 2, and 3. The differences in the totals come from which of several tied optima a window picks, which changes the boundary the next window sees. Do not read them as "higher reps is worse".
* Most of the loss comes from the window cutting in `person4/backend/pipeline.py`, which commits every window without overlap. Solving a longer window and committing only its first half would recover most of it. That file belongs to Person 4.
* These runs used the built-in Nelder-Mead optimizer because SciPy was missing. COBYLA, used when SciPy is installed, may behave differently.
* `person1/examples/ode_to_joy_chorale_problem.json` has 32 slots, but the current compiler's `slot_durations` gives 34 for Ode to Joy. The example file looks stale. Regenerate it and re-run `validate_compilation` against live output, which could not be done here.
