"""Adapter between the team contract (Schema.py / person4 pipeline) and the Person 2 QAOA core.

Wire it into the backend with::

    PIPELINE_MODE=team
    SOLVER_CALLABLE=person2.quantum.adapter:solve

``solve(window, *, run, execution, initial_params) -> SolverResult`` is the signature that
``person4.backend.pipeline.run_pipeline`` calls once per window. The pipeline already cuts
the melody with ``CompiledProblem.window`` and folds the boundary into ``h``, so this module
solves each window as a standalone problem (no ``fixed`` slots).

How a window is solved
----------------------
1. ``CompiledProblem`` -> ``HarmonyProblem`` (``h``, adjacent ``W``, and ``long_range`` terms).
2. Angles are optimised on an *exact* local simulator (pure Python up to 12 qubits, Qiskit
   statevector above that). Expectation values are exact, so the optimiser sees no shot noise.
3. The final circuit is sampled. With an ``execution`` context (Aer, fake noisy device, IBM) the
   circuit is transpiled through ``execution.transpile`` and sampled with ``execution.sampler``,
   and ``execution.resources`` supplies the depth and two-qubit gate count of the transpiled
   circuit. Without one, sampling uses the exact local simulator and ``resources`` is ``None``.
4. Samples that are not one-hot are dropped. If none survive, a greedy classical repair runs and
   the result carries ``resources.extra["repaired"] = 1.0`` (when resources exist) so reports can
   say the answer was not a clean quantum sample.

Solver routing
--------------
The pipeline has a single ``SOLVER_CALLABLE`` but a request can ask for ``qaoa`` or ``xy_qaoa``.
``solve`` handles ``qaoa`` itself and forwards ``xy_qaoa`` to the callable named by the
``XY_SOLVER_CALLABLE`` environment variable (``module:function``, same signature as ``solve``).
Person 3 owns that callable; it must return ``Schema.SolverResult`` with ``solver="xy_qaoa"``.
If the variable is unset, requests for ``xy_qaoa`` fail with a message saying so.
"""

from __future__ import annotations

import math
import os
from importlib import import_module
from typing import Any, Sequence

from Schema import CompiledProblem, ResourceReport, SolverResult
from Schema import Sample as SchemaSample

from .problem import HarmonyProblem
from .qaoa import WindowResult, solve_window
from .qubo import build_qubo

LOCAL_PYTHON_QUBIT_LIMIT = 12  # above this the exact local simulator switches to Qiskit
XY_SOLVER_ENV = "XY_SOLVER_CALLABLE"


def _public_error(message: str) -> Exception:
    """The pipeline's PipelineError (message is safe to show users) when available."""
    try:
        from person4.backend.contracts import PipelineError

        return PipelineError(message)
    except ImportError:
        return ValueError(message)


def _delegate_xy(window: CompiledProblem, run: Any, execution: Any, initial_params: Any) -> SolverResult:
    ref = os.getenv(XY_SOLVER_ENV, "").strip()
    if not ref:
        raise _public_error(
            f"The xy_qaoa solver is not connected. Set {XY_SOLVER_ENV}=module:function to the "
            "xy_qaoa adapter that returns Schema.SolverResult, or request solver 'qaoa'."
        )
    try:
        module, _, name = ref.partition(":")
        fn = getattr(import_module(module), name)
        if not callable(fn):
            raise TypeError(name)
    except (ImportError, AttributeError, TypeError, ValueError) as exc:
        raise _public_error(f"{XY_SOLVER_ENV} does not point to a callable solver. Check the server configuration.") from exc
    return fn(window, run=run, execution=execution, initial_params=initial_params)


def to_harmony_problem(problem: CompiledProblem) -> HarmonyProblem:
    """Convert the shared ``CompiledProblem`` into the solver's sparse ``h`` / ``W`` form."""
    h = [[float(x) for x in row] for row in problem.h]
    W: dict[tuple[int, int, int, int], float] = {}
    for t, block in enumerate(problem.W):  # block[k][l]: slot t -> slot t + 1
        for k in range(len(h[t])):
            for l in range(len(h[t + 1])):
                w = float(block[k][l])
                if w != 0.0:
                    W[(t, k, t + 1, l)] = W.get((t, k, t + 1, l), 0.0) + w
    for c in problem.long_range:
        key = (c.t, c.k, c.s, c.l)
        W[key] = W.get(key, 0.0) + float(c.weight)
    return HarmonyProblem(h, W)


class _ExecutionBackend:
    """Exact local expectation for the optimiser, ``execution.sampler`` for the final shots."""

    def __init__(self, ising, p: int, execution: Any) -> None:
        from .qiskit_backend import build_qaoa_circuit

        self.n, self.p, self.execution = ising.n, p, execution
        self.name = getattr(execution, "backend_name", "execution")
        if ising.n <= LOCAL_PYTHON_QUBIT_LIMIT:
            from .simulator import PythonBackend

            self._exact = PythonBackend(ising, p)
        else:
            from .qiskit_backend import QiskitBackend

            self._exact = QiskitBackend(ising, p)
        circuit, gammas, betas = build_qaoa_circuit(ising, p)
        self._order = list(gammas) + list(betas)
        self.isa_circuit = execution.transpile(circuit.measure_all(inplace=False))

    def expectation(self, params: Sequence[float]) -> float:
        return self._exact.expectation(params)

    def sample(self, params: Sequence[float], shots: int, seed: int | None = None) -> dict[tuple[int, ...], int]:
        pos = {par: i for i, par in enumerate(self._order)}
        values = [float(params[pos[par]]) for par in self.isa_circuit.parameters]
        data = self.execution.sampler.run([(self.isa_circuit, values)], shots=shots).result()[0].data
        out: dict[tuple[int, ...], int] = {}
        for key, c in data.meas.get_counts().items():
            key = key.replace(" ", "")
            bits = tuple(int(key[self.n - 1 - q]) for q in range(self.n))  # qubit 0 is right-most
            out[bits] = out.get(bits, 0) + int(c)
        return out

    def resources(self, reps: int, shots: int) -> ResourceReport:
        return self.execution.resources(self.isa_circuit, reps, shots)


def _usable_start(initial_params: Sequence[float] | None) -> list[float] | None:
    if not initial_params or len(initial_params) % 2 or not all(math.isfinite(x) for x in initial_params):
        return None
    return [float(x) for x in initial_params]


def to_solver_result(
    problem: CompiledProblem,
    res: WindowResult,
    resources: ResourceReport | None,
) -> SolverResult:
    """Package a ``WindowResult`` as the contract's ``SolverResult``.

    Energies come from ``problem.energy`` so they match the pipeline's validation exactly.
    """
    n = problem.num_slots
    best = tuple(int(res.best[s]) for s in range(n))
    total = max(res.shots, 1)
    samples = []
    for s in res.samples:
        sel = tuple(int(s.assignment[i]) for i in range(n))
        samples.append(SchemaSample(selection=sel, energy=problem.energy(sel), probability=s.count / total))
    samples.sort(key=lambda s: s.energy)
    if resources is not None:
        resources.extra.update(
            {
                "optimizer_evaluations": float(res.n_evals),
                "repaired": 1.0 if res.repaired else 0.0,
                "warm_started": 1.0 if res.warm_started else 0.0,
                "final_expectation": float(res.expectation),
            }
        )
    return SolverResult(
        solver="qaoa",
        best=best,
        best_energy=problem.energy(best),
        samples=samples,
        feasible_fraction=float(res.feasible_fraction),
        optimal_params=list(res.params),
        resources=resources,
        window=(problem.first_slot, problem.first_slot + n),
    )


def solve(
    window: CompiledProblem,
    *,
    run: Any = None,
    execution: Any = None,
    initial_params: Sequence[float] | None = None,
    reps: int | None = None,
    shots: int | None = None,
    seed: int | None = None,
    max_iterations: int | None = None,
    max_qubits: int | None = None,
    optimizer: str = "auto",
) -> SolverResult:
    """Solve one pipeline window. ``qaoa`` runs here; ``xy_qaoa`` is forwarded (see module docs).

    Keyword overrides (``reps``, ``shots``, ...) replace the matching ``run`` fields.
    """

    def opt(name: str, override: Any, default: Any) -> Any:
        return override if override is not None else getattr(run, name, default)

    solver = getattr(run, "solver", "qaoa")
    if solver == "xy_qaoa":
        return _delegate_xy(window, run, execution, initial_params)
    if solver != "qaoa":
        raise _public_error(f"Unknown solver {solver!r}; expected 'qaoa' or 'xy_qaoa'.")
    p = opt("reps", reps, 1)
    shots_ = opt("shots", shots, 1024)
    seed_ = opt("seed", seed, 42)
    iters = opt("max_iterations", max_iterations, 50)
    limit = opt("max_qubits", max_qubits, 16)

    window.validate()
    qubo = build_qubo(to_harmony_problem(window), list(range(window.num_slots)))
    if qubo.n > limit:
        raise ValueError(f"window needs {qubo.n} qubits, above the limit of {limit}")

    ising_scaled = qubo.to_ising().scaled()[0]
    if execution is not None:
        backend: Any = _ExecutionBackend(ising_scaled, p, execution)
    elif qubo.n <= LOCAL_PYTHON_QUBIT_LIMIT:
        backend = "python"
    else:
        backend = "qiskit"

    start = _usable_start(initial_params)
    res = solve_window(
        qubo,
        p=p,
        shots=shots_,
        backend=backend,
        optimizer=optimizer,
        maxiter=iters,
        init_params=start,
        init_p=len(start) // 2 if start else None,
        seed=seed_,
        max_qubits=max(limit, qubo.n),
    )
    resources = backend.resources(p, shots_) if isinstance(backend, _ExecutionBackend) else None
    return to_solver_result(window, res, resources)
