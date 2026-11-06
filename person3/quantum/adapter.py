"""Adapter between the team contract (Schema.py / person4 pipeline) and Person 3's XY-QAOA.

Wire it into the backend with::

    XY_SOLVER_CALLABLE=person3.quantum.adapter:solve

Person 2's ``person2.quantum.adapter.solve`` stays the main ``SOLVER_CALLABLE``. It runs
``qaoa`` requests itself and forwards ``xy_qaoa`` requests here, with the same arguments
(``window, *, run, execution, initial_params``) that ``person4.backend.pipeline`` passes.

How a window is solved
----------------------
1. ``CompiledProblem`` -> cost of every valid selection, including ``long_range`` terms,
   rescaled to [0, 1] over the window (the argmin is unchanged).
2. Angles are optimised on the exact simulator in ``fast_sim`` (it only tracks the valid
   selections, since the XY mixer never leaves them). ``initial_params`` from the previous
   window is tried first as a warm start, alongside fresh starts; the best is kept.
3. The final circuit (W state per slot, then cost layer + XY ring mixer, repeated p times)
   is transpiled with ``execution.transpile`` and sampled with ``execution.sampler``, and
   ``execution.resources`` reports the transpiled depth and two-qubit gates. Without an
   ``execution`` context, shots are drawn from the exact distribution.
4. On a noiseless simulator every shot is a valid selection. On noisy hardware some may not
   be; those are dropped. If none survive, shots are drawn from the noiseless distribution
   instead and ``resources.extra["repaired"] = 1.0`` says so.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

from Schema import CompiledProblem, ResourceReport, SolverResult
from Schema import Sample as SchemaSample

from . import fast_sim
from .xy_mixer import OneHotLayout, cost_layer, energy_tensor, prepare_w_state, qubo_terms, xy_ring_mixer
from .xy_solve import optimize

SOLVER_NAME = "xy_qaoa"


def cost_tensor(problem: CompiledProblem) -> np.ndarray:
    """``problem.energy`` of every selection, as an array of shape (K_0, ..., K_{N-1})."""
    E = energy_tensor(problem.h, problem.W)
    for c in problem.long_range:
        idx = [slice(None)] * problem.num_slots
        idx[c.t], idx[c.s] = c.k, c.l
        E[tuple(idx)] += float(c.weight)
    return E


def cost_terms(problem: CompiledProblem, layout: OneHotLayout, scale: float):
    """QUBO coefficients (lin, quad) of the energy, divided by ``scale``. Constants are dropped:
    they only add a global phase to the circuit."""
    lin, quad, _ = qubo_terms(problem.h, problem.W, layout)
    for c in problem.long_range:
        key = (layout.qubit(c.t, c.k), layout.qubit(c.s, c.l))
        quad[key] = quad.get(key, 0.0) + float(c.weight)
    return lin / scale, {key: w / scale for key, w in quad.items()}


def build_circuit(layout: OneHotLayout, lin, quad, p: int):
    """XY-QAOA ansatz with parameters gamma[0..p-1], beta[0..p-1] (no measurements)."""
    gammas, betas = ParameterVector("gamma", p), ParameterVector("beta", p)
    qc = QuantumCircuit(layout.n_qubits, name=f"xy_qaoa_p{p}")
    for t in range(len(layout.sizes)):
        prepare_w_state(qc, layout.slot_qubits(t))
    for r in range(p):
        cost_layer(qc, lin, quad, gammas[r])
        for t in range(len(layout.sizes)):
            xy_ring_mixer(qc, layout.slot_qubits(t), betas[r])
    return qc, gammas, betas


class _Model:
    """Exact output distribution over valid selections, for the angle optimizer."""

    def __init__(self, cost: np.ndarray) -> None:
        self.cost = cost
        self.n_evals = 0

    def assignment_probs(self, params: Sequence[float]) -> np.ndarray:
        p = len(params) // 2
        return fast_sim.xy_probabilities(self.cost, params[:p], params[p:])

    def stats(self, params: Sequence[float]) -> tuple[float, float]:
        self.n_evals += 1
        P = self.assignment_probs(params)
        return float((P * self.cost).sum()), 1.0


def _warm_start(initial_params: Sequence[float] | None, p: int) -> np.ndarray | None:
    """Previous window's angles, if they fit this depth (2p finite numbers)."""
    if not initial_params or len(initial_params) != 2 * p:
        return None
    x = np.asarray(initial_params, dtype=float)
    return x if np.all(np.isfinite(x)) else None


def _sample_exact(P: np.ndarray, shots: int, seed: int) -> dict[tuple[int, ...], int]:
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(shots, (P / P.sum()).ravel())
    return {tuple(int(i) for i in np.unravel_index(z, P.shape)): int(c)
            for z, c in enumerate(counts) if c}


def _sample_execution(execution: Any, circuit: QuantumCircuit, gammas, betas, params, layout: OneHotLayout,
                      shots: int):
    """Run on the pipeline's execution context. Returns (valid counts, invalid shots, isa circuit)."""
    isa = execution.transpile(circuit.measure_all(inplace=False))
    value = {par: float(params[i]) for i, par in enumerate(list(gammas) + list(betas))}
    values = [value[par] for par in isa.parameters]
    data = execution.sampler.run([(isa, values)], shots=shots).result()[0].data
    valid: dict[tuple[int, ...], int] = {}
    invalid = 0
    for bits, c in data.meas.get_counts().items():
        sel = layout.decode(bits)  # qubit 0 is the right-most character
        if sel is None:
            invalid += int(c)
        else:
            valid[sel] = valid.get(sel, 0) + int(c)
    return valid, invalid, isa


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
) -> SolverResult:
    """Solve one pipeline window with XY-QAOA. Keyword overrides replace the matching ``run`` fields."""

    def opt(name: str, override: Any, default: Any) -> Any:
        return override if override is not None else getattr(run, name, default)

    p = int(opt("reps", reps, 1))
    shots_ = int(opt("shots", shots, 1024))
    seed_ = int(opt("seed", seed, 42))
    iters = int(opt("max_iterations", max_iterations, 50))
    limit = int(opt("max_qubits", max_qubits, 16))

    window.validate()
    layout = OneHotLayout([window.num_candidates(t) for t in range(window.num_slots)])
    if layout.n_qubits > limit:
        raise ValueError(f"window needs {layout.n_qubits} qubits, above the limit of {limit}")

    E = cost_tensor(window)
    lo, hi = float(E.min()), float(E.max())
    scale = hi - lo if hi > lo else 1.0
    model = _Model((E - lo) / scale)

    start = _warm_start(initial_params, p)
    params, objective = optimize(model, p, init=start, n_restarts=4, seed=seed_, maxiter=iters)
    P = model.assignment_probs(params)

    repaired = False
    resources: ResourceReport | None = None
    if execution is not None:
        lin, quad = cost_terms(window, layout, scale)
        circuit, gammas, betas = build_circuit(layout, lin, quad, p)
        counts, invalid, isa = _sample_execution(execution, circuit, gammas, betas, params, layout, shots_)
        resources = execution.resources(isa, p, shots_)
        if not counts:
            counts, repaired = _sample_exact(P, shots_, seed_), True
        feasible_fraction = 0.0 if repaired else (shots_ - invalid) / shots_
    else:
        counts = _sample_exact(P, shots_, seed_)
        feasible_fraction = 1.0

    samples = sorted(
        (SchemaSample(selection=sel, energy=window.energy(sel), probability=c / shots_)
         for sel, c in counts.items()),
        key=lambda s: s.energy,
    )
    best = samples[0].selection
    if resources is not None:
        resources.extra.update({
            "optimizer_evaluations": float(model.n_evals),
            "repaired": 1.0 if repaired else 0.0,
            "warm_started": 1.0 if start is not None else 0.0,
            "final_expectation": float(objective * scale + lo),
            "p_optimum_noiseless": float(P[E <= lo + 1e-12].sum()),
        })
    return SolverResult(
        solver=SOLVER_NAME,
        best=best,
        best_energy=window.energy(best),
        samples=samples,
        feasible_fraction=float(feasible_fraction),
        optimal_params=[float(x) for x in params],
        resources=resources,
        window=(window.first_slot, window.first_slot + window.num_slots),
    )
