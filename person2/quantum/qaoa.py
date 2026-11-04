"""Plain QAOA for one window of the harmonisation problem.

Public entry point: :func:`solve_window`. It takes a :class:`~quantum.qubo.Qubo`, optimises the
QAOA angles against the expected (scaled) Ising energy, samples the final circuit, drops
samples that are not one-hot, and ranks the survivors by harmony energy.

Person 3's XY-mixer solver should expose the same ``solve(qubo, **kw) -> WindowResult``
signature so it can be passed to :func:`quantum.windowing.solve_melody` as ``solver=``.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from .optimizers import minimize
from .qubo import Qubo


@dataclass
class Sample:
    assignment: dict[int, int]  # slot -> candidate index
    energy: float  # harmony energy of the window slots (no penalty), original units
    count: int


@dataclass
class WindowResult:
    slots: list[int]
    best: dict[int, int]
    best_energy: float
    samples: list[Sample]  # feasible samples, best energy first
    feasible_fraction: float
    repaired: bool  # True if no feasible sample was drawn and ``best`` was repaired
    params: list[float]  # [gamma_1..gamma_p, beta_1..beta_p]
    p: int
    n_qubits: int
    n_evals: int
    expectation: float  # final <H_scaled>
    energy_scale: float  # divisor applied to the Ising coefficients
    penalty: float
    optimizer: str
    backend: str
    shots: int
    seconds: float
    warm_started: bool
    trace: list[float] = field(default_factory=list)  # expectation per optimizer evaluation


def linear_ramp_init(p: int, gamma_max: float = 0.8, beta_max: float = 0.6) -> list[float]:
    """Annealing-inspired start: gamma rises and beta falls linearly across the layers."""
    gam = [gamma_max * (l + 0.5) / p for l in range(p)]
    bet = [beta_max * (1 - (l + 0.5) / p) for l in range(p)]
    return gam + bet


def adapt_params(params: Sequence[float], p_old: int, p_new: int) -> list[float]:
    """Resample angles to a new depth by linear interpolation along the layer index."""
    if p_old == p_new:
        return list(params)
    if p_old < 1 or len(params) != 2 * p_old:
        raise ValueError("bad parameter vector")

    def resample(v: Sequence[float]) -> list[float]:
        if len(v) == 1:
            return [v[0]] * p_new
        out = []
        for l in range(p_new):
            t = (l + 0.5) / p_new * len(v) - 0.5
            t = min(max(t, 0.0), len(v) - 1.0)
            lo = int(math.floor(t))
            hi = min(lo + 1, len(v) - 1)
            out.append(v[lo] + (t - lo) * (v[hi] - v[lo]))
        return out

    return resample(params[:p_old]) + resample(params[p_old:])


def make_backend(qubo_ising_scaled, p: int, backend: Any = "python", **backend_options: Any):
    """Return an object with ``expectation(params)`` and ``sample(params, shots, seed)``."""
    if not isinstance(backend, str):
        return backend  # already a backend instance
    if backend == "python":
        from .simulator import PythonBackend

        return PythonBackend(qubo_ising_scaled, p)
    if backend == "qiskit":
        from .qiskit_backend import QiskitBackend

        return QiskitBackend(qubo_ising_scaled, p, **backend_options)
    raise ValueError(f"unknown backend {backend!r}; use 'python' or 'qiskit'")


def repair(qubo: Qubo, bits: Sequence[int]) -> dict[int, int]:
    """Turn a non-one-hot bitstring into a valid assignment (last resort).

    For each slot, start from the candidates that are set (or all candidates if none are),
    then greedily pick, slot by slot, the one with the lowest energy given only the slots
    already decided. This is classical post-processing and the result is flagged ``repaired``.
    """
    chosen: dict[int, int] = {}
    for slot, group in zip(qubo.slots, qubo.groups):
        pool = [qubo.var[q][1] for q in group if bits[q]] or [qubo.var[q][1] for q in group]
        best_k, best_e = pool[0], math.inf
        for k in pool:
            trial = [0] * qubo.n  # undecided slots have no bits set, so they add no energy
            for s, g in zip(qubo.slots, qubo.groups):
                if s in chosen:
                    trial[g[chosen[s]]] = 1
            trial[group[k]] = 1
            e = qubo.harmony_energy(trial)
            if e < best_e:
                best_k, best_e = k, e
        chosen[slot] = best_k
    return chosen


def solve_window(
    qubo: Qubo,
    p: int = 2,
    shots: int = 1024,
    backend: Any = "python",
    optimizer: str = "auto",
    maxiter: int = 150,
    tol: float = 1e-4,
    init_params: Sequence[float] | None = None,
    init_p: int | None = None,
    seed: int | None = 0,
    max_qubits: int = 20,
    backend_options: dict[str, Any] | None = None,
) -> WindowResult:
    """Solve one window with QAOA.

    ``init_params`` warm-starts the optimiser (angles from the previous window). If
    ``init_p`` differs from ``p`` the angles are interpolated to the new depth.
    """
    if p < 1:
        raise ValueError("p must be >= 1")
    if qubo.n > max_qubits:
        raise ValueError(f"window needs {qubo.n} qubits, above max_qubits={max_qubits}; use a smaller window or K")
    t0 = time.perf_counter()
    ising = qubo.to_ising()
    scaled, scale = ising.scaled()
    be = make_backend(scaled, p, backend, **(backend_options or {}))

    warm = init_params is not None
    if warm:
        x0 = adapt_params(init_params, init_p or p, p)
    else:
        x0 = linear_ramp_init(p)

    trace: list[float] = []

    def objective(x: Sequence[float]) -> float:
        v = be.expectation(list(x))
        trace.append(v)
        return v

    x, fval, nfev, used = minimize(objective, x0, method=optimizer, maxiter=maxiter, tol=tol)

    counts = be.sample(x, shots, seed)
    total = sum(counts.values())
    feasible: dict[tuple[int, ...], int] = {}
    for bits, c in counts.items():
        if qubo.decode(bits) is not None:
            feasible[bits] = c
    feas_frac = sum(feasible.values()) / total if total else 0.0

    repaired = False
    if feasible:
        samples = [Sample(qubo.decode(b), qubo.harmony_energy(b), c) for b, c in feasible.items()]
        samples.sort(key=lambda s: (s.energy, -s.count))
        best = samples[0]
        best_assign, best_energy = best.assignment, best.energy
    else:
        top_bits = max(counts, key=counts.get)
        best_assign = repair(qubo, top_bits)
        best_energy = qubo.harmony_energy(qubo.encode(best_assign))
        samples = [Sample(best_assign, best_energy, 0)]
        repaired = True

    return WindowResult(
        slots=list(qubo.slots),
        best=best_assign,
        best_energy=best_energy,
        samples=samples,
        feasible_fraction=feas_frac,
        repaired=repaired,
        params=[float(v) for v in x],
        p=p,
        n_qubits=qubo.n,
        n_evals=nfev,
        expectation=float(fval),
        energy_scale=scale,
        penalty=qubo.penalty,
        optimizer=used,
        backend=getattr(be, "name", type(be).__name__),
        shots=total,
        seconds=time.perf_counter() - t0,
        warm_started=warm,
        trace=trace,
    )


# Alias matching the "clean solve(window)" wording in the project plan.
solve = solve_window
