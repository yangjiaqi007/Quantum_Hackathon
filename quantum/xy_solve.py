"""Optimize and sample XY-mixer QAOA (and the plain-QAOA baseline) on one window.

    res = xy_solve(h, W, p=3, shots=2000)
    res.best          # lowest-energy assignment that was sampled
    res.samples       # list of assignments, one per shot (all feasible for XY)
    res.params        # optimized [gammas..., betas...], pass as init= to warm-start
                      # the next window (Person 2's windowing)

Angles are optimized on the exact simulator in fast_sim.py (the objective is
the expected energy), then the bound Qiskit circuit is sampled with Aer shots.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from quantum import fast_sim
from quantum.xy_mixer import (OneHotLayout, energy_tensor, qubo_terms,
                              x_qaoa_circuit, xy_qaoa_circuit)


@dataclass
class Normalized:
    """Energies rescaled to [0, 1] over the window: E' = (E - offset) / scale.

    QAOA angles only make sense relative to the energy scale, so we always
    optimize on E'. The argmin is unchanged.
    """
    h: list
    W: list
    offset: float
    scale: float


def normalize(h, W):
    E = energy_tensor(h, W)
    lo, hi = float(E.min()), float(E.max())
    scale = hi - lo if hi > lo else 1.0
    # spread the offset over slot 0's unary terms so E' = (E - lo) / scale exactly
    h2 = [np.asarray(x, dtype=float) / scale for x in h]
    h2[0] = h2[0] - lo / scale
    W2 = [np.asarray(w, dtype=float) / scale for w in W]
    return Normalized([x.tolist() for x in h2], [w.tolist() for w in W2], lo, scale)


# ---------------------------------------------------------------- exact objective

class XYModel:
    """Exact output distribution of XY-QAOA over assignments (optionally dephased)."""
    mixer = "xy"

    def __init__(self, h, W, dephasing=0.0):
        self.cost = energy_tensor(h, W)
        self.dephasing = dephasing
        self.sizes = self.cost.shape

    def assignment_probs(self, params):
        p = len(params) // 2
        g, b = params[:p], params[p:]
        if self.dephasing:
            return fast_sim.xy_probabilities_dephased(self.cost, g, b, self.dephasing)
        return fast_sim.xy_probabilities(self.cost, g, b)

    def stats(self, params):
        """(expected energy, P(valid)) - for XY, P(valid) is 1 by construction."""
        P = self.assignment_probs(params)
        return float((P * self.cost).sum()), float(P.sum())


class XModel:
    """Exact output distribution of plain QAOA with one-hot penalty."""
    mixer = "x"

    def __init__(self, h, W, penalty):
        self.layout = OneHotLayout.from_h(h)
        lin, quad, const = qubo_terms(h, W, self.layout, penalty)
        self.diag = fast_sim.qubo_diagonal(lin, quad, const, self.layout.n_qubits)
        self.cost = energy_tensor(h, W)
        self.sizes = self.cost.shape
        idx = np.array(np.unravel_index(np.arange(self.cost.size), self.sizes)).T
        self.valid_index = np.array([self.layout.encode(a) for a in idx])

    def bitstring_probs(self, params):
        p = len(params) // 2
        return fast_sim.x_probabilities(self.diag, self.layout.n_qubits, params[:p], params[p:])

    def assignment_probs(self, params):
        """Probability of each valid assignment (sums to P(valid), not 1)."""
        return self.bitstring_probs(params)[self.valid_index].reshape(self.sizes)

    def stats(self, params):
        probs = self.bitstring_probs(params)
        return float(probs @ self.diag), float(probs[self.valid_index].sum())


def expected_objective(model, params):
    """What the optimizer minimizes: <H> including the penalty for plain QAOA."""
    return model.stats(params)[0]


def optimize(model, p, init=None, n_restarts=6, seed=0, maxiter=400):
    """Minimize expected energy over 2p angles.

    init: warm start (from the previous window, or interp_angles from depth p-1).
    Without init, a coarse p=1 grid seeds a linear ramp (an annealing-like
    schedule), plus random restarts. Returns (best params, best objective).
    """
    rng = np.random.default_rng(seed)
    starts = []
    if init is not None:
        starts.append(np.asarray(init, dtype=float))
    else:
        grid = [(g, b) for g in np.linspace(0.2, 6.0, 15) for b in np.linspace(0.1, 1.5, 8)]
        g0, b0 = min(grid, key=lambda gb: expected_objective(model, np.array(gb)))
        ramp = (np.arange(p) + 0.5) / p
        starts.append(np.concatenate([g0 * ramp, b0 * (1 - ramp) + 0.05]))
    for _ in range(n_restarts - len(starts)):
        starts.append(np.concatenate([rng.uniform(0, 2 * np.pi, p), rng.uniform(0, np.pi / 2, p)]))

    best_x, best_f = None, np.inf
    for x0 in starts:
        r = minimize(lambda x: expected_objective(model, x), x0, method="BFGS",
                     options={"maxiter": maxiter})
        if r.fun < best_f:
            best_x, best_f = r.x, float(r.fun)
    return best_x, best_f


def interp_angles(params):
    """INTERP warm start (Zhou et al., PRX 2020): angles for depth p+1 from depth p."""
    p = len(params) // 2
    out = []
    for v in (params[:p], params[p:]):
        v = np.concatenate([[0.0], v, [0.0]])
        out.append([(i / p) * v[i] + (1 - i / p) * v[i + 1] for i in range(p + 1)])
    return np.concatenate(out)


def depth_sweep(model, p_max, n_restarts=6, seed=0):
    """Optimized params for p = 1..p_max, each depth warm-started by INTERP and
    also given fresh restarts, keeping whichever is better."""
    out = {}
    prev = None
    for p in range(1, p_max + 1):
        x, f = optimize(model, p, n_restarts=n_restarts, seed=seed + p)
        if prev is not None:
            xi, fi = optimize(model, p, init=interp_angles(prev), n_restarts=1)
            if fi < f:
                x, f = xi, fi
        out[p] = x
        prev = x
    return out


# ---------------------------------------------------------------- sampling

@dataclass
class SolveResult:
    best: tuple
    best_energy: float
    samples: list           # feasible assignments, one per surviving shot
    params: np.ndarray
    p: int
    mixer: str
    shots: int
    n_infeasible: int
    info: dict = field(default_factory=dict)


def sample_circuit(qaoa, params, shots, backend=None, seed=None):
    """Run the bound circuit on Aer (or any backend with .run) and decode counts."""
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    backend = backend or AerSimulator(seed_simulator=seed)
    qc = qaoa.bind(params)
    qc.measure_all()
    counts = backend.run(transpile(qc, backend), shots=shots).result().get_counts()
    samples, bad = [], 0
    for bits, c in counts.items():
        a = qaoa.layout.decode(bits)
        if a is None:
            bad += c
        else:
            samples.extend([a] * c)
    return samples, bad


def _solve(model, circuit_fn, h, W, p, shots, init, backend, seed):
    norm = normalize(h, W)
    params, _ = optimize(model, p, init=init, seed=seed)
    qaoa = circuit_fn(norm.h, norm.W, p)
    samples, bad = sample_circuit(qaoa, params, shots, backend, seed)
    if not samples:
        raise RuntimeError(f"all {shots} shots were infeasible; raise shots or the penalty")
    E = energy_tensor(h, W)
    best = min(set(samples), key=lambda a: E[a])
    return SolveResult(best, float(E[best]), samples, params, p, model.mixer, shots, bad,
                       info={"offset": norm.offset, "scale": norm.scale})


def xy_solve(h, W, p=3, shots=2000, init=None, backend=None, seed=None):
    """Constraint-preserving QAOA on one window. Every shot is a valid assignment."""
    norm = normalize(h, W)
    return _solve(XYModel(norm.h, norm.W), xy_qaoa_circuit, h, W, p, shots, init, backend, seed)


def x_solve(h, W, p=3, penalty=1.0, shots=2000, init=None, backend=None, seed=None):
    """Plain-QAOA baseline. penalty is in normalized units (energy range = 1)."""
    norm = normalize(h, W)
    circuit_fn = lambda hh, WW, pp: x_qaoa_circuit(hh, WW, pp, penalty)
    return _solve(XModel(norm.h, norm.W, penalty), circuit_fn, h, W, p, shots, init, backend, seed)
