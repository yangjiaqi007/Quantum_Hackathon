"""Constraint-preserving QAOA circuits (W-state start + XY ring mixer).

Input contract (from compiler/):
    h[t][k]     unary cost of voicing k at slot t
    W[t][k][l]  cost of voicing k at slot t followed by voicing l at slot t+1
    E(a) = sum_t h[t][a_t] + sum_t W[t][a_t][a_{t+1}]

Encoding: one-hot. Qubit layout.qubit(t, k) = 1 means "slot t uses voicing k"
(the same as the x_{t}_{k} variables in Person 2's QuadraticProgram).

Plain QAOA starts in |+>^n and mixes with X, so it explores all 2^n bitstrings
and needs a penalty P * sum_t (sum_k x_tk - 1)^2 to discourage invalid ones.
The XY version instead:
  * starts each slot in a W state (one excitation, uniform over the K voicings),
  * mixes with (XX + YY)/2 between neighbouring qubits of a slot's ring, which
    moves the excitation around but never creates or destroys one.
Every slot therefore always holds exactly one voicing. With no penalty term,
the cost layer only needs the true energy.

Qubit order follows Qiskit: qubit q is bit q of the basis-state index, and
the rightmost character of a counts bitstring.
"""
from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import XXPlusYYGate


class OneHotLayout:
    """Maps (slot t, voicing k) to a qubit index. Slots may have different sizes."""

    def __init__(self, sizes):
        self.sizes = tuple(int(s) for s in sizes)
        if not self.sizes or min(self.sizes) < 1:
            raise ValueError(f"every slot needs at least one candidate, got sizes {self.sizes}")
        self.offsets = tuple(int(x) for x in np.cumsum((0,) + self.sizes[:-1]))
        self.n_qubits = sum(self.sizes)

    @classmethod
    def from_h(cls, h):
        return cls([len(x) for x in h])

    def qubit(self, t, k):
        return self.offsets[t] + k

    def slot_qubits(self, t):
        return list(range(self.offsets[t], self.offsets[t] + self.sizes[t]))

    def encode(self, assignment):
        """Assignment (one voicing index per slot) -> basis-state index."""
        return sum(1 << self.qubit(t, k) for t, k in enumerate(assignment))

    def decode(self, bits):
        """Basis-state index or Qiskit bitstring -> assignment, or None if some
        slot has zero or several voicings selected."""
        z = int(bits.replace(" ", ""), 2) if isinstance(bits, str) else int(bits)
        a = []
        for t, size in enumerate(self.sizes):
            ones = [k for k in range(size) if (z >> (self.offsets[t] + k)) & 1]
            if len(ones) != 1:
                return None
            a.append(ones[0])
        return tuple(a)


def check_shapes(h, W):
    if len(W) != len(h) - 1:
        raise ValueError(f"need len(W) == len(h) - 1, got {len(W)} and {len(h)}")
    for t, w in enumerate(W):
        shape = np.shape(w)
        if shape != (len(h[t]), len(h[t + 1])):
            raise ValueError(f"W[{t}] has shape {shape}, expected {(len(h[t]), len(h[t + 1]))}")


def energy(h, W, a):
    return float(sum(h[t][a[t]] for t in range(len(h)))
                 + sum(W[t][a[t]][a[t + 1]] for t in range(len(W))))


def energy_tensor(h, W):
    """E[a_0, ..., a_{N-1}] for every assignment. Window-sized problems only."""
    check_shapes(h, W)
    E = np.asarray(h[0], dtype=float)
    for t, w in enumerate(W):
        w = np.asarray(w, dtype=float)
        E = (E[..., None] + w.reshape((1,) * t + w.shape)
             + np.asarray(h[t + 1], dtype=float).reshape((1,) * (t + 1) + (-1,)))
    return E


def qubo_terms(h, W, layout, penalty=0.0):
    """Energy as a QUBO over the one-hot qubits: const + lin.x + sum quad[i,j] x_i x_j.

    penalty > 0 adds P * sum_t (sum_k x_tk - 1)^2 (plain QAOA only). On valid
    bitstrings the value equals E(a) whatever the penalty.
    """
    lin = np.zeros(layout.n_qubits)
    quad = {}
    for t, row in enumerate(h):
        for k, c in enumerate(row):
            lin[layout.qubit(t, k)] += c
    for t, w in enumerate(W):
        for k, row in enumerate(w):
            for l, c in enumerate(row):
                if c != 0:
                    key = (layout.qubit(t, k), layout.qubit(t + 1, l))
                    quad[key] = quad.get(key, 0.0) + float(c)
    const = 0.0
    if penalty:
        # (sum x - 1)^2 = 1 - sum x + 2 sum_{i<j} x_i x_j   (x^2 = x for bits)
        for t in range(len(layout.sizes)):
            qs = layout.slot_qubits(t)
            for i, qi in enumerate(qs):
                lin[qi] -= penalty
                for qj in qs[i + 1:]:
                    quad[(qi, qj)] = quad.get((qi, qj), 0.0) + 2 * penalty
            const += penalty
    return lin, quad, const


# ---------------------------------------------------------------- building blocks

def prepare_w_state(qc, qubits):
    """Uniform W state (|10..0> + |01..0> + ... + |0..01>) / sqrt(K) with real amplitudes.

    Put the excitation on the first qubit, then each XX+YY rotation keeps
    1/(remaining) of the weight and passes the rest one qubit along. Each pass
    adds a factor -i, which the final phase gates remove.
    """
    K = len(qubits)
    qc.x(qubits[0])
    for i in range(K - 1):
        beta = np.arccos(np.sqrt(1.0 / (K - i)))
        qc.append(XXPlusYYGate(2 * beta), [qubits[i], qubits[i + 1]])
    for j in range(1, K):
        qc.p(j * np.pi / 2, qubits[j])


def ring_edge_layers(K):
    """Edges of a K-ring grouped into layers of disjoint pairs (parity ring mixer).

    Even K needs 2 layers, odd K needs 3. The fast simulator applies the same
    layers in the same order, so circuit and simulator agree exactly.
    """
    if K < 2:
        return []
    edges = [(0, 1)] if K == 2 else [(i, (i + 1) % K) for i in range(K)]
    layers = []
    for e in edges:
        for layer in layers:
            if all(not set(e) & set(f) for f in layer):
                layer.append(e)
                break
        else:
            layers.append([e])
    return layers


def xy_ring_mixer(qc, qubits, beta):
    """exp(-i beta (XX+YY)/2) on every ring edge of one slot."""
    for layer in ring_edge_layers(len(qubits)):
        for i, j in layer:
            qc.append(XXPlusYYGate(2 * beta), [qubits[i], qubits[j]])


def cost_layer(qc, lin, quad, gamma):
    """exp(-i gamma C) for C = lin.x + sum quad x_i x_j, up to a global phase.

    With x = (1 - Z)/2:  c x  -> RZ(-gamma c),
                         w x_i x_j -> RZ(-gamma w/2) on i and j, RZZ(gamma w/2).
    """
    eff = np.array(lin, dtype=float)
    for (i, j), w in quad.items():
        eff[i] += w / 2
        eff[j] += w / 2
    for q, c in enumerate(eff):
        if c != 0:
            qc.rz(-gamma * c, q)
    for (i, j), w in quad.items():
        if w != 0:
            qc.rzz(gamma * w / 2, i, j)


# ---------------------------------------------------------------- full ansatz

@dataclass
class QAOACircuit:
    circuit: QuantumCircuit
    gammas: ParameterVector
    betas: ParameterVector
    layout: OneHotLayout
    mixer: str

    @property
    def p(self):
        return len(self.gammas)

    def bind(self, params):
        """params = [gamma_1..gamma_p, beta_1..beta_p]."""
        params = np.asarray(params, dtype=float)
        return self.circuit.assign_parameters(
            {self.gammas: params[:self.p], self.betas: params[self.p:]})


def xy_qaoa_circuit(h, W, p, measure=False):
    """Constraint-preserving QAOA. Pass normalized h, W (see xy_solve) for sane angles."""
    check_shapes(h, W)
    layout = OneHotLayout.from_h(h)
    lin, quad, _ = qubo_terms(h, W, layout)
    gammas, betas = ParameterVector("gamma", p), ParameterVector("beta", p)
    qc = QuantumCircuit(layout.n_qubits, name=f"xy_qaoa_p{p}")
    for t in range(len(h)):
        prepare_w_state(qc, layout.slot_qubits(t))
    for r in range(p):
        cost_layer(qc, lin, quad, gammas[r])
        for t in range(len(h)):
            xy_ring_mixer(qc, layout.slot_qubits(t), betas[r])
    if measure:
        qc.measure_all()
    return QAOACircuit(qc, gammas, betas, layout, "xy")


def x_qaoa_circuit(h, W, p, penalty, measure=False):
    """Plain QAOA baseline (|+>^n start, X mixer, one-hot penalty), for comparisons."""
    check_shapes(h, W)
    layout = OneHotLayout.from_h(h)
    lin, quad, _ = qubo_terms(h, W, layout, penalty)
    gammas, betas = ParameterVector("gamma", p), ParameterVector("beta", p)
    qc = QuantumCircuit(layout.n_qubits, name=f"x_qaoa_p{p}")
    qc.h(range(layout.n_qubits))
    for r in range(p):
        cost_layer(qc, lin, quad, gammas[r])
        qc.rx(2 * betas[r], range(layout.n_qubits))
    if measure:
        qc.measure_all()
    return QAOACircuit(qc, gammas, betas, layout, "x")
