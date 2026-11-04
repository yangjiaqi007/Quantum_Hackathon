"""One-hot QUBO for a window of slots, and its Ising form.

Variable ``x[i, k] = 1`` means "slot ``i`` uses candidate ``k``". Qubits are numbered
slot-major: the window's first slot owns qubits ``0 .. K-1``, and so on.

Window energy::

    E(x) = sum_q lin[q] x_q + sum_{q<r} quad[q, r] x_q x_r + offset
           + A * sum_slots (sum_k x[slot, k] - 1)^2

* Slots outside the window that are already committed (``fixed``) are folded into the
  linear terms, so the boundary voicing from the previous window acts as a field.
* Slots outside the window that are not fixed (the future) are ignored.
* For feasible ``x`` the penalty term is 0, so ``E(x)`` is the harmony energy of the
  window slots, including their couplings to fixed slots.

Penalty: :func:`auto_penalty` returns a value that is provably large enough that every
ground state of the penalized QUBO is one-hot feasible. Flipping one bit changes the
harmony part by at most ``|lin_q| + sum_r |quad_qr|``. Fixing a slot with ``m != 1``
set bits by flipping one bit lowers the penalty by at least ``A``. So any ``A`` above
the largest such bound makes every infeasible state strictly improvable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from .problem import HarmonyProblem

_EPS = 1e-12


@dataclass
class Ising:
    """``E(z) = const + sum_q h[q] z_q + sum_{q<r} J[q, r] z_q z_r`` with ``z = 1 - 2 x``.

    Bit 0 maps to ``z = +1`` and bit 1 to ``z = -1``, which matches Qiskit's Z eigenvalues.
    """

    n: int
    h: list[float]
    J: dict[tuple[int, int], float]
    const: float = 0.0

    def energy(self, bits: Sequence[int]) -> float:
        z = [1 - 2 * b for b in bits]
        e = self.const
        for q, c in enumerate(self.h):
            e += c * z[q]
        for (q, r), c in self.J.items():
            e += c * z[q] * z[r]
        return e

    def scaled(self) -> tuple["Ising", float]:
        """Return ``(ising / s, s)`` with ``s = max |coefficient|``; the constant is dropped.

        QAOA angles are only comparable across windows when the Hamiltonian has a
        fixed scale, which is what makes warm-starting meaningful.
        """
        m = max([abs(c) for c in self.h] + [abs(c) for c in self.J.values()] + [0.0])
        s = m if m > _EPS else 1.0
        return Ising(self.n, [c / s for c in self.h], {k: c / s for k, c in self.J.items()}, 0.0), s


@dataclass
class Qubo:
    n: int
    base_linear: list[float]
    base_quad: dict[tuple[int, int], float]
    base_offset: float
    slots: list[int]
    groups: list[list[int]]
    var: list[tuple[int, int]]
    penalty: float
    fixed: dict[int, int] = field(default_factory=dict)

    # ------------------------------------------------------------- energies
    def harmony_energy(self, bits: Sequence[int]) -> float:
        """Energy without the one-hot penalty (meaningful for infeasible bits too)."""
        e = self.base_offset
        for q, c in enumerate(self.base_linear):
            if bits[q]:
                e += c
        for (q, r), c in self.base_quad.items():
            if bits[q] and bits[r]:
                e += c
        return e

    def penalty_energy(self, bits: Sequence[int]) -> float:
        return self.penalty * sum((sum(bits[q] for q in g) - 1) ** 2 for g in self.groups)

    def energy(self, bits: Sequence[int]) -> float:
        return self.harmony_energy(bits) + self.penalty_energy(bits)

    # ------------------------------------------------------------- feasibility
    def decode(self, bits: Sequence[int]) -> dict[int, int] | None:
        """Return ``{slot: candidate}`` if exactly one bit is set per slot, else ``None``."""
        out: dict[int, int] = {}
        for slot, g in zip(self.slots, self.groups):
            hot = [q for q in g if bits[q]]
            if len(hot) != 1:
                return None
            out[slot] = self.var[hot[0]][1]
        return out

    def encode(self, assign: Mapping[int, int]) -> list[int]:
        bits = [0] * self.n
        for slot, g in zip(self.slots, self.groups):
            bits[g[assign[slot]]] = 1
        return bits

    # ------------------------------------------------------------------ Ising
    def to_ising(self) -> Ising:
        """Ising form of the penalized QUBO, using ``x = (1 - z) / 2``."""
        lin = list(self.base_linear)
        quad = dict(self.base_quad)
        off = self.base_offset
        a = self.penalty
        # A (sum x - 1)^2 = A (1 - sum x + 2 sum_{k<l} x_k x_l) for binary x.
        for g in self.groups:
            off += a
            for q in g:
                lin[q] -= a
            for u in range(len(g)):
                for v in range(u + 1, len(g)):
                    quad[(g[u], g[v])] = quad.get((g[u], g[v]), 0.0) + 2.0 * a
        h = [0.0] * self.n
        J: dict[tuple[int, int], float] = {}
        const = off
        for q, c in enumerate(lin):
            const += c / 2.0
            h[q] -= c / 2.0
        for (q, r), c in quad.items():
            const += c / 4.0
            h[q] -= c / 4.0
            h[r] -= c / 4.0
            J[(q, r)] = J.get((q, r), 0.0) + c / 4.0
        h = [c if abs(c) > _EPS else 0.0 for c in h]
        J = {k: c for k, c in J.items() if abs(c) > _EPS}
        return Ising(self.n, h, J, const)


def build_qubo(
    problem: HarmonyProblem,
    slots: Sequence[int],
    fixed: Mapping[int, int] | None = None,
    penalty: float | None = None,
    penalty_factor: float = 1.0,
) -> Qubo:
    """Build the one-hot QUBO for ``slots`` given already committed ``fixed`` slots."""
    fixed = dict(fixed or {})
    slots = list(slots)
    if len(set(slots)) != len(slots):
        raise ValueError("slots must be unique")
    if sorted(slots) != slots:
        raise ValueError("slots must be in increasing order")
    for s in slots:
        if not 0 <= s < problem.n_slots:
            raise ValueError(f"slot {s} out of range")
        if s in fixed:
            raise ValueError(f"slot {s} is both in the window and fixed")
    for s, k in fixed.items():
        if not (0 <= s < problem.n_slots and 0 <= k < len(problem.h[s])):
            raise ValueError(f"fixed assignment {s}->{k} out of range")

    var: list[tuple[int, int]] = []
    groups: list[list[int]] = []
    index: dict[tuple[int, int], int] = {}
    for s in slots:
        g = []
        for k in range(len(problem.h[s])):
            index[(s, k)] = len(var)
            g.append(len(var))
            var.append((s, k))
        groups.append(g)
    n = len(var)

    lin = [problem.h[s][k] for (s, k) in var]
    quad: dict[tuple[int, int], float] = {}
    for (i, k, j, l), w in problem.W.items():
        in_i, in_j = i in slots_set(slots), j in slots_set(slots)
        if in_i and in_j:
            q, r = index[(i, k)], index[(j, l)]
            quad[(q, r)] = quad.get((q, r), 0.0) + w
        elif in_i and fixed.get(j) == l:
            lin[index[(i, k)]] += w
        elif in_j and fixed.get(i) == k:
            lin[index[(j, l)]] += w

    if penalty is None:
        penalty = auto_penalty(lin, quad, n) * penalty_factor
    return Qubo(n, lin, quad, 0.0, slots, groups, var, float(penalty), fixed)


def slots_set(slots: Sequence[int]) -> frozenset[int]:
    return frozenset(slots)


def auto_penalty(linear: Sequence[float], quad: Mapping[tuple[int, int], float], n: int) -> float:
    """Smallest safe one-hot penalty (plus a tiny margin), see module docstring."""
    row = [abs(c) for c in linear]
    for (q, r), c in quad.items():
        row[q] += abs(c)
        row[r] += abs(c)
    bound = max(row) if row else 0.0
    return bound + 1e-6 if bound > 0 else 1.0
