"""Dependency-free statevector QAOA simulator.

Used for tests and as a reference for the Qiskit circuit. It is exact but slow, so
keep windows to about 14 qubits or fewer here. Use :mod:`quantum.qiskit_backend` (Aer or
the Qiskit statevector primitives) for 16-qubit windows.

State layout: amplitude index bit ``q`` is qubit ``q`` (little-endian), the same as Qiskit.
Circuit: ``H^n``, then for each layer ``exp(-i gamma H_C)`` and ``prod_q RX_q(2 beta)``.
"""

from __future__ import annotations

import cmath
import math
import random
from typing import Sequence

from .qubo import Ising


class PythonBackend:
    name = "python-statevector"

    def __init__(self, ising: Ising, p: int) -> None:
        if ising.n > 20:
            raise ValueError("pure-Python simulator is limited to 20 qubits; use backend='qiskit'")
        self.ising, self.p, self.n = ising, p, ising.n
        size = 1 << self.n
        diag = [ising.const] * size
        for q, c in enumerate(ising.h):
            if c:
                for idx in range(size):
                    diag[idx] += -c if (idx >> q) & 1 else c
        for (q, r), c in ising.J.items():
            for idx in range(size):
                same = ((idx >> q) ^ (idx >> r)) & 1 == 0
                diag[idx] += c if same else -c
        self.diag = diag

    def _state(self, params: Sequence[float]) -> list[complex]:
        p, n = self.p, self.n
        if len(params) != 2 * p:
            raise ValueError(f"expected {2 * p} parameters (p gammas then p betas)")
        gammas, betas = params[:p], params[p:]
        size = 1 << n
        amp0 = 1.0 / math.sqrt(size)
        st = [complex(amp0)] * size
        for g, b in zip(gammas, betas):
            st = [a * cmath.exp(-1j * g * d) for a, d in zip(st, self.diag)]
            c, s = math.cos(b), -1j * math.sin(b)
            for q in range(n):
                bit = 1 << q
                for base in range(size):
                    if base & bit:
                        continue
                    a0, a1 = st[base], st[base | bit]
                    st[base] = c * a0 + s * a1
                    st[base | bit] = s * a0 + c * a1
        return st

    def expectation(self, params: Sequence[float]) -> float:
        st = self._state(params)
        return sum((a.real * a.real + a.imag * a.imag) * d for a, d in zip(st, self.diag))

    def sample(self, params: Sequence[float], shots: int, seed: int | None = None) -> dict[tuple[int, ...], int]:
        st = self._state(params)
        probs = [a.real * a.real + a.imag * a.imag for a in st]
        rng = random.Random(seed)
        counts: dict[tuple[int, ...], int] = {}
        for idx in rng.choices(range(len(probs)), weights=probs, k=shots):
            bits = tuple((idx >> q) & 1 for q in range(self.n))
            counts[bits] = counts.get(bits, 0) + 1
        return counts

    def probabilities(self, params: Sequence[float]) -> list[float]:
        return [a.real * a.real + a.imag * a.imag for a in self._state(params)]
