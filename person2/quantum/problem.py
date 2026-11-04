"""The h / W contract produced by Person 1's compiler and consumed by the quantum core.

Definitions
-----------
* A melody is split into ``N`` slots (one chord per slot).
* Slot ``i`` has ``K_i`` pruned candidate voicings, indexed ``0 .. K_i - 1``.
* ``h[i][k]``              unary cost of choosing candidate ``k`` in slot ``i``
                           (consonance, spacing, range, voice crossing, ...).
* ``W[(i, k, j, l)]``      pairwise cost of choosing ``k`` in slot ``i`` AND ``l`` in
                           slot ``j``. Only ``i < j`` is stored (parallel fifths and
                           octaves, leaps, leading-tone resolution, ...). ``j`` does
                           not have to be ``i + 1``.

Lower energy means better harmony. The total energy of a full assignment ``a`` is::

    E(a) = sum_i h[i][a[i]] + sum_{i<j} W[(i, a[i], j, a[j])]

Dense input (nested lists or a numpy array of shape ``(N, K, N, K)``) is accepted
through :meth:`HarmonyProblem.from_dense`. Only the ``i < j`` entries are read, and
non-zero entries with ``i >= j`` raise an error unless ``symmetric=True``. This is
deliberate: a symmetric W that is read twice would silently double every coupling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

WKey = tuple[int, int, int, int]


@dataclass
class HarmonyProblem:
    h: list[list[float]]
    W: dict[WKey, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.h:
            raise ValueError("h must contain at least one slot")
        self.h = [[float(x) for x in row] for row in self.h]
        for i, row in enumerate(self.h):
            if not row:
                raise ValueError(f"slot {i} has no candidates")
        clean: dict[WKey, float] = {}
        for (i, k, j, l), w in self.W.items():
            if not (0 <= i < j < self.n_slots):
                raise ValueError(f"W key {(i, k, j, l)} must satisfy 0 <= i < j < n_slots")
            if not (0 <= k < len(self.h[i]) and 0 <= l < len(self.h[j])):
                raise ValueError(f"W key {(i, k, j, l)} has a candidate index out of range")
            if w != 0:
                clean[(i, k, j, l)] = clean.get((i, k, j, l), 0.0) + float(w)
        self.W = clean

    # ------------------------------------------------------------------ shape
    @property
    def n_slots(self) -> int:
        return len(self.h)

    @property
    def sizes(self) -> list[int]:
        return [len(r) for r in self.h]

    # ------------------------------------------------------------ construction
    @classmethod
    def from_dense(cls, h: Sequence[Sequence[float]], W: Any, *, symmetric: bool = False) -> "HarmonyProblem":
        """Build from dense ``W[i][k][j][l]`` (lists or numpy). See module docstring."""
        n = len(h)
        sizes = [len(r) for r in h]
        sparse: dict[WKey, float] = {}
        for i in range(n):
            for k in range(sizes[i]):
                for j in range(n):
                    for l in range(sizes[j]):
                        w = float(W[i][k][j][l])
                        if w == 0.0:
                            continue
                        if i < j:
                            sparse[(i, k, j, l)] = w
                        elif not symmetric:
                            raise ValueError(
                                f"W[{i}][{k}][{j}][{l}] = {w} is non-zero but i >= j. "
                                "Only the i < j block is read; pass symmetric=True if W is "
                                "symmetric and the lower block should be ignored."
                            )
        return cls(h=[list(r) for r in h], W=sparse)

    # ----------------------------------------------------------------- energy
    def pair(self, i: int, k: int, j: int, l: int) -> float:
        """Coupling between (i, k) and (j, l) in either argument order."""
        if i < j:
            return self.W.get((i, k, j, l), 0.0)
        if i > j:
            return self.W.get((j, l, i, k), 0.0)
        return 0.0

    def partial_energy(self, assign: Mapping[int, int]) -> float:
        """Energy of the slots present in ``assign`` (unary terms and couplings among them)."""
        e = 0.0
        for i, k in assign.items():
            e += self.h[i][k]
        for (i, k, j, l), w in self.W.items():
            if assign.get(i) == k and assign.get(j) == l:
                e += w
        return e

    def energy(self, assign: Sequence[int]) -> float:
        if len(assign) != self.n_slots:
            raise ValueError(f"assignment has {len(assign)} entries, expected {self.n_slots}")
        return self.partial_energy(dict(enumerate(assign)))
