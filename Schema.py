"""
schema.py: the shared data contract between the compiler (Person 1) and the
quantum team (Persons 2 and 3).

This file contains DATA DEFINITIONS AND SMALL HELPERS ONLY. No music-theory
rules live here (those go in compiler/rules.py). Change this file only by
agreement with the whole team.

=============================== CONVENTIONS ================================

Voice order (everywhere, never changes):
    index 0 = BASS, 1 = TENOR, 2 = ALTO, 3 = SOPRANO
    Voicing.pitches = (bass, tenor, alto, soprano)

Pitches: MIDI note numbers (middle C = 60).

Time slots: a melody is cut into N slots (one chord per slot). Slot t has
K_t candidate voicings. K_t may differ between slots (ragged), so there is NO
padding: use problem.num_candidates(t).

Indexing:
    candidates[t][k]   -> Voicing k at slot t
    h[t]               -> 1-D float array, shape (K_t,)
                          h[t][k] = unary cost of choosing candidate k at slot t
    W[t]               -> 2-D float array, shape (K_t, K_{t+1}), for t = 0..N-2
                          W[t][k][l] = cost of going from candidate k at slot t
                                       to candidate l at slot t+1
    long_range         -> list of LongRangeCoupling for rules that involve
                          NON-adjacent slots (e.g. "bar 1 and bar 5 voiced
                          identically"). Empty for the basic rule set.

Energy of a selection x = (x_0, ..., x_{N-1}), where x_t is a candidate index:
    E(x) = sum_t h[t][x_t]
         + sum_t W[t][x_t][x_{t+1}]
         + sum over long_range c with x_{c.t} == c.k and x_{c.s} == c.l of c.weight
Lower energy = better harmonization. Implemented once in
CompiledProblem.energy() so everyone computes it identically.

Normalization: all costs in h, W and long_range are already NORMALIZED
(roughly in [0, 1] per term; the largest single-term cost is ~1.0).
To recover raw weighted-violation units:  raw_energy = energy * energy_scale.

One-hot constraint: exactly one candidate must be chosen per slot. This is NOT
encoded in h/W (no penalty term included). Person 2 adds it (penalty or XY
mixer). Qubit count with one-hot encoding = sum_t K_t.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil, log2
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------
# Voice constants
# --------------------------------------------------------------------------
BASS, TENOR, ALTO, SOPRANO = 0, 1, 2, 3
VOICE_NAMES = ("bass", "tenor", "alto", "soprano")
N_VOICES = 4


# --------------------------------------------------------------------------
# Music-side data
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Note:
    midi: int                 # MIDI pitch of the melody note
    duration: float           # length in beats (quarter note = 1.0)


@dataclass
class Melody:
    """Input melody (the soprano line)."""
    title: str
    tonic: int                          # tonic pitch class 0-11 (C=0, D=2, ...)
    mode: str                           # "major" or "minor"
    time_signature: Tuple[int, int]     # e.g. (4, 4)
    notes: List[Note]
    harmonic_rhythm: float = 1.0        # beats per chord (1.0 = chord per beat)

    def __post_init__(self) -> None:
        if self.mode not in ("major", "minor"):
            raise ValueError(f"mode must be 'major' or 'minor', got {self.mode!r}")
        if not 0 <= self.tonic < 12:
            raise ValueError("tonic must be a pitch class in 0..11")


@dataclass(frozen=True)
class Voicing:
    """One candidate harmonization for one time slot."""
    pitches: Tuple[int, int, int, int]   # MIDI, order (bass, tenor, alto, soprano)
    chord: str                           # Roman numeral label, e.g. "V7", "IV", "vi"

    def __post_init__(self) -> None:
        if len(self.pitches) != N_VOICES:
            raise ValueError("a Voicing needs exactly 4 pitches")

    @property
    def bass(self) -> int:
        return self.pitches[BASS]

    @property
    def soprano(self) -> int:
        return self.pitches[SOPRANO]


# --------------------------------------------------------------------------
# Long-range (non-adjacent) couplings
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class LongRangeCoupling:
    """
    Extra cost `weight` if candidate k is chosen at slot t AND candidate l is
    chosen at slot s. Requires t < s and s - t >= 2 (adjacent pairs belong in W).
    These become extra ZZ terms in the Hamiltonian.
    """
    t: int
    k: int
    s: int
    l: int
    weight: float


# --------------------------------------------------------------------------
# The object the compiler hands to the quantum team
# --------------------------------------------------------------------------
@dataclass
class CompiledProblem:
    candidates: List[List[Voicing]]          # candidates[t][k]
    h: List[np.ndarray]                      # h[t]: shape (K_t,)
    W: List[np.ndarray]                      # W[t]: shape (K_t, K_{t+1}), len N-1
    long_range: List[LongRangeCoupling] = field(default_factory=list)
    style: str = "chorale"
    energy_scale: float = 1.0                # raw_energy = energy * energy_scale
    first_slot: int = 0                      # slot offset (nonzero for windows)
    # Optional explainability: per-rule share of costs, same shapes as h / W.
    # h_by_rule["NoParallelFifths"][t] is an array of shape (K_t,), etc.
    h_by_rule: Dict[str, List[np.ndarray]] = field(default_factory=dict)
    W_by_rule: Dict[str, List[np.ndarray]] = field(default_factory=dict)

    # ---------------- basic shape helpers ----------------
    @property
    def num_slots(self) -> int:
        return len(self.candidates)

    def num_candidates(self, t: int) -> int:
        return len(self.candidates[t])

    @property
    def n_qubits_one_hot(self) -> int:
        """Qubits needed with one qubit per (slot, candidate)."""
        return sum(len(c) for c in self.candidates)

    @property
    def n_qubits_binary(self) -> int:
        """Qubits needed with a binary index per slot (denser, deeper circuits)."""
        return sum(max(1, ceil(log2(len(c)))) for c in self.candidates)

    def var_index(self, t: int, k: int) -> int:
        """Flat qubit/variable index of (slot t, candidate k) in one-hot encoding.
        Slot-major order: slot 0's candidates first, then slot 1's, etc."""
        return sum(len(self.candidates[i]) for i in range(t)) + k

    def var_name(self, t: int, k: int) -> str:
        """Variable name used for QuadraticProgram: x_{t}_{k}."""
        return f"x_{t}_{k}"

    # ---------------- energy ----------------
    def energy(self, selection: Sequence[int]) -> float:
        """Normalized energy of a selection (one candidate index per slot)."""
        n = self.num_slots
        if len(selection) != n:
            raise ValueError(f"selection has length {len(selection)}, expected {n}")
        e = 0.0
        for t in range(n):
            e += float(self.h[t][selection[t]])
        for t in range(n - 1):
            e += float(self.W[t][selection[t], selection[t + 1]])
        for c in self.long_range:
            if selection[c.t] == c.k and selection[c.s] == c.l:
                e += c.weight
        return e

    def raw_energy(self, selection: Sequence[int]) -> float:
        return self.energy(selection) * self.energy_scale

    def decode(self, selection: Sequence[int]) -> List[Voicing]:
        """Selection of candidate indices -> list of chosen Voicings."""
        return [self.candidates[t][k] for t, k in enumerate(selection)]

    def decode_bitstring(self, bits: Sequence[int]) -> Optional[List[int]]:
        """
        One-hot bitstring (length n_qubits_one_hot, slot-major order) ->
        selection, or None if infeasible (any slot without exactly one 1).
        `bits` must be ordered by var_index, i.e. bits[var_index(t, k)].
        NOTE: Qiskit bitstrings are little-endian (qubit 0 is the RIGHTMOST
        character); reverse the string before passing it here.
        """
        sel: List[int] = []
        pos = 0
        for t in range(self.num_slots):
            k_t = self.num_candidates(t)
            chunk = list(bits[pos:pos + k_t])
            if sum(chunk) != 1:
                return None
            sel.append(chunk.index(1))
            pos += k_t
        return sel

    # ---------------- windowing (used by Person 2) ----------------
    def window(self, start: int, stop: int,
               fixed: Optional[Mapping[int, int]] = None) -> "CompiledProblem":
        """
        Sub-problem over slots [start, stop). `fixed` maps already-decided slot
        indices (in THIS problem's numbering) to their chosen candidate index.

        Costs that connect a window slot to a fixed slot are folded into h:
          - W between a fixed neighbour and the window edge
          - long-range couplings between a window slot and a fixed slot
        Couplings with both ends inside the window are kept and re-indexed.
        Couplings to a slot that is neither inside nor fixed are dropped.

        The returned problem renumbers slots 0..(stop-start-1); first_slot
        records the original offset so results can be stitched back.
        """
        if not 0 <= start < stop <= self.num_slots:
            raise ValueError("invalid window bounds")
        fixed = dict(fixed or {})

        h_w = [self.h[t].copy() for t in range(start, stop)]
        W_w = [self.W[t].copy() for t in range(start, stop - 1)]

        if start > 0 and (start - 1) in fixed:
            h_w[0] = h_w[0] + self.W[start - 1][fixed[start - 1], :]
        if stop < self.num_slots and stop in fixed:
            h_w[-1] = h_w[-1] + self.W[stop - 1][:, fixed[stop]]

        lr_w: List[LongRangeCoupling] = []
        for c in self.long_range:
            t_in = start <= c.t < stop
            s_in = start <= c.s < stop
            if t_in and s_in:
                lr_w.append(LongRangeCoupling(c.t - start, c.k, c.s - start,
                                              c.l, c.weight))
            elif t_in and c.s in fixed:
                if fixed[c.s] == c.l:
                    h_w[c.t - start][c.k] += c.weight
            elif s_in and c.t in fixed:
                if fixed[c.t] == c.k:
                    h_w[c.s - start][c.l] += c.weight

        return CompiledProblem(
            candidates=[self.candidates[t] for t in range(start, stop)],
            h=h_w,
            W=W_w,
            long_range=lr_w,
            style=self.style,
            energy_scale=self.energy_scale,
            first_slot=self.first_slot + start,
        )

    # ---------------- sanity checking ----------------
    def validate(self) -> None:
        """Raise ValueError if the problem violates the contract."""
        n = self.num_slots
        if n == 0:
            raise ValueError("problem has no slots")
        if len(self.h) != n:
            raise ValueError("len(h) must equal number of slots")
        if len(self.W) != n - 1:
            raise ValueError("len(W) must equal number of slots - 1")
        for t in range(n):
            k_t = self.num_candidates(t)
            if k_t == 0:
                raise ValueError(f"slot {t} has no candidates")
            if self.h[t].shape != (k_t,):
                raise ValueError(f"h[{t}] has shape {self.h[t].shape}, expected ({k_t},)")
        for t in range(n - 1):
            expected = (self.num_candidates(t), self.num_candidates(t + 1))
            if self.W[t].shape != expected:
                raise ValueError(f"W[{t}] has shape {self.W[t].shape}, expected {expected}")
        for c in self.long_range:
            if not (0 <= c.t < c.s < n) or c.s - c.t < 2:
                raise ValueError(f"bad long-range coupling slots: {c}")
            if not (0 <= c.k < self.num_candidates(c.t)
                    and 0 <= c.l < self.num_candidates(c.s)):
                raise ValueError(f"bad long-range coupling candidates: {c}")
        for t in range(n):
            for v in self.candidates[t]:
                if v.soprano != self.candidates[t][0].soprano:
                    raise ValueError(f"slot {t}: candidates disagree on soprano note")


# --------------------------------------------------------------------------
# Solver output (produced by Persons 2 and 3, consumed by the API / experiments)
# --------------------------------------------------------------------------
@dataclass
class Sample:
    selection: Tuple[int, ...]       # one candidate index per slot (feasible only)
    energy: float                    # normalized energy via problem.energy()
    probability: float               # fraction of shots that produced it


@dataclass
class ResourceReport:
    n_qubits: int
    depth: int
    two_qubit_gates: int
    shots: int
    reps: int                        # QAOA depth p
    backend: str                     # "aer", "fake_...", "ibm_..."
    extra: Dict[str, float] = field(default_factory=dict)


@dataclass
class SolverResult:
    """What any solver (QAOA, XY-QAOA, Viterbi, ...) returns for one problem."""
    solver: str                              # "qaoa", "xy_qaoa", "viterbi", ...
    best: Tuple[int, ...]                    # lowest-energy feasible selection found
    best_energy: float
    samples: List[Sample] = field(default_factory=list)   # sorted by energy, ascending
    feasible_fraction: float = 1.0           # share of shots that were valid one-hot
    optimal_params: Optional[List[float]] = None          # gammas+betas, for warm starts
    resources: Optional[ResourceReport] = None
    window: Optional[Tuple[int, int]] = None              # (first_slot, stop) if windowed