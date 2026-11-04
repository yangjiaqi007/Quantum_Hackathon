"""
Checker.py: classical reference solvers and an INDEPENDENT rule checker.

Solvers (work on a CompiledProblem from Compile.py):
    viterbi(problem)               exact minimum-energy selection, any N
    viterbi_k_best(problem, k)     the k lowest-energy selections, exact
    brute_force(problem)           exhaustive, tiny problems only (sanity check)
Persons 2 and 3 use these as the reference optimum.

Checker:
    count_violations(seq, tonic)   violation counts per rule, written from the
                                   paper's wording WITHOUT importing Rules.py
    weighted_violation_energy(...) sum of weight * count
The energy test confirms   problem.raw_energy(x) == weighted_violation_energy(x)
for random selections. If the rule classes, weights, normalization or indexing
contain a bug, this test is what exposes it.

NOTE: the three surrogate rules (LeapResolution, StepwiseMotion, and the leap
penalties) are counted here the way the Hamiltonian defines them, so the energy
test can match. Rules 4, 9 and the two rhythm rules are not in the Hamiltonian.
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Sequence, Tuple

import numpy as np

from Schema import CompiledProblem, Voicing

# --------------------------------------------------------------------------
# Solvers
# --------------------------------------------------------------------------
def viterbi(problem: CompiledProblem) -> Tuple[List[int], float]:
    """Exact minimum of problem.energy over all selections (chain structure)."""
    if problem.long_range:
        raise ValueError("viterbi cannot handle long-range couplings; use brute_force "
                         "or simulated annealing")
    n = problem.num_slots
    cost = problem.h[0].astype(float).copy()
    back: List[np.ndarray] = []
    for t in range(1, n):
        total = cost[:, None] + problem.W[t - 1] + problem.h[t][None, :]
        back.append(np.argmin(total, axis=0))
        cost = np.min(total, axis=0)
    k = int(np.argmin(cost))
    best = float(cost[k])
    selection = [k]
    for t in range(n - 2, -1, -1):
        k = int(back[t][k])
        selection.append(k)
    selection.reverse()
    return selection, best


def viterbi_k_best(problem: CompiledProblem, k: int) -> List[Tuple[List[int], float]]:
    """The k lowest-energy selections, best first (exact, list Viterbi)."""
    if problem.long_range:
        raise ValueError("viterbi_k_best cannot handle long-range couplings")
    n = problem.num_slots
    prev = [[(float(problem.h[0][j]), (j,))] for j in range(problem.num_candidates(0))]
    for t in range(1, n):
        cur = []
        for l in range(problem.num_candidates(t)):
            options = []
            for j, paths in enumerate(prev):
                step = float(problem.W[t - 1][j, l]) + float(problem.h[t][l])
                options.extend((c + step, p + (l,)) for c, p in paths)
            options.sort(key=lambda x: (x[0], x[1]))
            cur.append(options[:k])
        prev = cur
    finals = sorted((x for paths in prev for x in paths), key=lambda x: (x[0], x[1]))[:k]
    return [(list(p), c) for c, p in finals]


def brute_force(problem: CompiledProblem, max_states: int = 2_000_000
                ) -> Tuple[List[int], float]:
    """Try every selection. Only for tiny problems (e.g. a 4-5 slot window)."""
    sizes = [problem.num_candidates(t) for t in range(problem.num_slots)]
    total = int(np.prod(sizes, dtype=object))
    if total > max_states:
        raise ValueError(f"{total} selections is too many for brute force")
    best_sel, best_e = None, float("inf")
    for sel in itertools.product(*[range(s) for s in sizes]):
        e = problem.energy(sel)
        if e < best_e:
            best_sel, best_e = list(sel), e
    return best_sel, best_e


# --------------------------------------------------------------------------
# Independent rule checker (written from the paper; does NOT import Rules.py)
# --------------------------------------------------------------------------
RULE_NAMES = (
    "OpeningChord", "ClosingChord", "FinalCadence", "LeapHeight", "LeapResolution",
    "VoiceCrossing", "VoiceRange", "VoiceSpacing", "LeadingToneResolution",
    "StepwiseMotion", "ParallelOctaves",
    "ParallelFifths", "VerticalConsonance",
)
_TONIC_LABELS = ("I", "i")
# Fig. 3 of the paper, MIDI (low, high) for bass, tenor, alto, soprano.
_RANGES = ((40, 60), (48, 67), (55, 74), (60, 81))
_DISSONANT = (1, 2, 5, 10, 11)     # seconds, fourths, sevenths (mod octave)


def _trailing_run_start(soprano: Sequence[int]) -> int:
    i = len(soprano) - 1
    while i > 0 and soprano[i - 1] == soprano[-1]:
        i -= 1
    return i


def count_violations(seq: Sequence[Voicing], tonic: int) -> Dict[str, int]:
    """Integer violation count per rule for a full harmonization."""
    n = len(seq)
    P = [v.pitches for v in seq]                       # (bass, tenor, alto, soprano)
    c0 = _trailing_run_start([p[3] for p in P])
    counts = {name: 0 for name in RULE_NAMES}

    # Opening / closing chord: first chord, and every chord of the final held note
    if seq[0].chord not in _TONIC_LABELS:
        counts["OpeningChord"] += 1
    counts["ClosingChord"] += sum(1 for t in range(c0, n) if seq[t].chord not in _TONIC_LABELS)

    # Final cadence: V or V7 into the tonic; bass up a fourth (or down a fifth) onto
    # the tonic note; tenor and alto close by step (<= 2 semitones). Soprano is fixed.
    if c0 >= 1:
        a, b = P[c0 - 1], P[c0]
        if seq[c0 - 1].chord not in ("V", "V7"):
            counts["FinalCadence"] += 1
        if not ((b[0] - a[0]) % 12 == 5 and b[0] % 12 == tonic):
            counts["FinalCadence"] += 1
        for voice in (1, 2):
            if abs(b[voice] - a[voice]) > 2:
                counts["FinalCadence"] += 1

    # Single-chord rules
    for p in P:
        counts["VoiceCrossing"] += sum(1 for i in range(4) for j in range(i + 1, 4) if p[i] > p[j])
        counts["VoiceRange"] += sum(1 for i in range(4) if not _RANGES[i][0] <= p[i] <= _RANGES[i][1])
        counts["VerticalConsonance"] += sum(
            1 for i in range(4) for j in range(i + 1, 4) if abs(p[j] - p[i]) % 12 in _DISSONANT)
        counts["VoiceSpacing"] += int(p[3] - p[2] > 12) + int(p[2] - p[1] > 12) + int(p[1] - p[0] > 19)

    # Rules over consecutive chords
    for t in range(n - 1):
        a, b = P[t], P[t + 1]
        for voice in (0, 1, 2):                        # free voices only
            leap = abs(b[voice] - a[voice])
            if leap > 14:                              # larger than a 9th
                counts["LeapHeight"] += 1
            if leap > 9:                               # surrogate: larger than a major 6th
                counts["LeapResolution"] += 1
            if leap >= 3:                              # surrogate: not a step
                counts["StepwiseMotion"] += 1
        counts["LeadingToneResolution"] += sum(
            1 for voice in range(4)
            if a[voice] % 12 == (tonic + 11) % 12 and b[voice] - a[voice] != 1
        )
        for lower in range(3):                         # overlap between adjacent voices
            upper = lower + 1
            if b[lower] > a[upper] or b[upper] < a[lower]:
                counts["VoiceCrossing"] += 1
        for i in range(4):                             # parallel perfect intervals
            for j in range(i + 1, 4):
                if (a[i], a[j]) == (b[i], b[j]):
                    continue
                before, after = abs(a[j] - a[i]) % 12, abs(b[j] - b[i]) % 12
                if before == after == 0:
                    counts["ParallelOctaves"] += 1
                if before == after == 7:
                    counts["ParallelFifths"] += 1
    return counts


def weighted_violation_energy(seq: Sequence[Voicing], tonic: int,
                              weights: Dict[str, float]) -> float:
    counts = count_violations(seq, tonic)
    return sum(weights.get(name, 0.0) * c for name, c in counts.items())


def check_energy(problem: CompiledProblem, selection: Sequence[int], tonic: int,
                 weights: Dict[str, float], tol: float = 1e-8) -> bool:
    """True if the Hamiltonian energy equals the independent weighted violation count."""
    expected = weighted_violation_energy(problem.decode(selection), tonic, weights)
    return abs(problem.raw_energy(selection) - expected) <= tol


# --------------------------------------------------------------------------
# Self-check:  python Checker.py
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import random

    from Compile import compile_melody
    from Melodies import ode_to_joy
    from Weights import get_weights

    # Hand-made cases for the checker itself (D major, tonic = 2)
    I = Voicing((50, 57, 62, 66), "I")
    ii = Voicing((52, 59, 62, 66), "ii")      # bass D->E, tenor A->B: parallel fifths
    ii2 = Voicing((52, 59, 64, 66), "ii")     # bass D->E, alto D->E: parallel octaves
    assert count_violations([I, ii], 2)["ParallelFifths"] == 1
    r = count_violations([I, ii2], 2)
    assert r["ParallelOctaves"] == 1 and r["ParallelFifths"] == 1
    V = Voicing((45, 57, 61, 64), "V")
    I_end = Voicing((50, 57, 62, 62), "I")
    assert count_violations([I, V, I_end], 2)["FinalCadence"] == 0
    assert count_violations([I, Voicing((45, 57, 61, 64), "IV"), I_end], 2)["FinalCadence"] == 1
    assert count_violations([Voicing((50, 57, 55, 66), "I")], 2)["VoiceCrossing"] == 1
    print("hand-made checker cases: ok")

    m = ode_to_joy()
    for style in ("strict", "chorale", "jazz_loose"):
        p = compile_melody(m, style=style, K=4)
        w = get_weights(style)
        assert set(RULE_NAMES) == set(w), "checker and weights disagree on rule names"

        # Viterbi == brute force on small windows (with and without a fixed boundary)
        for win in (p.window(0, 5), p.window(5, 10, fixed={4: 0}), p.window(10, 14)):
            sel_v, e_v = viterbi(win)
            sel_b, e_b = brute_force(win)
            assert abs(e_v - e_b) < 1e-9, (style, e_v, e_b)
            assert abs(win.energy(sel_v) - e_v) < 1e-9

        # k-best: the first entry is the Viterbi optimum, energies never decrease
        best_sel, best_e = viterbi(p)
        kb = viterbi_k_best(p, 5)
        assert abs(kb[0][1] - best_e) < 1e-9
        assert all(kb[i][1] <= kb[i + 1][1] + 1e-12 for i in range(len(kb) - 1))
        assert len({tuple(s) for s, _ in kb}) == len(kb)

        # THE ENERGY TEST: Hamiltonian energy == weighted independent violation count
        rng = random.Random(0)
        for _ in range(300):
            sel = [rng.randrange(len(slot)) for slot in p.candidates]
            assert check_energy(p, sel, m.tonic, w), f"{style}: energy mismatch for {sel}"
        assert check_energy(p, best_sel, m.tonic, w)
        print(f"{style:11s} viterbi optimum raw energy = {p.raw_energy(best_sel):7.2f}   "
              f"energy test: 300 random selections ok")

    # Show the optimal chorale harmonization and its violations
    p = compile_melody(m, style="chorale", K=4)
    sel, _ = viterbi(p)
    print("\noptimal chords:", " ".join(v.chord for v in p.decode(sel)))
    print("violations:", {k: v for k, v in count_violations(p.decode(sel), m.tonic).items() if v})
    print("\nChecker.py: all checks passed")
