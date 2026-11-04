"""How good is a QAOA output distribution, measured against the checker's optimum?

All functions take either
    P : exact probability of each valid assignment, shape = slot sizes
        (sums to P(valid); below 1 for plain QAOA), or
    samples : list of assignments from shots.
E is the energy tensor (xy_mixer.energy_tensor). The reference optimum defaults
to E.min(), which on a window is exactly what Person 1's Viterbi checker
returns; pass e_opt to use the checker's value explicitly.
"""
import math
from collections import Counter

import numpy as np


def distribution_metrics(P, E, chords=None, e_opt=None, within=0.05, tol=1e-9):
    e_min = float(E.min()) if e_opt is None else float(e_opt)
    e_max = float(E.max())
    span = e_max - e_min if e_max > e_min else 1.0
    p_valid = float(P.sum())
    p_opt = float(P[E <= e_min + tol].sum())
    mean_e = float((P * E).sum() / p_valid) if p_valid > 0 else math.nan
    ratio = (e_max - E) / span                   # 1 at the optimum, 0 at the worst
    good = ratio >= 1 - within - tol
    out = {
        "p_valid": p_valid,
        "p_opt": p_opt,
        "p_opt_uniform": float(good.size and (E <= e_min + tol).sum() / E.size),
        "approx_ratio": (e_max - mean_e) / span,                  # among valid outputs
        "approx_ratio_all": float((P * ratio).sum()),             # invalid counts as 0
        "p_good": float(P[good].sum()),
        "shots_for_99pct": shots_to_hit(p_opt),
    }
    if chords is not None:
        out["n_good_chord_sequences"] = len(good_chord_sequences(E, chords, within, e_min))
    return out


def shots_to_hit(p, confidence=0.99):
    """Shots needed to sample the optimum at least once with the given confidence."""
    if p <= 0:
        return math.inf
    if p >= 1:
        return 1
    return math.ceil(math.log(1 - confidence) / math.log(1 - p))


def chord_sequence(a, chords):
    return tuple(chords[t][k] for t, k in enumerate(a))


def good_chord_sequences(E, chords, within=0.05, e_opt=None):
    """Distinct chord sequences among assignments within `within` of the optimum
    (on the [E_min, E_max] scale). This is the set a sampler could offer."""
    e_min = float(E.min()) if e_opt is None else float(e_opt)
    span = float(E.max()) - e_min or 1.0
    good = np.argwhere((E - e_min) / span <= within + 1e-9)
    return {chord_sequence(tuple(a), chords) for a in good}


def sample_metrics(samples, E, chords, e_opt=None, within=0.05, n_shots=None):
    """Metrics from shots. n_shots includes infeasible shots (plain QAOA) if given."""
    e_min = float(E.min()) if e_opt is None else float(e_opt)
    span = float(E.max()) - e_min or 1.0
    n_shots = n_shots or len(samples)
    counts = Counter(samples)
    good = {a for a in counts if (E[a] - e_min) / span <= within + 1e-9}
    found = {chord_sequence(a, chords) for a in good}
    reachable = good_chord_sequences(E, chords, within, e_min)
    return {
        "shots": n_shots,
        "feasible_fraction": len(samples) / n_shots,
        "hit_optimum": any(E[a] <= e_min + 1e-9 for a in counts),
        "empirical_p_opt": sum(c for a, c in counts.items() if E[a] <= e_min + 1e-9) / n_shots,
        "distinct_good_voicings": len(good),
        "distinct_good_chord_sequences": len(found),
        "reachable_good_chord_sequences": len(reachable),
    }
