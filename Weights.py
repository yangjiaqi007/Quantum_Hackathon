"""
weights.py: how much each rule matters in each style, plus the helpers that
turn rule violations into weighted costs.

A weight is "cost per violation". Weights are kept within about a 20:1 range on
purpose: QAOA struggles when the energy landscape spans several orders of
magnitude. Final scaling to ~[0, 1] happens in normalize_costs().

Styles
------
strict      Textbook counterpoint: parallels and crossings are near-forbidden.
chorale     Bach-chorale default.
jazz_loose  Parallels and dissonance tolerated; instrument ranges still respected.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from Rules import ALL_RULES, RULES_BY_NAME, RuleContext
from Schema import Voicing

STYLE_PRESETS: Dict[str, Dict[str, float]] = {
    "strict": {
        "ParallelFifths": 20, "ParallelOctaves": 20, "VoiceCrossing": 20,
        "VoiceRange": 20, "LeapHeight": 12, "OpeningChord": 10, "ClosingChord": 10,
        "FinalCadence": 8, "LeapResolution": 4, "StepwiseMotion": 2,
        "VerticalConsonance": 1,
    },
    "chorale": {
        "ParallelFifths": 10, "ParallelOctaves": 10, "VoiceCrossing": 10,
        "VoiceRange": 10, "LeapHeight": 8, "OpeningChord": 8, "ClosingChord": 8,
        "FinalCadence": 6, "LeapResolution": 2, "StepwiseMotion": 1,
        "VerticalConsonance": 0.5,
    },
    "jazz_loose": {
        "ParallelFifths": 0.5, "ParallelOctaves": 0.5, "VoiceCrossing": 4,
        "VoiceRange": 10, "LeapHeight": 3, "OpeningChord": 4, "ClosingChord": 4,
        "FinalCadence": 2, "LeapResolution": 0.5, "StepwiseMotion": 0.5,
        "VerticalConsonance": 0.0,
    },
}


def _validate(weights: Dict[str, float]) -> None:
    unknown = set(weights) - set(RULES_BY_NAME)
    if unknown:
        raise ValueError(f"weights for unknown rules: {sorted(unknown)}")
    if any(w < 0 for w in weights.values()):
        raise ValueError("weights must be non-negative")


def get_weights(style: str) -> Dict[str, float]:
    """Weights for a preset; rules missing from a preset get weight 0."""
    if style not in STYLE_PRESETS:
        raise ValueError(f"unknown style {style!r}; choose from {sorted(STYLE_PRESETS)}")
    w = {r.name: 0.0 for r in ALL_RULES}
    w.update(STYLE_PRESETS[style])
    _validate(w)
    return w


def blend_styles(a: Dict[str, float], b: Dict[str, float], alpha: float) -> Dict[str, float]:
    """alpha * a + (1 - alpha) * b, e.g. blend_styles(chorale, jazz, 0.7)."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    keys = set(a) | set(b)
    return {k: alpha * a.get(k, 0.0) + (1 - alpha) * b.get(k, 0.0) for k in keys}


# --------------------------------------------------------------------------
# Weighted scoring (used by candidates.py for pruning and by compile.py for h, W)
# --------------------------------------------------------------------------
def unary_breakdown(v: Voicing, t: int, ctx: RuleContext,
                    weights: Dict[str, float]) -> Dict[str, float]:
    """Weighted unary cost of each rule (for explainability)."""
    return {r.name: weights.get(r.name, 0.0) * r.unary(v, t, ctx) for r in ALL_RULES}


def pairwise_breakdown(a: Voicing, b: Voicing, t: int, ctx: RuleContext,
                       weights: Dict[str, float]) -> Dict[str, float]:
    return {r.name: weights.get(r.name, 0.0) * r.pairwise(a, b, t, ctx) for r in ALL_RULES}


def weighted_unary(v: Voicing, t: int, ctx: RuleContext, weights: Dict[str, float]) -> float:
    return sum(weights.get(r.name, 0.0) * r.unary(v, t, ctx) for r in ALL_RULES)


def weighted_pairwise(a: Voicing, b: Voicing, t: int, ctx: RuleContext,
                      weights: Dict[str, float]) -> float:
    return sum(weights.get(r.name, 0.0) * r.pairwise(a, b, t, ctx) for r in ALL_RULES)


# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------
def normalize_costs(h: List[np.ndarray], W: List[np.ndarray]
                    ) -> Tuple[List[np.ndarray], List[np.ndarray], float]:
    """Divide every cost by the largest single cost so the biggest is 1.0.
    Returns (h, W, scale); raw = normalized * scale (this is CompiledProblem.energy_scale)."""
    peaks = [float(np.max(x)) for x in h] + [float(np.max(x)) for x in W if x.size]
    peak = max(peaks + [0.0])
    if peak <= 0.0:
        return h, W, 1.0
    return [x / peak for x in h], [x / peak for x in W], peak


# --------------------------------------------------------------------------
# Self-check:  python weights.py
# --------------------------------------------------------------------------
if __name__ == "__main__":
    for style in STYLE_PRESETS:
        w = get_weights(style)
        nonzero = [v for v in w.values() if v > 0]
        ratio = max(nonzero) / min(nonzero)
        print(f"{style:11s} rules weighted: {len(nonzero):2d}   max/min weight ratio: {ratio:.0f}")
        assert ratio <= 40, "dynamic range too wide for QAOA"
    assert set(get_weights("chorale")) == set(RULES_BY_NAME)

    mix = blend_styles(get_weights("chorale"), get_weights("jazz_loose"), 0.7)
    assert abs(mix["ParallelFifths"] - (0.7 * 10 + 0.3 * 0.5)) < 1e-9

    h = [np.array([0.0, 20.0]), np.array([5.0, 10.0])]
    W = [np.array([[0.0, 10.0], [4.0, 2.0]])]
    h2, W2, scale = normalize_costs(h, W)
    assert scale == 20.0 and float(np.max(h2[0])) == 1.0 and float(W2[0][0, 1]) == 0.5

    try:
        get_weights("nonsense")
        raise AssertionError("should have rejected unknown style")
    except ValueError:
        pass
    print("weights.py: all checks passed")