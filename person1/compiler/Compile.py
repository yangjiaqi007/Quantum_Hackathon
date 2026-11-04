"""
Compile.py: the entry point of the compiler.

    melody + style  ->  CompiledProblem   (the object Persons 2 and 3 consume)

    from person1.compiler.Melodies import ode_to_joy
    from person1.compiler.Compile import compile_melody
    problem = compile_melody(ode_to_joy(), style="chorale", K=4)

Steps: context -> candidates -> raw h and W from the rules and weights ->
normalize (largest cost = 1.0) -> per-rule breakdowns -> validate().

The function is called compile_melody (not compile) so it never shadows
Python's built-in compile().
"""

from __future__ import annotations

import json
from typing import Dict, List, Sequence, Union

import numpy as np

from .Candidates import DEFAULT_K, generate_candidates
from .Rules import ALL_RULES, RULES_BY_NAME, build_context
from Schema import CompiledProblem, Melody, Voicing
from .Weights import (get_weights, normalize_costs, pairwise_breakdown,
                      unary_breakdown)


def _resolve_weights(style: Union[str, Dict[str, float]]):
    """A preset name, or a dict of weights (e.g. from blend_styles)."""
    if isinstance(style, str):
        return style, get_weights(style)
    unknown = set(style) - set(RULES_BY_NAME)
    if unknown:
        raise ValueError(f"weights for unknown rules: {sorted(unknown)}")
    weights = {r.name: 0.0 for r in ALL_RULES}
    weights.update(style)
    return "custom", weights


def compile_melody(melody: Melody, style: Union[str, Dict[str, float]] = "chorale",
                   K: int = DEFAULT_K, ranges=None,
                   on_unfit: str = "nearest") -> CompiledProblem:
    style_name, weights = _resolve_weights(style)
    ctx = build_context(melody, ranges)
    candidates = generate_candidates(melody, style=style_name, K=K, ctx=ctx,
                                     weights=weights, on_unfit=on_unfit)
    n = len(candidates)
    names = [r.name for r in ALL_RULES]

    # ---- unary costs h (raw, weighted) ----
    h_raw: List[np.ndarray] = []
    h_rule: Dict[str, List[np.ndarray]] = {nm: [] for nm in names}
    for t, slot in enumerate(candidates):
        per = [unary_breakdown(v, t, ctx, weights) for v in slot]
        h_raw.append(np.array([sum(d.values()) for d in per], dtype=float))
        for nm in names:
            h_rule[nm].append(np.array([d[nm] for d in per], dtype=float))

    # ---- transition costs W (raw, weighted) ----
    W_raw: List[np.ndarray] = []
    W_rule: Dict[str, List[np.ndarray]] = {nm: [] for nm in names}
    for t in range(n - 1):
        A, B = candidates[t], candidates[t + 1]
        total = np.zeros((len(A), len(B)))
        parts = {nm: np.zeros((len(A), len(B))) for nm in names}
        for k, a in enumerate(A):
            for l, b in enumerate(B):
                d = pairwise_breakdown(a, b, t, ctx, weights)
                total[k, l] = sum(d.values())
                for nm in names:
                    parts[nm][k, l] = d[nm]
        W_raw.append(total)
        for nm in names:
            W_rule[nm].append(parts[nm])

    # ---- normalize: largest single cost becomes 1.0 ----
    h, W, scale = normalize_costs(h_raw, W_raw)
    h_by_rule = {nm: [x / scale for x in arrs] for nm, arrs in h_rule.items()
                 if any(np.any(x) for x in arrs)}
    W_by_rule = {nm: [x / scale for x in mats] for nm, mats in W_rule.items()
                 if any(np.any(x) for x in mats)}

    problem = CompiledProblem(
        candidates=candidates, h=h, W=W, long_range=[], style=style_name,
        energy_scale=scale, h_by_rule=h_by_rule, W_by_rule=W_by_rule,
    )
    problem.validate()
    return problem


# --------------------------------------------------------------------------
# Explainability: which rules does a given harmonization violate, and how much?
# --------------------------------------------------------------------------
def explain_selection(problem: CompiledProblem, selection: Sequence[int]) -> Dict[str, float]:
    """Raw weighted cost per rule for a selection (sums to problem.raw_energy)."""
    n = problem.num_slots
    out: Dict[str, float] = {}
    for nm, arrs in problem.h_by_rule.items():
        out[nm] = out.get(nm, 0.0) + sum(float(arrs[t][selection[t]]) for t in range(n))
    for nm, mats in problem.W_by_rule.items():
        out[nm] = out.get(nm, 0.0) + sum(
            float(mats[t][selection[t], selection[t + 1]]) for t in range(n - 1))
    return {nm: v * problem.energy_scale for nm, v in out.items() if v > 1e-12}


# --------------------------------------------------------------------------
# Hand-off file for teammates (JSON, no custom classes needed to read it)
# --------------------------------------------------------------------------
def save_problem(problem: CompiledProblem, path: str) -> None:
    data = {
        "style": problem.style,
        "energy_scale": problem.energy_scale,
        "first_slot": problem.first_slot,
        "candidates": [[{"pitches": list(v.pitches), "chord": v.chord} for v in slot]
                       for slot in problem.candidates],
        "h": [x.tolist() for x in problem.h],
        "W": [x.tolist() for x in problem.W],
        "convention": "W[t][k][l] = cost of candidate k at slot t -> candidate l at slot t+1; "
                      "pitches ordered (bass, tenor, alto, soprano) as MIDI",
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def load_problem(path: str) -> CompiledProblem:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    problem = CompiledProblem(
        candidates=[[Voicing(tuple(v["pitches"]), v["chord"]) for v in slot]
                    for slot in data["candidates"]],
        h=[np.array(x, dtype=float) for x in data["h"]],
        W=[np.array(x, dtype=float) for x in data["W"]],
        style=data["style"],
        energy_scale=data["energy_scale"],
        first_slot=data.get("first_slot", 0),
    )
    problem.validate()
    return problem


# --------------------------------------------------------------------------
# Self-check:  python Compile.py   (also writes the hand-off file)
# --------------------------------------------------------------------------
if __name__ == "__main__":
    from .Melodies import ode_to_joy

    m = ode_to_joy()
    for style in ("strict", "chorale", "jazz_loose"):
        p = compile_melody(m, style=style, K=4)
        peak = max([float(x.max()) for x in p.h] + [float(x.max()) for x in p.W])
        assert abs(peak - 1.0) < 1e-9, "largest cost should be normalized to 1.0"
        assert p.num_slots == 32 and len(p.W) == 31
        print(f"{style:11s} slots={p.num_slots}  one-hot qubits (full)={p.n_qubits_one_hot}  "
              f"energy_scale={p.energy_scale:.1f}")

    p = compile_melody(m, style="chorale", K=4)

    # Per-rule breakdown must add up to the energy from h and W
    for sel in ([0] * 32, [1 if len(s) > 1 else 0 for s in p.candidates]):
        parts = explain_selection(p, sel)
        assert abs(sum(parts.values()) - p.raw_energy(sel)) < 1e-9
    print("explain_selection (all-first-candidate harmonization):")
    for nm, v in sorted(explain_selection(p, [0] * 32).items(), key=lambda x: -x[1]):
        print(f"   {nm:18s} {v:6.2f}")

    # Windows for Person 2
    w0 = p.window(0, 4)
    w0.validate()
    w1 = p.window(4, 8, fixed={3: 0})
    w1.validate()
    print(f"4-slot window qubits (one-hot): {w0.n_qubits_one_hot}, {w1.n_qubits_one_hot}")
    assert w0.n_qubits_one_hot <= 16

    # Custom (blended) weights
    from .Weights import blend_styles
    mix = blend_styles(get_weights("chorale"), get_weights("jazz_loose"), 0.7)
    assert compile_melody(m, style=mix).style == "custom"

    # Save / load round trip
    out_path = "ode_to_joy_chorale_problem.json"
    save_problem(p, out_path)
    q = load_problem(out_path)
    sel = [0] * 32
    assert abs(q.energy(sel) - p.energy(sel)) < 1e-12
    print(f"hand-off file written: {out_path}")
    print("Compile.py: all checks passed")
