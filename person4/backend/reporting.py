"""Independent cost benchmark for the exact candidate problem sent to Qiskit.

This is a correctness/quality check, not a competing product solver. Matching
its optimum does not establish musical correctness or quantum advantage.
"""
from __future__ import annotations

from itertools import product
from math import isfinite, prod
from numbers import Integral
from typing import Sequence

import numpy as np

from Schema import CompiledProblem


def validate_selection(problem: CompiledProblem, selection: Sequence[int]) -> None:
    if len(selection) != problem.num_slots:
        raise ValueError("Solver must select exactly one candidate per time slot.")
    for t, k in enumerate(selection):
        if isinstance(k, bool) or not isinstance(k, Integral) or not 0 <= k < problem.num_candidates(t):
            raise ValueError(f"Invalid selected candidate at slot {t}.")


def _exact_chain(problem: CompiledProblem) -> tuple[list[int], float]:
    previous = problem.h[0].astype(float).copy()
    parents = []
    for t in range(1, problem.num_slots):
        edges = previous[:, None] + problem.W[t - 1]
        parent = np.argmin(edges, axis=0)
        previous = edges[parent, np.arange(len(parent))] + problem.h[t]
        parents.append(parent)
    last = int(np.argmin(previous))
    energy = float(previous[last])
    selection = [last]
    for parent in reversed(parents):
        last = int(parent[last])
        selection.append(last)
    selection.reverse()
    return selection, energy


def build_quality_report(
    problem: CompiledProblem, selection: Sequence[int], *, exhaustive_limit: int = 100_000
) -> dict:
    """Benchmark against the same full candidate set and normalized cost model.

    Adjacent-only models have an exact dynamic-programming baseline. Models
    with long-range costs use bounded enumeration, or clearly report that no
    exact benchmark was run. Gaps remain meaningful with negative/zero costs.
    """
    problem.validate()
    validate_selection(problem, selection)
    if (not isfinite(problem.energy_scale) or problem.energy_scale <= 0
            or any(not np.isfinite(a).all() for a in [*problem.h, *problem.W])
            or any(not isfinite(c.weight) for c in problem.long_range)):
        raise ValueError("Problem costs must be finite and energy_scale positive.")
    energy = problem.energy(selection)
    if not isfinite(energy):
        raise ValueError("Selected energy must be finite.")
    optimum = None
    method = None
    reference_selection = None
    if not problem.long_range:
        reference_selection, optimum = _exact_chain(problem)
        method = "exact_dynamic_programming"
    elif prod(problem.num_candidates(t) for t in range(problem.num_slots)) <= exhaustive_limit:
        choices = product(*(range(problem.num_candidates(t)) for t in range(problem.num_slots)))
        reference_selection = list(min(choices, key=problem.energy))
        optimum = problem.energy(reference_selection)
        method = "exact_enumeration"
    gap = None if optimum is None else energy - optimum
    if gap is not None and abs(gap) < 1e-10:
        gap = 0.0
    contributions = {}
    for name in sorted(set(problem.h_by_rule) | set(problem.W_by_rule)):
        unary = problem.h_by_rule.get(name)
        pairwise = problem.W_by_rule.get(name)
        value = 0.0
        if unary is not None:
            if len(unary) != problem.num_slots or any(
                a.shape != problem.h[t].shape or not np.isfinite(a).all()
                for t, a in enumerate(unary)
            ):
                raise ValueError(f"Invalid per-rule unary costs for {name}.")
            value += sum(float(a[selection[t]]) for t, a in enumerate(unary))
        if pairwise is not None:
            if len(pairwise) != problem.num_slots - 1 or any(
                a.shape != problem.W[t].shape or not np.isfinite(a).all()
                for t, a in enumerate(pairwise)
            ):
                raise ValueError(f"Invalid per-rule transition costs for {name}.")
            value += sum(float(a[selection[t], selection[t + 1]]) for t, a in enumerate(pairwise))
        contributions[name] = value
    return {
        "energy": energy,
        "raw_energy": energy * problem.energy_scale,
        "benchmark_method": method,
        "benchmark_scope": "Same full compiled candidate set and cost model; not all possible music.",
        "optimal_energy": optimum,
        "optimal_selection": reference_selection,
        "energy_gap": gap,
        "matches_optimum": None if gap is None else abs(gap) <= 1e-8,
        "benchmark_skipped_reason": (
            "Long-range problem exceeds the bounded exact-check size." if method is None else None
        ),
        "weighted_rule_costs": contributions,
        "rule_violation_counts": None,
        "rule_note": "Weighted costs are not violation counts; an independent music rule checker is needed.",
        "quantum_advantage_demonstrated": False,
        "verdict": (
            "Educational music-to-quantum bridge. This report does not establish a speed or quality advantage."
        ),
    }
