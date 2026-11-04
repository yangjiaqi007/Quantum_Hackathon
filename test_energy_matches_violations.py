"""The project-wide safety net: CompiledProblem energy == weighted violations.

For selections of candidates, the energy read from the compiled ``h`` / ``W``
(through the schema's own ``energy()``) must equal the weighted violation count
from the independent checker, in total and per rule.
"""
from __future__ import annotations

import itertools
import random
import warnings
from collections import Counter

import pytest

from compiler import RULE_NAMES, CostModel, Key, check, get_preset

from .helpers import random_candidates, random_selection

KEYS = [Key(0, "major"), Key(9, "minor"), Key(7, "major")]
PRESET_NAMES = ["strict_counterpoint", "bach_chorale", "jazz_loose"]


def build(candidates, key, weights, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # unreadable chord labels are deliberate in random data
        return CostModel(weights, **kwargs).build(candidates, key)


def rule_energy(problem, selection, rule):
    """Raw-unit contribution of one rule, read from h_by_rule / W_by_rule."""
    n = problem.num_slots
    total = sum(float(problem.h_by_rule[rule][t][selection[t]]) for t in range(n))
    total += sum(float(problem.W_by_rule[rule][t][selection[t], selection[t + 1]]) for t in range(n - 1))
    return total * problem.energy_scale


def assert_selection_matches(problem, key, weights, selection):
    report = check(problem.decode(selection), key, weights)
    expected = report.weighted_total(weights)
    assert problem.raw_energy(selection) == pytest.approx(expected, abs=1e-9)
    assert problem.energy(selection) == pytest.approx(expected / problem.energy_scale, abs=1e-9)
    contributions = report.contributions(weights)
    for rule in RULE_NAMES:
        assert rule_energy(problem, selection, rule) == pytest.approx(contributions[rule], abs=1e-9), rule
    return report


@pytest.mark.parametrize("preset", PRESET_NAMES)
@pytest.mark.parametrize("key", KEYS, ids=str)
def test_all_selections_of_a_small_problem(preset, key):
    """Exhaustive: every one of the 4*3*3*4 = 144 selections."""
    rng = random.Random(f"{preset}-{key}")
    weights = get_preset(preset)
    cands = random_candidates(rng, num_slots=4, k_min=3, k_max=4)
    cands[1] = cands[1][:3]
    cands[2] = cands[2][:3]
    problem = build(cands, key, weights)
    ranges = [range(problem.num_candidates(t)) for t in range(problem.num_slots)]
    for selection in itertools.product(*ranges):
        assert_selection_matches(problem, key, weights, list(selection))


@pytest.mark.parametrize("preset", PRESET_NAMES)
@pytest.mark.parametrize("key", KEYS, ids=str)
@pytest.mark.parametrize("num_slots", [1, 2, 6, 12])
def test_random_selections(preset, key, num_slots):
    rng = random.Random(f"{preset}-{key}-{num_slots}")
    weights = get_preset(preset)
    problem = build(random_candidates(rng, num_slots, k_min=2, k_max=8), key, weights)
    for _ in range(80):
        assert_selection_matches(problem, key, weights, random_selection(rng, problem))


def test_without_normalisation():
    rng = random.Random(7)
    weights = get_preset("bach_chorale")
    problem = build(random_candidates(rng, 5), KEYS[0], weights, normalize="none")
    assert problem.energy_scale == 1.0
    for _ in range(50):
        assert_selection_matches(problem, KEYS[0], weights, random_selection(rng, problem))


def test_every_rule_is_exercised():
    """Guards against a vacuous pass: each rule must fire somewhere in the random samples."""
    fired = Counter()
    weights = get_preset("bach_chorale")
    for seed, key in enumerate(KEYS):
        rng = random.Random(100 + seed)
        problem = build(random_candidates(rng, 10, k_min=4, k_max=8), key, weights)
        for _ in range(200):
            report = assert_selection_matches(problem, key, weights, random_selection(rng, problem))
            fired.update(report.count)
    missing = [rule for rule in RULE_NAMES if fired[rule] == 0]
    assert not missing, f"rules never violated in samples (test is vacuous for them): {missing}; counts={dict(fired)}"


def test_single_candidate_slots():
    """A slot with one candidate (fully determined) works like any other."""
    rng = random.Random(3)
    weights = get_preset("strict_counterpoint")
    cands = random_candidates(rng, 5, k_min=1, k_max=1)
    cands[2] = random_candidates(rng, 1, k_min=4, k_max=4)[0]
    problem = build(cands, KEYS[0], weights)
    for _ in range(50):
        assert_selection_matches(problem, KEYS[0], weights, random_selection(rng, problem))
