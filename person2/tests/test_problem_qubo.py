import itertools
import random

import pytest

from p2_helpers import random_problem
from person2.quantum.problem import HarmonyProblem
from person2.quantum.qubo import build_qubo


def test_energy_matches_hand_calculation():
    p = HarmonyProblem(h=[[1.0, 2.0], [0.5, 0.0]], W={(0, 1, 1, 0): 3.0})
    assert p.energy([0, 0]) == pytest.approx(1.5)
    assert p.energy([1, 0]) == pytest.approx(2.0 + 0.5 + 3.0)
    assert p.pair(1, 0, 0, 1) == 3.0  # either argument order


def test_validation():
    with pytest.raises(ValueError):
        HarmonyProblem(h=[])
    with pytest.raises(ValueError):
        HarmonyProblem(h=[[1.0], [1.0]], W={(1, 0, 0, 0): 1.0})  # i >= j
    with pytest.raises(ValueError):
        HarmonyProblem(h=[[1.0], [1.0]], W={(0, 5, 1, 0): 1.0})  # candidate out of range


def test_from_dense_rejects_double_counted_lower_block():
    h = [[0.0, 0.0], [0.0, 0.0]]
    W = [[[[0.0] * 2 for _ in range(2)] for _ in range(2)] for _ in range(2)]
    W[0][1][1][0] = 2.0
    W[1][0][0][1] = 2.0  # symmetric copy
    with pytest.raises(ValueError, match="symmetric"):
        HarmonyProblem.from_dense(h, W)
    p = HarmonyProblem.from_dense(h, W, symmetric=True)
    assert p.W == {(0, 1, 1, 0): 2.0}  # counted once


def test_ising_and_qubo_energies_agree_on_every_bitstring():
    prob = random_problem(3, 3, seed=1)
    q = build_qubo(prob, [0, 1, 2])
    ising = q.to_ising()
    for bits in itertools.product([0, 1], repeat=q.n):
        assert ising.energy(bits) == pytest.approx(q.energy(bits), abs=1e-9)


def test_feasible_qubo_energy_equals_problem_energy():
    prob = random_problem(4, 3, seed=2)
    q = build_qubo(prob, [0, 1, 2, 3])
    for assign in itertools.product(range(3), repeat=4):
        bits = q.encode(dict(enumerate(assign)))
        assert q.decode(bits) == dict(enumerate(assign))
        assert q.energy(bits) == pytest.approx(prob.energy(assign))


def test_fixed_boundary_is_folded_into_linear_terms():
    prob = random_problem(5, 3, seed=3, reach=2)  # couplings jump over one slot
    fixed = {0: 1, 1: 2}
    q = build_qubo(prob, [2, 3], fixed=fixed)
    for a2, a3 in itertools.product(range(3), repeat=2):
        bits = q.encode({2: a2, 3: a3})
        full = prob.partial_energy({0: 1, 1: 2, 2: a2, 3: a3})
        fixed_only = prob.partial_energy(fixed)
        assert q.energy(bits) == pytest.approx(full - fixed_only)


@pytest.mark.parametrize("seed", range(5))
def test_penalty_makes_every_ground_state_feasible(seed):
    """Even adversarial costs (large negatives) must not favour a non-one-hot state."""
    rng = random.Random(seed)
    prob = random_problem(3, 3, seed=seed, scale=5.0)
    prob.h = [[x - rng.uniform(0, 8) for x in row] for row in prob.h]  # many negative costs
    q = build_qubo(prob, [0, 1, 2])
    energies = {bits: q.energy(bits) for bits in itertools.product([0, 1], repeat=q.n)}
    emin = min(energies.values())
    ground = [b for b, e in energies.items() if e <= emin + 1e-9]
    assert all(q.decode(b) is not None for b in ground)


def test_build_qubo_input_checks():
    prob = random_problem(3, 2)
    with pytest.raises(ValueError):
        build_qubo(prob, [0, 0])
    with pytest.raises(ValueError):
        build_qubo(prob, [1, 0])
    with pytest.raises(ValueError):
        build_qubo(prob, [1], fixed={1: 0})
    with pytest.raises(ValueError):
        build_qubo(prob, [5])


def test_ising_scaling_has_unit_max_coefficient():
    prob = random_problem(3, 3, seed=4, scale=10.0)
    scaled, s = build_qubo(prob, [0, 1, 2]).to_ising().scaled()
    assert s > 1
    coeffs = [abs(c) for c in scaled.h] + [abs(c) for c in scaled.J.values()]
    assert max(coeffs) == pytest.approx(1.0)
