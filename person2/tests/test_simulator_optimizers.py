import math

import pytest

from p2_helpers import random_problem
from person2.quantum.optimizers import minimize, nelder_mead
from person2.quantum.qaoa import adapt_params, linear_ramp_init
from person2.quantum.qubo import Ising, build_qubo
from person2.quantum.simulator import PythonBackend


def test_single_qubit_closed_form():
    # H = Z, p = 1: <Z> = sin(2 gamma) sin(2 beta)
    be = PythonBackend(Ising(1, [1.0], {}), p=1)
    for g, b in [(0.3, 0.7), (-0.5, 0.2), (1.1, -0.4)]:
        assert be.expectation([g, b]) == pytest.approx(math.sin(2 * g) * math.sin(2 * b), abs=1e-12)


def test_state_is_normalised_and_expectation_matches_probabilities():
    q = build_qubo(random_problem(2, 3, seed=5), [0, 1])
    ising, _ = q.to_ising().scaled()
    be = PythonBackend(ising, p=2)
    params = [0.3, 0.5, 0.2, 0.4]
    probs = be.probabilities(params)
    assert sum(probs) == pytest.approx(1.0)
    manual = sum(pr * ising.energy([(i >> k) & 1 for k in range(ising.n)]) for i, pr in enumerate(probs))
    assert be.expectation(params) == pytest.approx(manual)


def test_zero_angles_give_uniform_distribution():
    q = build_qubo(random_problem(2, 2, seed=6), [0, 1])
    ising, _ = q.to_ising().scaled()
    probs = PythonBackend(ising, p=1).probabilities([0.0, 0.0])
    assert all(p == pytest.approx(1 / 16) for p in probs)


def test_sampling_is_reproducible_and_sums_to_shots():
    q = build_qubo(random_problem(2, 2, seed=7), [0, 1])
    ising, _ = q.to_ising().scaled()
    be = PythonBackend(ising, p=1)
    a = be.sample([0.4, 0.3], 500, seed=11)
    b = be.sample([0.4, 0.3], 500, seed=11)
    assert a == b and sum(a.values()) == 500


def test_wrong_parameter_count_rejected():
    be = PythonBackend(Ising(1, [1.0], {}), p=2)
    with pytest.raises(ValueError):
        be.expectation([0.1, 0.2])


def test_nelder_mead_finds_quadratic_minimum():
    x, f, n = nelder_mead(lambda v: (v[0] - 1) ** 2 + 2 * (v[1] + 0.5) ** 2, [0.0, 0.0], maxiter=300, tol=1e-9)
    assert x[0] == pytest.approx(1.0, abs=1e-3) and x[1] == pytest.approx(-0.5, abs=1e-3)
    assert n > 0


def test_minimize_reports_method_used():
    _, _, _, used = minimize(lambda v: v[0] ** 2, [1.0], method="nelder-mead")
    assert used == "nelder-mead"
    with pytest.raises(ValueError):
        minimize(lambda v: 0.0, [0.0], method="nope")


def test_init_and_depth_adaptation():
    init = linear_ramp_init(3)
    assert len(init) == 6 and init[0] < init[2] and init[3] > init[5]  # gamma up, beta down
    assert adapt_params(init, 3, 3) == init
    longer = adapt_params(init, 3, 5)
    assert len(longer) == 10
    assert all(min(init[:3]) - 1e-9 <= g <= max(init[:3]) + 1e-9 for g in longer[:5])
    assert len(adapt_params([0.5, 0.2], 1, 4)) == 8


@pytest.mark.parametrize("budget", [10, 30, 80])
def test_maxiter_is_an_evaluation_budget_for_nelder_mead(budget):
    """max_iterations must mean the same thing as SciPy COBYLA's maxiter: objective evaluations."""
    calls = []

    def f(v):
        calls.append(1)
        return sum((x - 0.7) ** 2 for x in v)

    _, _, nfev = nelder_mead(f, [0.0] * 6, maxiter=budget, tol=1e-15)
    assert nfev == len(calls)
    assert nfev <= max(budget, 7) + 6  # a final shrink step may overshoot by at most n evaluations
