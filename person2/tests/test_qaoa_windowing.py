import pytest

from helpers import brute_force, planted_problem, random_problem
from quantum import build_qubo, plan_windows, solve_melody, solve_window
from quantum.qaoa import repair
from quantum.simulator import PythonBackend

FAST = dict(optimizer="nelder-mead", maxiter=80, shots=512)


# ----------------------------------------------------------------- one window
def test_solve_window_returns_ranked_feasible_samples():
    prob = random_problem(3, 3, seed=10)
    q = build_qubo(prob, [0, 1, 2])
    res = solve_window(q, p=2, **FAST)
    assert res.n_qubits == 9 and res.backend == "python-statevector"
    assert res.shots == 512 and 0 < res.feasible_fraction <= 1
    assert not res.repaired
    energies = [s.energy for s in res.samples]
    assert energies == sorted(energies)
    assert res.best_energy == energies[0]
    assert res.best_energy == pytest.approx(prob.energy([res.best[0], res.best[1], res.best[2]]))
    for s in res.samples:  # every returned sample is a valid one-hot decode
        assert sorted(s.assignment) == [0, 1, 2]


def test_optimizer_lowers_expectation_and_boosts_feasibility():
    prob = random_problem(3, 3, seed=11)
    q = build_qubo(prob, [0, 1, 2])
    res = solve_window(q, p=2, **FAST)
    assert res.expectation < res.trace[0] - 1e-3
    assert res.feasible_fraction > 27 / 512  # uniform sampling gives 27 feasible of 512 states


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_window_finds_true_optimum_among_samples(seed):
    prob = random_problem(3, 3, seed=seed)
    q = build_qubo(prob, [0, 1, 2])
    res = solve_window(q, p=2, optimizer="nelder-mead", maxiter=100, shots=2048)
    _, opt = brute_force(prob)
    assert res.best_energy == pytest.approx(opt, abs=1e-9)


def test_same_seed_is_reproducible():
    q = build_qubo(random_problem(2, 3, seed=12), [0, 1])
    a = solve_window(q, p=1, seed=5, **FAST)
    b = solve_window(q, p=1, seed=5, **FAST)
    assert a.best == b.best and a.params == b.params


def test_window_too_large_is_rejected():
    q = build_qubo(random_problem(3, 4, seed=13), [0, 1, 2])  # 12 qubits
    with pytest.raises(ValueError, match="max_qubits"):
        solve_window(q, p=1, max_qubits=8)


def test_infeasible_only_samples_trigger_flagged_repair():
    prob = random_problem(2, 3, seed=14)
    q = build_qubo(prob, [0, 1])

    class AllZeroBackend(PythonBackend):
        def sample(self, params, shots, seed=None):
            return {tuple([0] * self.n): shots}  # no bit set, never one-hot

    from quantum.qaoa import make_backend  # noqa: F401

    ising, _ = q.to_ising().scaled()
    res = solve_window(q, p=1, backend=AllZeroBackend(ising, 1), **FAST)
    assert res.repaired and res.feasible_fraction == 0.0
    assert sorted(res.best) == [0, 1]
    assert res.best_energy == pytest.approx(prob.energy([res.best[0], res.best[1]]))


def test_repair_picks_from_set_bits_and_is_greedy():
    prob = random_problem(2, 3, seed=15)
    q = build_qubo(prob, [0, 1])
    bits = [0] * 6
    bits[1] = bits[2] = 1  # slot 0: candidates 1 and 2 are on; slot 1: nothing on
    out = repair(q, bits)
    assert out[0] in (1, 2) and out[1] in (0, 1, 2)


# ------------------------------------------------------------------ planning
@pytest.mark.parametrize("n,w,s", [(7, 3, 3), (7, 3, 2), (8, 4, 4), (2, 3, 3), (5, 3, 1), (1, 3, 3)])
def test_plan_covers_each_slot_exactly_once(n, w, s):
    plan = plan_windows(n, w, s)
    committed = [x for _, c in plan for x in c]
    assert committed == list(range(n))
    for slots, c in plan:
        assert len(slots) <= w and set(c) <= set(slots)


def test_plan_rejects_bad_arguments():
    with pytest.raises(ValueError):
        plan_windows(5, 3, 4)
    with pytest.raises(ValueError):
        plan_windows(5, 0, 1)


# ------------------------------------------------------------------ windowing
def test_stitched_energy_is_the_true_full_energy():
    prob = random_problem(7, 3, seed=20, reach=2)
    res = solve_melody(prob, window=3, p=1, **FAST)
    assert len(res.assignment) == 7
    assert res.energy == pytest.approx(prob.energy(res.assignment))
    assert [len(w.slots) for w in res.windows] == [3, 3, 1]
    assert res.max_qubits == 9


def test_windows_respect_fixed_boundary():
    """Window 2's qubo must contain the field from window 1's committed choice."""
    prob = random_problem(6, 3, seed=21)
    seen = []

    def spy(qubo, **kw):
        seen.append(dict(qubo.fixed))
        return solve_window(qubo, **kw)

    res = solve_melody(prob, window=3, p=1, solver=spy, **FAST)
    assert seen[0] == {}
    assert seen[1] == {0: res.assignment[0], 1: res.assignment[1], 2: res.assignment[2]}


def test_never_below_global_optimum_and_close_on_planted_problem():
    prob, target = planted_problem(6, 3, seed=22)
    res = solve_melody(prob, window=3, p=2, optimizer="nelder-mead", maxiter=100, shots=1024)
    _, opt = brute_force(prob)
    assert res.energy >= opt - 1e-9
    assert res.assignment == target
    assert res.energy == pytest.approx(opt)


def test_overlapping_windows_recommit_only_stride_slots():
    prob = random_problem(6, 3, seed=23)
    res = solve_melody(prob, window=3, stride=2, p=1, **FAST)
    assert res.committed == [[0, 1], [2, 3], [4, 5]]
    assert res.energy == pytest.approx(prob.energy(res.assignment))


def test_warm_start_passes_angles_and_saves_evaluations():
    # Identical slots everywhere: every window is the same problem, so transferred angles fit.
    prob, _ = planted_problem(9, 3, seed=24)
    for i in range(9):
        prob.h[i] = [0.0, 1.0, 1.0]
    prob.W = {(i, a, i + 1, b): (0.0 if (a, b) == (0, 0) else 0.3) for i in range(8) for a in range(3) for b in range(3)}
    prob.W = {k: v for k, v in prob.W.items() if v}
    kw = dict(window=3, p=2, optimizer="nelder-mead", maxiter=100, shots=256)
    warm = solve_melody(prob, warm_start=True, **kw)
    cold = solve_melody(prob, warm_start=False, **kw)
    assert [w.warm_started for w in warm.windows] == [False, True, True]
    assert not any(w.warm_started for w in cold.windows)
    assert warm.total_evals < cold.total_evals


def test_progress_callback_called_per_window():
    prob = random_problem(5, 2, seed=25)
    calls = []
    solve_melody(prob, window=2, p=1, progress=lambda d, t, r: calls.append((d, t)), **FAST)
    assert calls == [(1, 3), (2, 3), (3, 3)]


def test_single_short_melody_uses_one_window():
    prob = random_problem(2, 3, seed=26)
    res = solve_melody(prob, window=4, p=1, **FAST)
    assert len(res.windows) == 1 and res.max_qubits == 6
