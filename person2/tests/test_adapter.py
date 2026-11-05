"""Tests for quantum/adapter.py against the team contract (Schema.py + person4 pipeline).

Run from the repository root, in a separate pytest invocation from person3's tests
(both packages are called ``quantum``; person2's conftest puts person2/ on sys.path):

    pytest person2/tests -v
"""

import itertools
import math
import random

import pytest

np = pytest.importorskip("numpy")
Schema = pytest.importorskip("Schema")
pipeline = pytest.importorskip("person4.backend.pipeline")

from Schema import CompiledProblem, LongRangeCoupling, SolverResult, Voicing  # noqa: E402

from person2.quantum.adapter import solve, to_harmony_problem  # noqa: E402


def make_problem(n_slots=4, k=3, seed=0, long_range=True):
    rng = random.Random(seed)
    cands = [[Voicing((40 + a, 50, 60, 70), f"c{a}") for a in range(k)] for _ in range(n_slots)]
    h = [np.array([rng.uniform(0, 1) for _ in range(k)]) for _ in range(n_slots)]
    W = [np.array([[rng.uniform(0, 1) if rng.random() < 0.7 else 0.0 for _ in range(k)] for _ in range(k)]) for _ in range(n_slots - 1)]
    lr = []
    if long_range and n_slots >= 3:
        lr = [LongRangeCoupling(0, 1, 2, 2, 0.8), LongRangeCoupling(0, 1, 2, 2, 0.1)]  # duplicate key must add up
    return CompiledProblem(cands, h, W, long_range=lr)


class Run:
    solver, reps, shots, seed, max_iterations, max_qubits = "qaoa", 2, 1024, 42, 80, 16


def brute_force(problem):
    ranges = [range(problem.num_candidates(t)) for t in range(problem.num_slots)]
    return min(itertools.product(*ranges), key=problem.energy)


def test_conversion_preserves_energy_for_every_selection():
    prob = make_problem(4, 3, seed=1)
    hp = to_harmony_problem(prob)
    for sel in itertools.product(range(3), repeat=4):
        assert math.isclose(hp.energy(sel), prob.energy(sel), abs_tol=1e-9)


def test_conversion_of_a_folded_window_matches_window_energy():
    full = make_problem(6, 3, seed=2)
    win = full.window(2, 5, fixed={0: 1, 1: 2})
    hp = to_harmony_problem(win)
    for sel in itertools.product(range(3), repeat=3):
        assert math.isclose(hp.energy(sel), win.energy(sel), abs_tol=1e-9)


def test_result_passes_the_pipelines_own_validation():
    win = make_problem(3, 3, seed=3)
    out = solve(win, run=Run, execution=None, initial_params=None)
    pipeline.validate_result(win, out, mock=False)  # raises PipelineError on any contract violation
    assert isinstance(out, SolverResult) and out.solver == "qaoa"
    assert out.window == (0, 3)
    assert out.resources is None  # no execution context, so no invented hardware numbers


def test_finds_the_true_optimum_on_a_window():
    win = make_problem(3, 3, seed=4)
    out = solve(win, run=Run)
    assert out.best == brute_force(win)
    assert math.isclose(out.best_energy, win.energy(out.best), abs_tol=1e-9)
    assert out.samples == sorted(out.samples, key=lambda s: s.energy)
    assert sum(s.probability for s in out.samples) <= 1 + 1e-9


def test_pipeline_style_windows_stitch_to_a_full_selection():
    """Mirror run_pipeline's loop: window(), fixed prior choices, warm-start params."""
    full = make_problem(6, 3, seed=5)
    selection, params, used_warm = [], None, []
    for start, stop in pipeline.plan_windows(full, 3, 16):
        win = full.window(start, stop, fixed=dict(enumerate(selection)))
        out = solve(win, run=Run, execution=None, initial_params=params)
        pipeline.validate_result(win, out, mock=False)
        used_warm.append(params is not None)
        selection.extend(out.best)
        params = out.optimal_params
    assert used_warm == [False, True]
    assert full.num_slots == len(selection)
    assert full.energy(selection) >= full.energy(brute_force(full)) - 1e-9


def test_warm_start_params_with_different_depth_are_adapted_not_rejected():
    win = make_problem(2, 3, seed=6)
    out = solve(win, run=Run, initial_params=[0.3, 0.2])  # depth 1 params, run asks for reps=2
    assert len(out.optimal_params) == 4


@pytest.mark.parametrize("bad", [[float("nan"), 0.1], [0.1, 0.2, 0.3], [], None])
def test_unusable_warm_start_is_ignored(bad):
    solve(make_problem(2, 3, seed=7), run=Run, initial_params=bad)


class XY(Run):
    solver = "xy_qaoa"


def test_oversized_windows_are_refused():
    with pytest.raises(ValueError, match="limit"):
        solve(make_problem(3, 4), run=Run, max_qubits=8)


def test_xy_request_without_configured_solver_explains_how_to_connect(monkeypatch):
    monkeypatch.delenv("XY_SOLVER_CALLABLE", raising=False)
    with pytest.raises(ValueError, match="XY_SOLVER_CALLABLE"):
        solve(make_problem(2, 3), run=XY)


def test_xy_request_is_forwarded_with_the_pipelines_arguments(monkeypatch):
    import sys
    import types

    seen = {}

    def fake_xy(window, *, run, execution, initial_params):
        seen.update(window=window, run=run, execution=execution, initial_params=initial_params)
        return SolverResult("xy_qaoa", (0,) * window.num_slots, window.energy((0,) * window.num_slots))

    module = types.ModuleType("fake_xy_adapter")
    module.solve_xy = fake_xy
    monkeypatch.setitem(sys.modules, "fake_xy_adapter", module)
    monkeypatch.setenv("XY_SOLVER_CALLABLE", "fake_xy_adapter:solve_xy")

    win, marker = make_problem(2, 3, seed=9), object()
    out = solve(win, run=XY, execution=marker, initial_params=[0.1, 0.2])
    assert out.solver == "xy_qaoa"
    assert seen["window"] is win and seen["run"] is XY
    assert seen["execution"] is marker and seen["initial_params"] == [0.1, 0.2]


@pytest.mark.parametrize("ref", ["nonexistent_module_xyz:solve", "fake_xy_adapter2:missing", "no_colon_here"])
def test_bad_xy_reference_gives_a_clean_error(monkeypatch, ref):
    monkeypatch.setenv("XY_SOLVER_CALLABLE", ref)
    with pytest.raises(ValueError, match="XY_SOLVER_CALLABLE"):
        solve(make_problem(2, 3), run=XY)


def test_unknown_solver_name_is_refused():
    class Bad(Run):
        solver = "magic"

    with pytest.raises(ValueError, match="Unknown solver"):
        solve(make_problem(2, 3), run=Bad)


def test_qaoa_requests_never_touch_the_xy_setting(monkeypatch):
    monkeypatch.setenv("XY_SOLVER_CALLABLE", "nonexistent_module_xyz:solve")
    assert solve(make_problem(2, 3, seed=10), run=Run).solver == "qaoa"


def test_single_candidate_slots_work():
    """The mock compiler emits one candidate per slot; the solver must not choke on it."""
    win = make_problem(3, 1, seed=8, long_range=False)
    out = solve(win, run=Run)
    assert out.best == (0, 0, 0)
