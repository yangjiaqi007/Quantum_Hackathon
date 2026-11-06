import itertools
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from Schema import CompiledProblem, LongRangeCoupling, Voicing
from person3.quantum import adapter, fast_sim
from person3.quantum.xy_mixer import OneHotLayout


def make_problem(sizes=(3, 4, 2), seed=0, long_range=True):
    rng = np.random.default_rng(seed)
    candidates = [[Voicing((48 + k, 55, 60, 67 + t), "I") for k in range(K)] for t, K in enumerate(sizes)]
    h = [rng.random(K) for K in sizes]
    W = [rng.random((a, b)) for a, b in zip(sizes, sizes[1:])]
    lr = [LongRangeCoupling(0, 1, 2, 0, 0.7)] if long_range and len(sizes) > 2 else []
    return CompiledProblem(candidates, h, W, long_range=lr)


def selections(problem):
    return itertools.product(*(range(problem.num_candidates(t)) for t in range(problem.num_slots)))


def run_opts(**kw):
    base = dict(solver="xy_qaoa", reps=2, shots=512, seed=1, max_iterations=50, max_qubits=16)
    return SimpleNamespace(**{**base, **kw})


def test_cost_tensor_matches_problem_energy_including_long_range():
    problem = make_problem()
    E = adapter.cost_tensor(problem)
    for sel in selections(problem):
        assert E[sel] == pytest.approx(problem.energy(sel))


def test_circuit_matches_exact_simulator_with_long_range():
    from qiskit.quantum_info import Statevector

    problem = make_problem()
    layout = OneHotLayout([problem.num_candidates(t) for t in range(problem.num_slots)])
    E = adapter.cost_tensor(problem)
    scale = float(E.max() - E.min())
    lin, quad = adapter.cost_terms(problem, layout, scale)
    qc, gammas, betas = adapter.build_circuit(layout, lin, quad, 2)
    params = [0.8, 2.1, 0.5, 0.3]
    bound = qc.assign_parameters({gammas: params[:2], betas: params[2:]})
    probs = Statevector(bound).probabilities()
    exact = fast_sim.xy_probabilities((E - E.min()) / scale, params[:2], params[2:])
    for sel in selections(problem):
        assert probs[layout.encode(sel)] == pytest.approx(exact[sel], abs=1e-9)
    assert sum(probs[layout.encode(s)] for s in selections(problem)) == pytest.approx(1.0)


def test_result_passes_the_pipelines_own_validation():
    from person4.backend.pipeline import validate_result

    problem = make_problem()
    res = adapter.solve(problem, run=run_opts(), execution=None)
    validate_result(problem, res, mock=False)
    assert res.solver == "xy_qaoa"
    assert res.feasible_fraction == 1.0
    assert len(res.optimal_params) == 4


def test_finds_the_true_optimum_on_a_window():
    problem = make_problem((4, 4, 4), seed=3)
    best = min(selections(problem), key=problem.energy)
    res = adapter.solve(problem, run=run_opts(reps=3, shots=1024))
    assert res.best_energy == pytest.approx(problem.energy(best))


@pytest.mark.parametrize("bad", [None, [], [1.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [float("nan")] * 4])
def test_unusable_warm_start_is_ignored(bad):
    problem = make_problem()
    res = adapter.solve(problem, run=run_opts(), initial_params=bad)
    assert res.best_energy == pytest.approx(problem.energy(res.best))


def test_warm_start_is_used():
    problem = make_problem()
    first = adapter.solve(problem, run=run_opts())
    second = adapter.solve(problem, run=run_opts(), initial_params=first.optimal_params)
    assert second.best_energy <= first.best_energy + 1e-9


def test_single_candidate_slots_work():
    problem = make_problem((1, 3, 1), long_range=False)
    res = adapter.solve(problem, run=run_opts())
    assert res.best[0] == 0 and res.best[2] == 0


def test_oversized_windows_are_refused():
    with pytest.raises(ValueError, match="qubits"):
        adapter.solve(make_problem((4, 4, 4)), run=run_opts(max_qubits=8))


def test_aer_execution_reports_resources_and_stays_feasible():
    from person4.backend.hardware import create_execution_context
    from person4.backend.pipeline import validate_result

    problem = make_problem((4, 4, 4), seed=5)
    execution = create_execution_context("aer", shots=512, seed=1, settings=None)
    res = adapter.solve(problem, run=run_opts(), execution=execution)
    validate_result(problem, res, mock=False)
    assert res.feasible_fraction == 1.0
    assert res.resources.n_qubits == 12 and res.resources.two_qubit_gates > 0
    assert res.resources.extra["repaired"] == 0.0


def test_noisy_backend_drops_invalid_shots():
    from person4.backend.config import Settings
    from person4.backend.hardware import create_execution_context
    from person4.backend.pipeline import validate_result

    problem = make_problem((3, 3), long_range=False)
    execution = create_execution_context("noisy", shots=512, seed=1, settings=Settings())
    res = adapter.solve(problem, run=run_opts(reps=1), execution=execution)
    validate_result(problem, res, mock=False)
    assert 0.0 < res.feasible_fraction < 1.0  # hardware noise breaks the one-hot rule sometimes


def test_person2_router_forwards_xy_requests_here(monkeypatch):
    from person2.quantum.adapter import solve as router

    monkeypatch.setenv("XY_SOLVER_CALLABLE", "person3.quantum.adapter:solve")
    res = router(make_problem(), run=run_opts(), execution=None, initial_params=None)
    assert res.solver == "xy_qaoa"


def test_full_pipeline_runs_with_xy_qaoa(monkeypatch, tmp_path):
    from person4.backend.config import Settings
    from person4.backend.models import HarmonizeRequest
    from person4.backend.pipeline import run_pipeline

    monkeypatch.setenv("XY_SOLVER_CALLABLE", "person3.quantum.adapter:solve")
    notes = [64, 64, 65, 67, 67, 65, 64, 62]  # first bar and a half of Ode to Joy
    request = HarmonizeRequest.model_validate({
        "melody": {"notes": [{"midi": m, "duration": 1} for m in notes]},
        "run": {"backend": "aer", "solver": "xy_qaoa", "reps": 1, "shots": 256, "window_size": 3},
    })
    settings = Settings(data_dir=tmp_path, pipeline_mode="team",
                        compiler_callable="compiler.adapter:compile_melody",
                        solver_callable="person2.quantum.adapter:solve")
    result = run_pipeline(request, settings, lambda update: None)
    windows = result["quantum_report"]["windows"]
    assert len(result["slots"]) >= 1 and windows
    assert all(w["solver"] == "xy_qaoa" for w in windows)
    assert all(w["feasible_fraction"] == 1.0 for w in windows)
    assert result["quantum_report"]["resource_windows_reported"] == len(windows)
