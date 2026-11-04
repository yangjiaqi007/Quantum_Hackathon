from dataclasses import replace
from itertools import product
import numpy as np
import pytest

from Schema import SolverResult, CompiledProblem, Voicing
from backend.config import Settings
from backend.contracts import Compilation, PipelineError
from backend.models import HarmonizeRequest
from backend.pipeline import run_pipeline, validate_compilation, validate_result, plan_windows


def fixture_compiler(melody, *, style):
    candidates = [[Voicing((48, 55, 60, n.midi), 'I'), Voicing((43, 55, 59, n.midi), 'V')]
                  for n in melody.notes]
    n = len(candidates)
    return Compilation(CompiledProblem(candidates, [np.array([0., .2]) for _ in candidates],
                                       [np.array([[2., 0.], [0., 2.]]) for _ in range(n - 1)]),
                       [note.duration for note in melody.notes])


def exact_fixture_solver(problem, *, run, execution, initial_params=None):
    best = min(product(*(range(problem.num_candidates(t)) for t in range(problem.num_slots))),
               key=problem.energy)
    return SolverResult('qaoa', best, problem.energy(best))


def test_window_boundaries_include_previous_cost_and_full_energy(monkeypatch, tmp_path):
    request = HarmonizeRequest.model_validate({'melody': {'notes': [
        {'midi': 64, 'duration': 1}, {'midi': 65, 'duration': 1}, {'midi': 67, 'duration': 1}]},
        'run': {'backend': 'aer', 'window_size': 1}})
    settings = Settings(data_dir=tmp_path, pipeline_mode='team',
                        compiler_callable='tests.test_pipeline:fixture_compiler',
                        solver_callable='tests.test_pipeline:exact_fixture_solver')
    monkeypatch.setattr('backend.pipeline.create_execution_context', lambda *a, **kw: object())
    updates = []
    result = run_pipeline(request, settings, updates.append)
    assert [s['candidate_index'] for s in result['slots']] == [0, 1, 0]
    assert result['quality_report']['energy'] == pytest.approx(.2)
    assert updates[-1]['completed_windows'] == 3
    assert len(result['warnings']) == 1  # fixture has no circuit resource data


def test_compiler_must_preserve_melody_and_timing():
    request = HarmonizeRequest.model_validate({'melody': {'notes': [{'midi': 64, 'duration': 1}]}})
    compilation = fixture_compiler(request.melody.to_schema(), style='chorale')
    compilation.slot_durations = [2.]
    with pytest.raises(PipelineError, match='boundary'):
        validate_compilation(compilation, request.melody.to_schema())
    compilation.slot_durations = [1.]
    compilation.problem.candidates[0] = [Voicing((48, 55, 60, 65), 'I')]
    compilation.problem.h[0] = np.zeros(1)
    with pytest.raises(PipelineError, match='soprano'):
        validate_compilation(compilation, request.melody.to_schema())


def test_bad_solver_energy_and_negative_selection_rejected():
    p = CompiledProblem([[Voicing((48, 55, 60, 64), 'I')]], [np.array([.3])], [])
    with pytest.raises(PipelineError, match='energy'):
        validate_result(p, SolverResult('qaoa', (0,), 0.), mock=False)
    with pytest.raises(ValueError):
        validate_result(p, SolverResult('qaoa', (-1,), .3), mock=False)


def test_adaptive_window_qubit_budget():
    v = Voicing((48, 55, 60, 64), 'I')
    p = CompiledProblem([[v] * 4, [v] * 3, [v] * 4],
                        [np.zeros(4), np.zeros(3), np.zeros(4)],
                        [np.zeros((4, 3)), np.zeros((3, 4))])
    assert plan_windows(p, 3, 7) == [(0, 2), (2, 3)]
    with pytest.raises(PipelineError, match='qubit limit'):
        plan_windows(p, 3, 2)
