"""Connect teammate modules to a bounded, observable harmonization job."""
from dataclasses import asdict
from importlib import import_module
from math import isclose, isfinite
from numbers import Integral
from time import perf_counter
import numpy as np

from Schema import SolverResult, VOICE_NAMES
from backend.contracts import Compilation, PipelineError
from backend.hardware import create_execution_context, HardwareConfigurationError
from backend.reporting import build_quality_report, validate_selection


def load_callable(reference):
    try:
        module, name = reference.split(':', 1)
        function = getattr(import_module(module), name)
        if not callable(function):
            raise TypeError('Adapter is not callable')
        return function
    except (ValueError, ImportError, AttributeError, TypeError) as exc:
        raise PipelineError('A team adapter is not configured correctly. Check the server configuration.') from exc


def validate_compilation(compilation, melody):
    if not isinstance(compilation, Compilation):
        raise PipelineError('Compiler must return backend.contracts.Compilation, including slot durations.')
    p = compilation.problem
    p.validate()
    if p.first_slot != 0 or not 1 <= p.num_slots <= 256:
        raise PipelineError('Compiler must return the full melody with at most 256 slots.')
    if not isfinite(p.energy_scale) or p.energy_scale <= 0:
        raise PipelineError('Compiler energy_scale must be finite and positive.')
    if (any(not np.isfinite(a).all() for a in [*p.h, *p.W])
            or any(not isfinite(c.weight) for c in p.long_range)):
        raise PipelineError('Compiler produced a non-finite harmony cost.')
    if len(compilation.slot_durations) != p.num_slots:
        raise PipelineError('Compiler must supply one duration for every harmony slot.')
    # Require a partition of the original melody: no pitch changes, dropped
    # notes, or slots spanning an input note boundary, including repeated notes.
    note_index, remaining = 0, melody.notes[0].duration
    for t, duration in enumerate(compilation.slot_durations):
        if not isfinite(duration) or duration <= 0 or note_index >= len(melody.notes):
            raise PipelineError('Compiler returned invalid slot timing.')
        if duration > remaining + 1e-7:
            raise PipelineError('A harmony slot crosses a melody note boundary.')
        for voicing in p.candidates[t]:
            if any(isinstance(pitch, bool) or not isinstance(pitch, Integral)
                   or not 0 <= pitch <= 127 for pitch in voicing.pitches):
                raise PipelineError('Compiler returned invalid MIDI pitches.')
            if voicing.soprano != melody.notes[note_index].midi:
                raise PipelineError('Compiler changed the supplied soprano melody.')
        remaining -= duration
        if abs(remaining) <= 1e-7:
            note_index += 1
            if note_index < len(melody.notes):
                remaining = melody.notes[note_index].duration
    if note_index != len(melody.notes):
        raise PipelineError('Compiled slots do not cover the complete melody.')


def validate_result(problem, result, *, mock):
    if not isinstance(result, SolverResult):
        raise PipelineError('Solver must return Schema.SolverResult.')
    validate_selection(problem, result.best)
    if not isfinite(result.best_energy) or not isclose(
        result.best_energy, problem.energy(result.best), rel_tol=1e-7, abs_tol=1e-7
    ):
        raise PipelineError('Solver energy disagrees with the shared problem.energy() function.')
    if not isfinite(result.feasible_fraction) or not 0 <= result.feasible_fraction <= 1:
        raise PipelineError('Solver returned an invalid feasible fraction.')
    if not mock and result.solver not in ('qaoa', 'xy_qaoa'):
        raise PipelineError('A quantum run must use the requested quantum solver.')
    for sample in result.samples:
        validate_selection(problem, sample.selection)
        if (not isfinite(sample.probability) or not 0 <= sample.probability <= 1
                or not isclose(sample.energy, problem.energy(sample.selection), abs_tol=1e-7)):
            raise PipelineError('Solver returned an invalid sample.')
    if sum(sample.probability for sample in result.samples) > 1 + 1e-7:
        raise PipelineError('Sample probabilities exceed one.')
    if result.optimal_params is not None and not all(isfinite(x) for x in result.optimal_params):
        raise PipelineError('Solver returned invalid warm-start parameters.')
    if result.resources:
        r = result.resources
        if any(isinstance(x, bool) or not isinstance(x, Integral) or x < 0
               for x in (r.n_qubits, r.depth, r.two_qubit_gates, r.shots, r.reps)):
            raise PipelineError('Solver returned invalid resource counts.')
        if any(not isfinite(v) for v in r.extra.values()):
            raise PipelineError('Solver returned non-finite resource measurements.')


def resource_payload(resources):
    if resources is None:
        return None
    result = asdict(resources)
    for name in ('n_qubits', 'depth', 'two_qubit_gates', 'shots', 'reps'):
        result[name] = int(result[name])
    result['extra'] = {str(k): float(v) for k, v in resources.extra.items()}
    return result


def plan_windows(problem, size, qubit_limit):
    windows, start = [], 0
    while start < problem.num_slots:
        stop, count = start, 0
        while stop < min(start + size, problem.num_slots):
            needed = problem.num_candidates(stop)
            if count + needed > qubit_limit:
                break
            count += needed
            stop += 1
        if stop == start:
            raise PipelineError('One slot exceeds the qubit limit. Reduce candidates in the compiler.')
        windows.append((start, stop))
        start = stop
    return windows


def run_pipeline(request, settings, progress):
    started = perf_counter()
    mock = request.run.backend == 'mock'
    melody = request.melody.to_schema()
    if mock:
        from backend.mock_solver import compile_melody, solve
    else:
        if settings.pipeline_mode != 'team' or not settings.compiler_callable or not settings.solver_callable:
            raise PipelineError('Connect the team compiler and solver before requesting quantum execution.')
        compile_melody = load_callable(settings.compiler_callable)
        solve = load_callable(settings.solver_callable)
    progress({'stage': 'Preparing harmony choices', 'completed_windows': 0, 'total_windows': None})
    compilation = compile_melody(melody, style=request.style)
    validate_compilation(compilation, melody)
    p = compilation.problem
    windows = plan_windows(p, request.run.window_size, request.run.max_qubits)
    execution = None
    if not mock:
        try:
            execution = create_execution_context(request.run.backend, shots=request.run.shots,
                                                 seed=request.run.seed, settings=settings)
        except HardwareConfigurationError as exc:
            raise PipelineError(str(exc)) from exc
    selection, params, window_reports = [], None, []
    for i, (start, stop) in enumerate(windows):
        progress({'stage': f'Solving window {i + 1} of {len(windows)}',
                  'completed_windows': i, 'total_windows': len(windows)})
        window = p.window(start, stop, fixed=dict(enumerate(selection)))
        tick = perf_counter()
        result = solve(window, run=request.run, execution=execution, initial_params=params)
        validate_result(window, result, mock=mock)
        if not mock and result.solver != request.run.solver:
            raise PipelineError('Solver returned a different algorithm than requested.')
        selection.extend(int(k) for k in result.best)
        # A changed window width may require a different parameter layout.
        # The adapter owns accepting or discarding these suggested angles.
        params = result.optimal_params
        window_reports.append({
            'start': start, 'stop': stop, 'solver': result.solver,
            'energy': float(result.best_energy),
            'feasible_fraction': float(result.feasible_fraction),
            'elapsed_seconds': perf_counter() - tick,
            'resources': resource_payload(result.resources),
        })
        progress({'stage': f'Solved window {i + 1} of {len(windows)}',
                  'completed_windows': i + 1, 'total_windows': len(windows)})
    progress({'stage': 'Checking the complete harmony', 'completed_windows': len(windows),
              'total_windows': len(windows)})
    voicings = p.decode(selection)
    quality = build_quality_report(p, selection)
    if mock:
        quality['verdict'] = 'Development fixture only. No quantum computation or musical validation was performed.'
    if settings.checker_callable and not mock:
        # Independent checker returns {rule_name: nonnegative violation_count}.
        counts = load_callable(settings.checker_callable)(melody, voicings, compilation.slot_durations)
        if not isinstance(counts, dict) or any(not isinstance(v, int) or v < 0 for v in counts.values()):
            raise PipelineError('Rule checker must return nonnegative integer counts by rule name.')
        quality['rule_violation_counts'] = counts
    slots, onset = [], 0.0
    voices = {name: [] for name in VOICE_NAMES}
    for t, (v, duration) in enumerate(zip(voicings, compilation.slot_durations)):
        slots.append({'slot': t, 'onset': onset, 'duration': float(duration),
                      'pitches': [int(x) for x in v.pitches], 'chord': v.chord,
                      'candidate_index': selection[t]})
        for j, name in enumerate(VOICE_NAMES):
            voices[name].append({'midi': int(v.pitches[j]), 'onset': onset, 'duration': float(duration)})
        onset += float(duration)
    # Keep original soprano articulation instead of retriggering long notes at
    # each harmonic slot. Lower voices remain available at slot resolution.
    onset, soprano = 0.0, []
    for note in melody.notes:
        soprano.append({'midi': note.midi, 'onset': onset, 'duration': note.duration})
        onset += note.duration
    voices['soprano'] = soprano
    resource_rows = [w['resources'] for w in window_reports if w['resources']]
    warnings = list(compilation.warnings)
    if not mock and len(resource_rows) != len(window_reports):
        warnings.append('Some solver windows did not provide resource measurements.')
    return {
        'title': melody.title, 'tempo': request.tempo, 'tonic': melody.tonic,
        'mode': melody.mode, 'time_signature': list(melody.time_signature),
        'voice_order': list(VOICE_NAMES), 'voices': voices, 'slots': slots,
        'execution': {'mode': request.run.backend, 'is_mock': mock,
                      'pipeline_version': settings.pipeline_version,
                      'elapsed_seconds': perf_counter() - started,
                      'estimated_runtime_seconds': None},
        'quantum_report': {
            'windows': window_reports,
            'max_qubits_per_window': max((r['n_qubits'] for r in resource_rows), default=None),
            'resource_windows_reported': len(resource_rows),
            'resource_note': 'Counts describe individual transpiled windows. Shots are per final sampling call, not the optimizer total.',
            'windowing_note': 'Greedy windows preserve boundary costs; the full-melody optimum is not guaranteed.',
        },
        'quality_report': quality, 'warnings': warnings,
    }
