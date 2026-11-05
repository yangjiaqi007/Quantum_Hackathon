"""Connect teammate modules to a bounded, observable harmonization job."""
from dataclasses import asdict
from importlib import import_module
from math import isclose, isfinite
from numbers import Integral
from time import perf_counter
import numpy as np

from Schema import SolverResult, VOICE_NAMES
from .contracts import Compilation, PipelineError
from .hardware import create_execution_context, HardwareConfigurationError
from .reporting import build_quality_report, validate_selection


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
        raise PipelineError('Compiler must return person4.backend.contracts.Compilation, including slot durations.')
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


def rank_sampled_paths(problem, window_choices, *, limit=3, beam_width=32):
    """Rank complete paths stitched from feasible samples returned by each window.

    This bounded sample search returns good candidates, not a guarantee of the global top-k.
    """
    beam = [()]

    def partial_energy(selection):
        n = len(selection)
        value = sum(float(problem.h[t][selection[t]]) for t in range(n))
        value += sum(float(problem.W[t][selection[t], selection[t + 1]]) for t in range(n - 1))
        value += sum(c.weight for c in problem.long_range if c.s < n
                     and selection[c.t] == c.k and selection[c.s] == c.l)
        return value

    for choices in window_choices:
        expanded = {prefix + tuple(choice) for prefix in beam for choice in choices}
        beam = sorted(expanded, key=lambda path: (partial_energy(path), path))[:beam_width]
    ranked = sorted(beam, key=lambda path: (problem.energy(path), path))
    # Distinct voicings can spell the same chord progression. Prefer alternatives
    # with different chord labels so the UI presents genuinely different harmony.
    distinct, seen_progressions = [], set()
    for path in ranked:
        progression = tuple(str(problem.candidates[t][candidate].chord)
                            for t, candidate in enumerate(path))
        if progression in seen_progressions:
            continue
        seen_progressions.add(progression)
        distinct.append(path)
        if len(distinct) >= limit:
            break
    return distinct


def run_pipeline(request, settings, progress):
    started = perf_counter()
    mock = request.run.backend == 'mock'
    melody = request.melody.to_schema()
    if mock:
        from .mock_solver import compile_melody, solve
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
    selection, params, window_reports, window_choices = [], None, [], []
    for i, (start, stop) in enumerate(windows):
        progress({'stage': f'Solving window {i + 1} of {len(windows)}',
                  'completed_windows': i, 'total_windows': len(windows)})
        window = p.window(start, stop, fixed=dict(enumerate(selection)))
        tick = perf_counter()
        result = solve(window, run=request.run, execution=execution, initial_params=params)
        validate_result(window, result, mock=mock)
        if not mock and result.solver != request.run.solver:
            raise PipelineError('Solver returned a different algorithm than requested.')
        choices = []
        for sample in result.samples:
            candidate = tuple(int(k) for k in sample.selection)
            if len(candidate) == stop - start and candidate not in choices:
                choices.append(candidate)
            if len(choices) >= 8:
                break
        best_window = tuple(int(k) for k in result.best)
        if best_window not in choices:
            choices.insert(0, best_window)
        window_choices.append(choices)
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
    def checked_quality(path):
        report = build_quality_report(p, path)
        if settings.checker_callable and not mock:
            counts = load_callable(settings.checker_callable)(melody, p.decode(path), compilation.slot_durations)
            if not isinstance(counts, dict) or any(not isinstance(v, int) or v < 0 for v in counts.values()):
                raise PipelineError('Rule checker must return nonnegative integer counts by rule name.')
            report['rule_violation_counts'] = counts
        return report

    def make_slots(path):
        output, onset = [], 0.0
        for t, (v, duration) in enumerate(zip(p.decode(path), compilation.slot_durations)):
            output.append({'slot': t, 'onset': onset, 'duration': float(duration),
                           'pitches': [int(x) for x in v.pitches], 'chord': v.chord,
                           'candidate_index': path[t]})
            onset += float(duration)
        return output

    candidates = sorted(set([tuple(selection), *rank_sampled_paths(p, window_choices, limit=32)]),
                        key=lambda path: (p.energy(path), path))
    sampled_paths, seen_progressions = [], set()
    for path in candidates:
        progression = tuple(str(p.candidates[t][candidate].chord)
                            for t, candidate in enumerate(path))
        if progression in seen_progressions:
            continue
        seen_progressions.add(progression)
        sampled_paths.append(path)
        if len(sampled_paths) == 3:
            break
    selection = list(sampled_paths[0])
    quality = checked_quality(selection)
    if mock:
        quality['verdict'] = 'Development fixture only. No quantum computation or musical validation was performed.'

    def window_energies(path):
        energies = []
        for start, stop in windows:
            subproblem = p.window(start, stop, fixed=dict(enumerate(path[:start])))
            energies.append(float(subproblem.energy(path[start:stop])))
        return energies

    best_window_energies = window_energies(selection)
    for row, energy in zip(window_reports, best_window_energies):
        row['energy'] = energy
    alternatives = []
    for rank, path in enumerate(sampled_paths, 1):
        if tuple(path) == tuple(selection):
            continue
        alternatives.append({'rank': rank, 'energy': float(p.energy(path)),
                             'window_energies': window_energies(path),
                             'slots': make_slots(path), 'quality_report': checked_quality(path)})
    slots = make_slots(selection)
    voices = {name: [] for name in VOICE_NAMES}
    for row in slots:
        for j, name in enumerate(VOICE_NAMES):
            voices[name].append({'midi': row['pitches'][j], 'onset': row['onset'], 'duration': row['duration']})
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
        'alternatives': alternatives,
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
