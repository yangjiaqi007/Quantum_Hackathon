"""Synthetic plumbing fixtures. No QAOA or music-theory claims."""
import numpy as np
from Schema import CompiledProblem, Voicing, SolverResult, Sample
from .contracts import Compilation


def compile_melody(melody, *, style):
    candidates, durations = [], []
    for note in melody.notes:
        remaining = note.duration
        while remaining > 1e-8:
            duration = min(remaining, melody.harmonic_rhythm)
            pitches = tuple(max(0, note.midi - distance) for distance in (24, 17, 12, 0))
            candidates.append([Voicing(pitches, 'DEMO')])
            durations.append(duration)
            remaining -= duration
    problem = CompiledProblem(candidates, [np.zeros(1) for _ in candidates],
                              [np.zeros((1, 1)) for _ in candidates[1:]], style=style)
    return Compilation(problem, durations, [
        'Development fixture: accompanying pitches are synthetic, not a validated harmonization.'])


def solve(problem, *, run, execution, initial_params=None):
    best = (0,) * problem.num_slots
    return SolverResult(solver='mock', best=best, best_energy=problem.energy(best),
                        samples=[Sample(best, problem.energy(best), 1.0)], resources=None)
