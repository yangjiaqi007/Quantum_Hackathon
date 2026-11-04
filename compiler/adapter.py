"""Adapter matching the backend compiler contract.

The returned Compilation contains the shared problem plus one slot duration
per h/W choice. Quantum callers can use ``compilation.problem.h`` and ``.W``
directly or serialize the standard h/W JSON handoff with Compile.save_problem.
"""

from person4.backend.contracts import Compilation
from person1.compiler.Compile import compile_melody as _compile
from person1.compiler.Melodies import slot_durations


def compile_melody(melody, *, style="chorale", K=4):
    problem = _compile(melody, style=style, K=K)
    return Compilation(problem=problem, slot_durations=slot_durations(melody))
