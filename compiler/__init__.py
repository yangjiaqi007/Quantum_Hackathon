"""Person 1 music-to-Hamiltonian compiler public package."""

from person1.compiler.Compile import compile_melody, explain_selection, load_problem, save_problem
from person1.compiler.Melodies import (
    get_preset,
    melody_from_midi,
    melody_from_text,
    o_canada,
    ode_to_joy,
    slot_durations,
    slot_sopranos,
)
from Schema import CompiledProblem, Melody, Note, Voicing
from person1.compiler.Weights import STYLE_PRESETS, blend_styles, get_weights

__all__ = [
    "CompiledProblem", "Melody", "Note", "Voicing", "STYLE_PRESETS",
    "blend_styles", "compile_melody", "explain_selection", "get_preset",
    "get_weights", "load_problem", "melody_from_midi", "melody_from_text",
    "o_canada", "ode_to_joy", "save_problem", "slot_durations",
    "slot_sopranos",
]
