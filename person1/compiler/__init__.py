"""Music parsing, candidate generation, rule scoring, compilation and checking."""

from .Compile import compile_melody, explain_selection, load_problem, save_problem
from .Melodies import get_preset, melody_from_midi, melody_from_text, o_canada, ode_to_joy
from Schema import CompiledProblem, Melody, Note, Voicing
from .Weights import STYLE_PRESETS, blend_styles, get_weights

__all__ = [
    "CompiledProblem", "Melody", "Note", "Voicing", "STYLE_PRESETS",
    "blend_styles", "compile_melody", "explain_selection", "get_preset",
    "get_weights", "load_problem", "melody_from_midi", "melody_from_text",
    "o_canada", "ode_to_joy", "save_problem",
]
