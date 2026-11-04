"""Quantum core (Person 2): QUBO/Ising construction, plain QAOA, sliding-window solver."""

from .problem import HarmonyProblem
from .qaoa import Sample, WindowResult, solve, solve_window
from .qubo import Ising, Qubo, build_qubo
from .windowing import MelodyResult, plan_windows, solve_melody

__all__ = [
    "HarmonyProblem",
    "Ising",
    "MelodyResult",
    "Qubo",
    "Sample",
    "WindowResult",
    "build_qubo",
    "plan_windows",
    "solve",
    "solve_melody",
    "solve_window",
]
