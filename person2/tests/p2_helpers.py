"""Shared test utilities: random problems and brute-force references."""

from __future__ import annotations

import itertools
import random

from person2.quantum.problem import HarmonyProblem


def random_problem(n_slots: int, k: int, seed: int = 0, reach: int = 1, scale: float = 1.0) -> HarmonyProblem:
    """Random costs. ``reach`` is the largest slot distance that gets a W coupling."""
    rng = random.Random(seed)
    h = [[rng.uniform(0, scale) for _ in range(k)] for _ in range(n_slots)]
    W = {}
    for i in range(n_slots):
        for j in range(i + 1, min(i + reach, n_slots - 1) + 1):
            for a in range(k):
                for b in range(k):
                    if rng.random() < 0.6:
                        W[(i, a, j, b)] = rng.uniform(0, scale)
    return HarmonyProblem(h, W)


def planted_problem(n_slots: int, k: int, seed: int = 0, gap: float = 1.0) -> tuple[HarmonyProblem, list[int]]:
    """Problem whose unique optimum is known: a random 'planted' candidate per slot is
    cheap and every planted-to-planted pair is free, everything else costs extra."""
    rng = random.Random(seed)
    target = [rng.randrange(k) for _ in range(n_slots)]
    h = [[0.0 if c == target[i] else gap * rng.uniform(0.8, 1.2) for c in range(k)] for i in range(n_slots)]
    W = {}
    for i in range(n_slots - 1):
        for a in range(k):
            for b in range(k):
                if not (a == target[i] and b == target[i + 1]):
                    W[(i, a, i + 1, b)] = 0.3 * gap * rng.uniform(0.5, 1.5)
    return HarmonyProblem(h, W), target


def brute_force(problem: HarmonyProblem) -> tuple[list[int], float]:
    best, best_e = None, float("inf")
    for assign in itertools.product(*[range(k) for k in problem.sizes]):
        e = problem.energy(assign)
        if e < best_e - 1e-12:
            best, best_e = list(assign), e
    return best, best_e
