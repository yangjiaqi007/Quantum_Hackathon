"""Sliding-window QAOA for melodies of any length.

A window covers ``window`` consecutive slots. Everything already committed (not just the
last slot) is passed as ``fixed``, so any ``W`` coupling that reaches back across the
boundary becomes a linear field on the window's qubits.

``stride`` is the number of slots committed per window. With ``stride < window`` the
windows overlap: the trailing slots are re-optimised in the next window, which gives
lookahead at the cost of more windows. ``stride == window`` (default) never revisits a slot.

Warm start: the optimised angles of window ``w`` start the optimiser of window ``w + 1``.
The Ising coefficients are normalised to a maximum of 1 in every window, so angles from
different windows are on the same scale and transfer well.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from .problem import HarmonyProblem
from .qaoa import WindowResult, solve_window
from .qubo import build_qubo


@dataclass
class MelodyResult:
    assignment: list[int]  # chosen candidate per slot
    energy: float  # full-problem energy of the stitched assignment
    windows: list[WindowResult]
    committed: list[list[int]]  # slots committed by each window
    total_evals: int
    seconds: float

    @property
    def max_qubits(self) -> int:
        return max(w.n_qubits for w in self.windows)

    @property
    def repaired_windows(self) -> int:
        return sum(w.repaired for w in self.windows)


def plan_windows(n_slots: int, window: int, stride: int) -> list[tuple[list[int], list[int]]]:
    """Return ``[(window_slots, committed_slots), ...]`` covering every slot exactly once."""
    if window < 1 or stride < 1:
        raise ValueError("window and stride must be >= 1")
    if stride > window:
        raise ValueError("stride cannot exceed window, or slots would be skipped")
    plan = []
    start = 0
    while start < n_slots:
        slots = list(range(start, min(start + window, n_slots)))
        last = start + window >= n_slots  # final window commits everything it covers
        committed = slots if last else slots[:stride]
        plan.append((slots, committed))
        start += len(committed)
    return plan


def solve_melody(
    problem: HarmonyProblem,
    window: int = 3,
    stride: int | None = None,
    p: int = 2,
    warm_start: bool = True,
    solver: Callable[..., WindowResult] = solve_window,
    seed: int | None = 0,
    progress: Callable[[int, int, WindowResult], None] | None = None,
    **solver_kwargs: Any,
) -> MelodyResult:
    """Harmonise a whole melody window by window.

    ``progress(done, total, window_result)`` is called after each window (for the API
    layer's progress bar). ``penalty_factor`` (default 1.0) scales the automatic one-hot
    penalty. Any other keyword argument (``shots``, ``backend``, ``optimizer``, ``maxiter``,
    ``backend_options``, ...) is forwarded to ``solver``.
    """
    t0 = time.perf_counter()
    stride = window if stride is None else stride
    penalty_factor = solver_kwargs.pop("penalty_factor", 1.0)
    plan = plan_windows(problem.n_slots, window, stride)

    fixed: dict[int, int] = {}
    results: list[WindowResult] = []
    committed_log: list[list[int]] = []
    prev_params: list[float] | None = None
    prev_p: int | None = None

    for w, (slots, commit) in enumerate(plan):
        qubo = build_qubo(problem, slots, fixed=fixed, penalty_factor=penalty_factor)
        kwargs = dict(solver_kwargs)
        kwargs["p"] = p
        kwargs["seed"] = None if seed is None else seed + w
        if warm_start and prev_params is not None:
            kwargs["init_params"] = prev_params
            kwargs["init_p"] = prev_p
        res = solver(qubo, **kwargs)
        results.append(res)
        for s in commit:
            fixed[s] = res.best[s]
        committed_log.append(list(commit))
        prev_params, prev_p = res.params, res.p
        if progress:
            progress(w + 1, len(plan), res)

    assignment = [fixed[i] for i in range(problem.n_slots)]
    return MelodyResult(
        assignment=assignment,
        energy=problem.energy(assignment),
        windows=results,
        committed=committed_log,
        total_evals=sum(r.n_evals for r in results),
        seconds=time.perf_counter() - t0,
    )
