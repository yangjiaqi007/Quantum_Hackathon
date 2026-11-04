"""Classical optimizers for the QAOA angles.

``cobyla`` uses SciPy when it is installed. ``nelder-mead`` is a small built-in
implementation, so the package works with no dependencies beyond Qiskit.
"""

from __future__ import annotations

from typing import Callable, Sequence


def nelder_mead(
    fun: Callable[[Sequence[float]], float],
    x0: Sequence[float],
    maxiter: int = 200,
    tol: float = 1e-4,
    step: float = 0.25,
) -> tuple[list[float], float, int]:
    """Minimise ``fun`` from ``x0``. Returns ``(x, f(x), n_function_evaluations)``."""
    n = len(x0)
    nfev = 0

    def f(x: Sequence[float]) -> float:
        nonlocal nfev
        nfev += 1
        return float(fun(x))

    pts = [list(map(float, x0))]
    for i in range(n):
        p = list(map(float, x0))
        p[i] += step
        pts.append(p)
    vals = [f(p) for p in pts]

    for _ in range(maxiter):
        order = sorted(range(n + 1), key=lambda i: vals[i])
        pts = [pts[i] for i in order]
        vals = [vals[i] for i in order]
        if abs(vals[-1] - vals[0]) < tol and max(abs(a - b) for a, b in zip(pts[0], pts[-1])) < tol:
            break
        cen = [sum(p[d] for p in pts[:-1]) / n for d in range(n)]

        def along(t: float) -> list[float]:
            return [c + t * (w - c) for c, w in zip(cen, pts[-1])]

        xr = along(-1.0)
        fr = f(xr)
        if fr < vals[0]:
            xe = along(-2.0)
            fe = f(xe)
            pts[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            xc = along(-0.5) if fr < vals[-1] else along(0.5)
            fc = f(xc)
            if fc < min(fr, vals[-1]):
                pts[-1], vals[-1] = xc, fc
            else:
                best = pts[0]
                for i in range(1, n + 1):
                    pts[i] = [b + 0.5 * (x - b) for b, x in zip(best, pts[i])]
                    vals[i] = f(pts[i])
    i = min(range(n + 1), key=lambda i: vals[i])
    return pts[i], vals[i], nfev


def minimize(
    fun: Callable[[Sequence[float]], float],
    x0: Sequence[float],
    method: str = "auto",
    maxiter: int = 200,
    tol: float = 1e-4,
) -> tuple[list[float], float, int, str]:
    """Return ``(x, f(x), n_evals, method_used)``.

    ``method``: ``"auto"`` (COBYLA if SciPy exists, else Nelder-Mead), ``"cobyla"``,
    or ``"nelder-mead"``. The method actually used is returned so reports are exact.
    """
    method = method.lower()
    if method == "auto":
        try:
            import scipy.optimize  # noqa: F401

            method = "cobyla"
        except ImportError:
            method = "nelder-mead"
    if method == "cobyla":
        try:
            from scipy.optimize import minimize as sp_minimize
        except ImportError as exc:
            raise ImportError("optimizer='cobyla' needs SciPy; use 'nelder-mead' or install scipy") from exc
        res = sp_minimize(
            lambda x: float(fun(list(x))), list(x0), method="COBYLA", tol=tol, options={"maxiter": maxiter, "rhobeg": 0.3}
        )
        return [float(v) for v in res.x], float(res.fun), int(res.nfev), "cobyla"
    if method == "nelder-mead":
        x, v, n = nelder_mead(fun, x0, maxiter=maxiter, tol=tol)
        return x, v, n, "nelder-mead"
    raise ValueError(f"unknown optimizer {method!r}")
