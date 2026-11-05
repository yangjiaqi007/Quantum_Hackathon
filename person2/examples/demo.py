"""End-to-end demo on a random 8-slot problem (stand-in for Person 1's compiler output).

    python person2/examples/demo.py              # pure-Python simulator
    python person2/examples/demo.py qiskit       # Qiskit statevector primitives
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))  # repo root, for ``person2.quantum`` and ``Schema``
sys.path.insert(0, os.path.join(HERE, "..", "tests"))

from p2_helpers import brute_force, random_problem  # noqa: E402
from person2.quantum import solve_melody  # noqa: E402

backend = sys.argv[1] if len(sys.argv) > 1 else "python"
problem = random_problem(n_slots=8, k=3, seed=42, reach=2)

result = solve_melody(
    problem,
    window=3,
    stride=2,
    p=2,
    backend=backend,
    optimizer="nelder-mead",
    maxiter=100,
    shots=1024,
    progress=lambda done, total, w: print(
        f"window {done}/{total}: slots {w.slots}  qubits={w.n_qubits}  evals={w.n_evals}  "
        f"feasible={w.feasible_fraction:.0%}  warm={w.warm_started}"
    ),
)
_, optimum = brute_force(problem)
print("\nassignment:", result.assignment)
print(f"energy {result.energy:.4f}   brute-force optimum {optimum:.4f}   total evals {result.total_evals}")
