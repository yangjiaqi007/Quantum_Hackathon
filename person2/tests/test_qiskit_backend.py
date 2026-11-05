"""Cross-checks the Qiskit V2-primitives backend against the pure-Python simulator.

Skipped automatically when Qiskit is not installed. These could NOT be run in the
environment where the package was written (no PyPI access), so run them on your machine:

    pytest tests/test_qiskit_backend.py -v
"""

import random

import pytest

pytest.importorskip("qiskit")

from p2_helpers import brute_force, random_problem  # noqa: E402
from person2.quantum import build_qubo, solve_melody, solve_window  # noqa: E402
from person2.quantum.qiskit_backend import QiskitBackend, build_qaoa_circuit, ising_to_pauli  # noqa: E402
from person2.quantum.simulator import PythonBackend  # noqa: E402


def _scaled_ising(seed=1, slots=(0, 1), k=3):
    q = build_qubo(random_problem(len(slots), k, seed=seed), list(slots))
    return q, q.to_ising().scaled()[0]


@pytest.mark.parametrize("p", [1, 2, 3])
def test_expectation_matches_python_simulator(p):
    _, ising = _scaled_ising()
    rng = random.Random(p)
    params = [rng.uniform(-1, 1) for _ in range(2 * p)]  # unsymmetric on purpose
    ref = PythonBackend(ising, p).expectation(params)
    got = QiskitBackend(ising, p).expectation(params)
    assert got == pytest.approx(ref, abs=1e-8)


def test_parameter_order_is_not_confused_by_qiskit_sorting():
    """p >= 10 would break name-sorted binding; the backend must still agree."""
    _, ising = _scaled_ising(slots=(0,), k=2)
    p = 11
    params = [0.01 * (i + 1) for i in range(2 * p)]
    assert QiskitBackend(ising, p).expectation(params) == pytest.approx(PythonBackend(ising, p).expectation(params), abs=1e-8)


def test_sampled_bit_order_matches_python_marginals():
    _, ising = _scaled_ising(seed=3)
    params = [0.5, 0.3, 0.2, 0.4]
    ref = PythonBackend(ising, 2)
    probs = ref.probabilities(params)
    n = ising.n
    marg = [sum(pr for i, pr in enumerate(probs) if (i >> q) & 1) for q in range(n)]
    counts = QiskitBackend(ising, 2).sample(params, 20000, seed=7)
    total = sum(counts.values())
    for q in range(n):
        freq = sum(c for bits, c in counts.items() if bits[q]) / total
        assert freq == pytest.approx(marg[q], abs=0.03)


def test_pauli_operator_matches_ising_energy_on_basis_states():
    import numpy as np

    _, ising = _scaled_ising(seed=4)
    mat = ising_to_pauli(ising).to_matrix()
    rng = random.Random(0)
    for _ in range(20):
        idx = rng.randrange(1 << ising.n)
        bits = [(idx >> q) & 1 for q in range(ising.n)]
        assert np.real(mat[idx, idx]) == pytest.approx(ising.energy(bits), abs=1e-9)


def test_circuit_shape_is_reportable():
    _, ising = _scaled_ising()
    qc, gam, bet = build_qaoa_circuit(ising, 2)
    ops = qc.count_ops()
    assert qc.num_qubits == ising.n and len(gam) == len(bet) == 2
    assert ops["rzz"] == 2 * len(ising.J) and ops["rx"] == 2 * ising.n


def test_solve_window_on_qiskit_backend_finds_optimum():
    prob = random_problem(3, 3, seed=2)
    q = build_qubo(prob, [0, 1, 2])
    res = solve_window(q, p=2, backend="qiskit", optimizer="nelder-mead", maxiter=100, shots=2048)
    assert res.backend == "qiskit"
    assert res.best_energy == pytest.approx(brute_force(prob)[1], abs=1e-9)


def test_windowed_melody_on_qiskit_matches_full_energy():
    prob = random_problem(6, 3, seed=5)
    res = solve_melody(prob, window=3, p=1, backend="qiskit", optimizer="nelder-mead", maxiter=60, shots=512)
    assert res.energy == pytest.approx(prob.energy(res.assignment))
