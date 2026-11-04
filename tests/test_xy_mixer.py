import itertools

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from quantum import fast_sim
from quantum.xy_mixer import (OneHotLayout, energy, energy_tensor, prepare_w_state,
                              qubo_terms, x_qaoa_circuit, xy_qaoa_circuit)
from quantum.xy_solve import XModel, XYModel, interp_angles, normalize, xy_solve, x_solve
from experiments.problems import random_window


def all_assignments(sizes):
    return list(itertools.product(*(range(s) for s in sizes)))


@pytest.mark.parametrize("K", [1, 2, 3, 4, 5])
def test_w_state_is_uniform_with_real_amplitudes(K):
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(K)
    prepare_w_state(qc, list(range(K)))
    sv = Statevector(qc).data
    expected = np.zeros(2 ** K, dtype=complex)
    for k in range(K):
        expected[1 << k] = 1 / np.sqrt(K)
    assert np.allclose(sv, expected, atol=1e-9)


def test_layout_round_trip():
    layout = OneHotLayout([3, 4, 2])
    for a in all_assignments(layout.sizes):
        assert layout.decode(layout.encode(a)) == a
        bits = format(layout.encode(a), f"0{layout.n_qubits}b")
        assert layout.decode(bits) == a
    assert layout.decode(0) is None
    assert layout.decode(0b111) is None  # three voicings in slot 0


@pytest.mark.parametrize("penalty", [0.0, 2.0])
def test_qubo_matches_energy_on_valid_bitstrings(penalty):
    h, W, _ = random_window([3, 2, 4], seed=1)
    layout = OneHotLayout.from_h(h)
    lin, quad, const = qubo_terms(h, W, layout, penalty)
    diag = fast_sim.qubo_diagonal(lin, quad, const, layout.n_qubits)
    for a in all_assignments(layout.sizes):
        assert diag[layout.encode(a)] == pytest.approx(energy(h, W, a))
    if penalty:
        assert diag[0] == pytest.approx(const + 0)  # empty selection pays P per slot
        assert diag[0] >= penalty * len(h) - 1e-9


@pytest.mark.parametrize("sizes", [[3, 4, 2], [4, 4, 4], [2, 5]])
def test_xy_circuit_matches_fast_sim_and_stays_feasible(sizes):
    h, W, _ = random_window(sizes, seed=2)
    rng = np.random.default_rng(0)
    p = 2
    params = np.concatenate([rng.uniform(0, 3, p), rng.uniform(0, 1.5, p)])
    qaoa = xy_qaoa_circuit(h, W, p)
    probs = Statevector(qaoa.bind(params)).probabilities()
    layout = qaoa.layout
    fast = fast_sim.xy_probabilities(energy_tensor(h, W), params[:p], params[p:])
    valid = 0.0
    for a in all_assignments(layout.sizes):
        assert probs[layout.encode(a)] == pytest.approx(fast[a], abs=1e-9)
        valid += probs[layout.encode(a)]
    assert valid == pytest.approx(1.0, abs=1e-9)


def test_x_circuit_matches_fast_sim():
    h, W, _ = random_window([3, 3, 2], seed=3)
    params = np.array([0.7, 1.9, 0.4, 0.8])
    qaoa = x_qaoa_circuit(h, W, 2, penalty=1.5)
    probs = Statevector(qaoa.bind(params)).probabilities()
    model = XModel(h, W, 1.5)
    assert np.allclose(probs, model.bitstring_probs(params), atol=1e-9)
    assert model.stats(params)[1] < 0.99  # plain QAOA leaks into invalid states


def test_dephasing_matches_aer_density_matrix():
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import pauli_error
    from quantum.xy_mixer import cost_layer, xy_ring_mixer

    h, W, _ = random_window([3, 2], seed=4)
    layout = OneHotLayout.from_h(h)
    lin, quad, _ = qubo_terms(h, W, layout)
    gs, bs, q = [0.9, 2.1], [0.6, 0.3], 0.07
    z_err = pauli_error([("Z", q), ("I", 1 - q)]).to_instruction()

    qc = QuantumCircuit(layout.n_qubits)
    for t in range(len(h)):
        prepare_w_state(qc, layout.slot_qubits(t))
    for g, b in zip(gs, bs):
        for k in range(layout.n_qubits):
            qc.append(z_err, [k])
        cost_layer(qc, lin, quad, g)
        for t in range(len(h)):
            xy_ring_mixer(qc, layout.slot_qubits(t), b)
    qc.save_probabilities()
    sim = AerSimulator(method="density_matrix")
    probs = sim.run(transpile(qc, sim)).result().data(0)["probabilities"]

    fast = fast_sim.xy_probabilities_dephased(energy_tensor(h, W), gs, bs, q)
    for a in all_assignments(layout.sizes):
        assert probs[layout.encode(a)] == pytest.approx(fast[a], abs=1e-9)


def test_full_dephasing_gives_uniform_output():
    h, W, _ = random_window([3, 4], seed=5)
    P = fast_sim.xy_probabilities_dephased(energy_tensor(h, W), [1.3, 2.0], [0.5, 0.2], 0.5)
    assert np.allclose(P, 1 / P.size)


def test_normalize_preserves_order():
    h, W, _ = random_window([3, 3, 3], seed=6)
    n = normalize(h, W)
    E, E2 = energy_tensor(h, W), energy_tensor(n.h, n.W)
    assert np.allclose(E2, (E - n.offset) / n.scale)
    assert E2.min() == pytest.approx(0) and E2.max() == pytest.approx(1)


def test_interp_angles_shape():
    assert interp_angles(np.array([1.0, 0.5])).shape == (4,)
    assert interp_angles(np.array([1.0, 2.0, 0.5, 0.2])).shape == (6,)


def test_xy_solve_finds_optimum_and_is_feasible():
    h, W, _ = random_window([4, 4, 4], seed=7)
    res = xy_solve(h, W, p=3, shots=1000, seed=1)
    E = energy_tensor(h, W)
    assert res.n_infeasible == 0 and len(res.samples) == 1000
    assert res.best_energy == pytest.approx(E.min())
    # beats uniform sampling over the 64 valid assignments by a wide margin
    p_opt = sum(E[a] == E.min() for a in res.samples) / len(res.samples)
    assert p_opt > 5 / 64


def test_x_solve_reports_infeasible_shots():
    h, W, _ = random_window([3, 3], seed=8)
    res = x_solve(h, W, p=2, penalty=1.0, shots=500, seed=1)
    assert res.n_infeasible > 0
    assert len(res.samples) + res.n_infeasible == 500


def test_xy_model_probabilities_sum_to_one():
    h, W, _ = random_window([2, 3, 4], seed=9)
    P = XYModel(h, W).assignment_probs(np.array([0.4, 1.1, 0.9, 0.3]))
    assert P.sum() == pytest.approx(1.0)
