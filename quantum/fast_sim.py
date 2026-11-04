"""Exact, fast simulators for the two QAOA circuits in xy_mixer.py.

Angle optimization runs the circuit hundreds of times, so we simulate it
directly instead of through a generic statevector:

* XY-QAOA never leaves the valid subspace (one voicing per slot), so its state
  is a tensor over assignments with prod(K_t) entries (64 for 3 slots x 4
  voicings) instead of 2^12 = 4096 amplitudes.
* Dephasing (Z errors) also preserves that subspace, so the noisy version is
  an exact prod(K_t)-dimensional density matrix.
* Plain QAOA uses the full 2^n statevector.

tests/test_xy_mixer.py checks all three against Qiskit Statevector / Aer
on the actual circuits.
"""
import numpy as np

from quantum.xy_mixer import ring_edge_layers


def slot_mixer_unitary(K, beta):
    """The XY ring mixer of one slot, restricted to its K one-excitation states.

    On {|..1_i..0_j..>, |..0_i..1_j..>} the gate exp(-i beta (XX+YY)/2) is
    [[cos b, -i sin b], [-i sin b, cos b]]: a hop of the excitation between i and j.
    """
    U = np.eye(K, dtype=complex)
    c, s = np.cos(beta), np.sin(beta)
    for layer in ring_edge_layers(K):
        for i, j in layer:
            G = np.eye(K, dtype=complex)
            G[i, i] = G[j, j] = c
            G[i, j] = G[j, i] = -1j * s
            U = G @ U
    return U


def _apply_on_axis(psi, U, axis):
    return np.moveaxis(np.tensordot(U, psi, axes=([1], [axis])), 0, axis)


def xy_state(cost, gammas, betas):
    """Final state of XY-QAOA as a tensor over assignments (shape = slot sizes)."""
    sizes = cost.shape
    psi = np.full(sizes, 1 / np.sqrt(cost.size), dtype=complex)  # product of W states
    for g, b in zip(gammas, betas):
        psi = psi * np.exp(-1j * g * cost)
        for t, K in enumerate(sizes):
            if K > 1:
                psi = _apply_on_axis(psi, slot_mixer_unitary(K, b), t)
    return psi


def xy_probabilities(cost, gammas, betas):
    return np.abs(xy_state(cost, gammas, betas)) ** 2


def xy_probabilities_dephased(cost, gammas, betas, q):
    """XY-QAOA with a Z error of probability q on every qubit before each cost layer.

    rho -> (1-q) rho + q Z rho Z on qubit (t, k) multiplies rho[a, b] by (1 - 2q)
    when exactly one of a, b selects voicing k at slot t. Over all qubits of all
    slots that is (1 - 2q)^(2 * number of slots where a and b differ).
    At q = 0.5 the state is a classical mixture before every cost layer, so the
    cost phases cannot interfere and the output is uniform.
    """
    sizes = cost.shape
    D = cost.size
    idx = np.array(np.unravel_index(np.arange(D), sizes)).T
    n_diff = (idx[:, None, :] != idx[None, :, :]).sum(-1)
    damp = (1 - 2 * q) ** (2 * n_diff)
    c = cost.ravel()
    diff = np.subtract.outer(c, c)
    rho = np.full((D, D), 1 / D, dtype=complex)
    for g, b in zip(gammas, betas):
        rho = rho * damp * np.exp(-1j * g * diff)
        U = np.ones((1, 1), dtype=complex)
        for K in sizes:  # C-order ravel: first slot is the most significant index
            U = np.kron(U, slot_mixer_unitary(K, b))
        rho = U @ rho @ U.conj().T
    return np.real(np.diag(rho)).reshape(sizes)


def qubo_diagonal(lin, quad, const, n):
    """QUBO value of every basis state, indexed like a Qiskit statevector."""
    z = np.arange(2 ** n)
    bits = ((z[:, None] >> np.arange(n)) & 1).astype(float)
    vals = const + bits @ np.asarray(lin, dtype=float)
    for (i, j), w in quad.items():
        vals += w * bits[:, i] * bits[:, j]
    return vals


def x_state(diag, n, gammas, betas):
    """Plain QAOA statevector: |+>^n, then (cost phase, RX(2 beta) on every qubit)^p."""
    psi = np.full(2 ** n, 2 ** (-n / 2), dtype=complex)
    for g, b in zip(gammas, betas):
        psi *= np.exp(-1j * g * diag)
        c, s = np.cos(b), np.sin(b)
        for q in range(n):
            v = psi.reshape(2 ** (n - 1 - q), 2, 2 ** q)
            a0, a1 = v[:, 0, :].copy(), v[:, 1, :].copy()
            v[:, 0, :] = c * a0 - 1j * s * a1
            v[:, 1, :] = c * a1 - 1j * s * a0
    return psi


def x_probabilities(diag, n, gammas, betas):
    return np.abs(x_state(diag, n, gammas, betas)) ** 2
