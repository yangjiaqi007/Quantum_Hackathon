"""Qiskit (V2 primitives) backend for the QAOA solver.

Default: ``qiskit.primitives.StatevectorEstimator`` and ``StatevectorSampler`` (exact,
ships with Qiskit). To use Aer, a noisy fake backend, or IBM hardware, pass your own V2
``estimator`` / ``sampler`` and, for hardware, a preset ``pass_manager``::

    from qiskit_aer.primitives import EstimatorV2, SamplerV2           # Aer
    solve_window(qubo, backend="qiskit", estimator=EstimatorV2(), sampler=SamplerV2())

    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    pm = generate_preset_pass_manager(optimization_level=1, backend=ibm_backend)
    # estimator/sampler = qiskit_ibm_runtime.EstimatorV2 / SamplerV2(mode=ibm_backend)

Conventions match :mod:`quantum.simulator`: qubit ``q`` is the q-th one-hot variable, Ising
spin ``z = +1`` for bit 0, cost layer ``RZ(2 gamma h)`` / ``RZZ(2 gamma J)``, mixer ``RX(2 beta)``.
"""

from __future__ import annotations

from typing import Any, Sequence

from .qubo import Ising


def _require_qiskit():
    try:
        import qiskit  # noqa: F401
    except ImportError as exc:
        raise ImportError("backend='qiskit' needs qiskit (pip install qiskit)") from exc


def build_qaoa_circuit(ising: Ising, p: int):
    """Return ``(circuit, gammas, betas)``; ``gammas``/``betas`` are lists of Parameters."""
    _require_qiskit()
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter

    gammas = [Parameter(f"gamma_{l:02d}") for l in range(p)]
    betas = [Parameter(f"beta_{l:02d}") for l in range(p)]
    qc = QuantumCircuit(ising.n, name="qaoa")
    qc.h(range(ising.n))
    for l in range(p):
        for q, c in enumerate(ising.h):
            if c:
                qc.rz(2 * c * gammas[l], q)
        for (q, r), c in sorted(ising.J.items()):
            qc.rzz(2 * c * gammas[l], q, r)
        for q in range(ising.n):
            qc.rx(2 * betas[l], q)
    return qc, gammas, betas


def ising_to_pauli(ising: Ising):
    """``SparsePauliOp`` for the Ising Hamiltonian (constant included)."""
    _require_qiskit()
    from qiskit.quantum_info import SparsePauliOp

    terms: list[tuple[str, list[int], float]] = [("Z", [q], c) for q, c in enumerate(ising.h) if c]
    terms += [("ZZ", [q, r], c) for (q, r), c in sorted(ising.J.items())]
    op = SparsePauliOp.from_sparse_list(terms, num_qubits=ising.n) if terms else SparsePauliOp("I" * ising.n, 0.0)
    if ising.const:
        op = op + SparsePauliOp("I" * ising.n, ising.const)
    return op.simplify()


class QiskitBackend:
    def __init__(
        self,
        ising: Ising,
        p: int,
        estimator: Any = None,
        sampler: Any = None,
        pass_manager: Any = None,
        precision: float | None = None,
    ) -> None:
        _require_qiskit()
        from qiskit.primitives import StatevectorEstimator, StatevectorSampler

        self.ising, self.p, self.n = ising, p, ising.n
        self.name = "qiskit"
        self._custom_sampler = sampler is not None
        self._estimator = estimator if estimator is not None else StatevectorEstimator()
        self._sampler = sampler if sampler is not None else StatevectorSampler()
        self._precision = precision

        circuit, gammas, betas = build_qaoa_circuit(ising, p)
        self._order_ref = list(gammas) + list(betas)  # user parameter order: gammas then betas
        observable = ising_to_pauli(ising)
        measured = circuit.measure_all(inplace=False)
        if pass_manager is not None:
            circuit = pass_manager.run(circuit)
            observable = observable.apply_layout(circuit.layout)
            measured = pass_manager.run(measured)
        self.circuit, self.measured_circuit, self.observable = circuit, measured, observable

    # Qiskit orders circuit.parameters by name, so map explicitly instead of trusting it.
    def _values(self, circuit: Any, params: Sequence[float]) -> list[float]:
        if len(params) != 2 * self.p:
            raise ValueError(f"expected {2 * self.p} parameters (p gammas then p betas)")
        pos = {par: i for i, par in enumerate(self._order_ref)}
        return [float(params[pos[par]]) for par in circuit.parameters]

    def expectation(self, params: Sequence[float]) -> float:
        import numpy as np

        pub = (self.circuit, self.observable, [self._values(self.circuit, params)])
        kwargs = {} if self._precision is None else {"precision": self._precision}
        evs = self._estimator.run([pub], **kwargs).result()[0].data.evs
        return float(np.asarray(evs).reshape(-1)[0])

    def sample(self, params: Sequence[float], shots: int, seed: int | None = None) -> dict[tuple[int, ...], int]:
        # ``seed`` only applies to the default sampler; an injected sampler owns its own seeding.
        if seed is not None and not self._custom_sampler:
            from qiskit.primitives import StatevectorSampler

            self._sampler = StatevectorSampler(seed=seed)
        pub = (self.measured_circuit, [self._values(self.measured_circuit, params)])
        data = self._sampler.run([pub], shots=shots).result()[0].data
        counts = getattr(data, "meas").get_counts()
        out: dict[tuple[int, ...], int] = {}
        for key, c in counts.items():
            key = key.replace(" ", "")
            # Qiskit bitstrings are little-endian: the right-most character is qubit 0.
            bits = tuple(int(key[self.n - 1 - q]) for q in range(self.n))
            out[bits] = out.get(bits, 0) + int(c)
        return out
