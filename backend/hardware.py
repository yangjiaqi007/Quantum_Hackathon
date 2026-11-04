"""Optional execution targets for Person 2's Qiskit circuits.

No Qiskit import or IBM connection happens at API startup. Only an explicit
``create_execution_context`` call prepares a target. Credentials come from
server settings, never from a melody request, and are never saved to disk.

Official references:
https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.primitives.SamplerV2.html
https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/sampler-v2
https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/fake-provider
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from Schema import ResourceReport


FAKE_BACKENDS = frozenset({"FakeManilaV2", "FakeGuadalupeV2"})
_RESOURCE_METADATA = "harmony_resources"


class HardwareConfigurationError(RuntimeError):
    """An actionable, credential-free configuration error for the caller."""


@dataclass
class ExecutionContext:
    mode: str
    backend_name: str
    sampler: Any = field(repr=False)
    backend: Any = field(repr=False)
    seed: int = 42
    # Queue delay and optimizer iterations are unknown in advance. The API
    # should expose null, not a made-up duration or a circuit-depth estimate.
    estimated_runtime_seconds: float | None = None

    def transpile(self, circuit: Any) -> Any:
        """Return a backend-compatible circuit, retaining logical width metadata.

        Keep classical measurement bits in the original variable order. Person 2
        must decode measurement bits, not infer choices from physical placement.
        """
        capacity = getattr(self.backend, "num_qubits", None)
        if capacity is not None and circuit.num_qubits > capacity:
            raise HardwareConfigurationError(
                f"Circuit needs {circuit.num_qubits} qubits; {self.backend_name} "
                f"supports {capacity}. Reduce the window size or candidate count."
            )
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

        manager = generate_preset_pass_manager(
            backend=self.backend, optimization_level=1, seed_transpiler=self.seed
        )
        compiled = manager.run(circuit)
        compiled.metadata = dict(compiled.metadata or {})
        compiled.metadata[_RESOURCE_METADATA] = {"logical_qubits": circuit.num_qubits}
        return compiled

    def resources(self, circuit: Any, reps: int, shots: int) -> ResourceReport:
        """Measure the *transpiled* circuit, not the abstract QAOA ansatz.

        n_qubits is logical input width; physical register width can include
        idle ancillas. Both physical width and actually used qubits are recorded
        separately. Shots are per final sampling call, not the optimizer total.
        """
        info = (circuit.metadata or {}).get(_RESOURCE_METADATA)
        if info is None:
            raise ValueError("Call context.transpile(circuit) before resources().")
        if shots < 1 or reps < 1:
            raise ValueError("shots and reps must be positive")
        used = set()
        two_qubit = 0
        for instruction in circuit.data:
            if instruction.operation.name in {"barrier", "delay"}:
                continue
            used.update(circuit.find_bit(bit).index for bit in instruction.qubits)
            if instruction.operation.num_qubits == 2:
                two_qubit += 1
        return ResourceReport(
            n_qubits=int(info["logical_qubits"]),
            depth=int(circuit.depth() or 0),
            two_qubit_gates=two_qubit,
            shots=shots,
            reps=reps,
            backend=self.backend_name,
            extra={
                "physical_register_width": float(circuit.num_qubits),
                "active_physical_qubits": float(len(used)),
            },
        )


def create_execution_context(
    mode: str, *, shots: int, seed: int, settings: Any
) -> ExecutionContext:
    """Create an Aer, fake-device-noise, or explicitly enabled IBM target.

    ``noisy`` uses packaged historical calibration snapshots; it neither
    refreshes them nor contacts IBM. FakeManilaV2 has five qubits, so use
    FakeGuadalupeV2 (16 qubits) for larger one-hot windows.
    """
    if mode not in {"aer", "noisy", "ibm"}:
        raise HardwareConfigurationError(f"Unsupported execution target: {mode}")
    if shots < 1 or seed < 0:
        raise HardwareConfigurationError("shots must be positive and seed nonnegative")
    if mode == "ibm":
        if not getattr(settings, "ibm_enabled", False):
            raise HardwareConfigurationError("IBM hardware is disabled in server settings.")
        if not all(getattr(settings, name, None) for name in
                   ("ibm_token", "ibm_instance", "ibm_backend")):
            raise HardwareConfigurationError(
                "IBM hardware requires server-side token, instance, and backend settings."
            )
    try:
        if mode == "aer":
            from qiskit_aer import AerSimulator
            from qiskit_aer.primitives import SamplerV2

            backend = AerSimulator(method="automatic")
            sampler = SamplerV2(default_shots=shots, seed=seed)
            return ExecutionContext(mode, "aer", sampler, backend, seed)
        if mode == "noisy":
            from qiskit_aer import AerSimulator
            from qiskit_aer.primitives import SamplerV2
            from qiskit_ibm_runtime import fake_provider

            name = getattr(settings, "fake_backend", "FakeManilaV2")
            if name not in FAKE_BACKENDS:
                raise HardwareConfigurationError(
                    "Unsupported fake backend. Choose FakeManilaV2 or FakeGuadalupeV2."
                )
            backend = getattr(fake_provider, name)()
            simulator = AerSimulator.from_backend(backend)
            sampler = SamplerV2.from_backend(simulator, default_shots=shots, seed=seed)
            return ExecutionContext(mode, f"noisy:{name}", sampler, backend, seed)
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    except ImportError as exc:
        raise HardwareConfigurationError(
            "Quantum packages are missing. Install requirements-quantum.txt."
        ) from exc

    # Never echo SDK exception text: it may contain connection details. This
    # branch runs only on an explicitly requested, server-enabled hardware job.
    try:
        service = QiskitRuntimeService(
            channel="ibm_quantum_platform",
            token=settings.ibm_token,
            instance=settings.ibm_instance,
        )
        backend = service.backend(settings.ibm_backend)
        sampler = SamplerV2(
            mode=backend,
            options={
                "default_shots": shots,
                "max_execution_time": getattr(settings, "ibm_max_execution_time", 300),
            },
        )
    except Exception:
        raise HardwareConfigurationError(
            "Could not initialize IBM hardware. Check the server credentials, "
            "instance access, backend name, and network connection."
        ) from None
    return ExecutionContext(mode, settings.ibm_backend, sampler, backend, seed)
