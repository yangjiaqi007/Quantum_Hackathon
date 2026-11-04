from itertools import product
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from Schema import CompiledProblem, LongRangeCoupling, Voicing
from person4.backend.hardware import ExecutionContext, HardwareConfigurationError, create_execution_context
from person4.backend.reporting import build_quality_report


def problem():
    v = Voicing((48, 55, 60, 64), "I")
    return CompiledProblem(
        candidates=[[v, v], [v, v, v], [v, v]],
        h=[np.array([-1., .5]), np.array([0., -.3, .6]), np.array([.2, -.5])],
        W=[np.array([[2., .1, -.4], [.2, .8, .1]]),
           np.array([[.4, -.2], [0., 1.], [1., -.3]])],
        energy_scale=10,
    )


def test_chain_checker_matches_brute_force_with_ragged_negative_costs():
    p = problem()
    report = build_quality_report(p, [0, 0, 0])
    optimum = min(p.energy(s) for s in product(range(2), range(3), range(2)))
    assert report["optimal_energy"] == pytest.approx(optimum)
    assert p.energy(report["optimal_selection"]) == pytest.approx(optimum)
    assert report["raw_energy"] == pytest.approx(p.energy([0, 0, 0]) * 10)
    assert report["energy_gap"] == pytest.approx(p.energy([0, 0, 0]) - optimum)


def test_long_range_is_not_silently_ignored():
    p = problem()
    p.long_range = [LongRangeCoupling(0, 0, 2, 1, 20)]
    report = build_quality_report(p, [0, 0, 0])
    assert report["benchmark_method"] == "exact_enumeration"
    assert report["optimal_energy"] == min(p.energy(s) for s in product(range(2), range(3), range(2)))
    skipped = build_quality_report(p, [0, 0, 0], exhaustive_limit=1)
    assert skipped["optimal_energy"] is None
    assert skipped["energy_gap"] is None
    assert skipped["benchmark_skipped_reason"]


@pytest.mark.parametrize("selection", [[-1, 0, 0], [0, 3, 0], [0, 0], [True, 0, 0]])
def test_invalid_selection_rejected(selection):
    with pytest.raises(ValueError):
        build_quality_report(problem(), selection)


def test_nonfinite_cost_rejected():
    p = problem()
    p.h[0][0] = np.nan
    with pytest.raises(ValueError):
        build_quality_report(p, [0, 0, 0])


def test_ibm_requires_explicit_server_enablement_before_import():
    with pytest.raises(HardwareConfigurationError, match="disabled"):
        create_execution_context("ibm", shots=32, seed=1, settings=SimpleNamespace(ibm_enabled=False))


def test_ibm_requires_credentials_before_import():
    with pytest.raises(HardwareConfigurationError, match="server-side"):
        create_execution_context("ibm", shots=32, seed=1, settings=SimpleNamespace(ibm_enabled=True))


def test_ibm_error_does_not_leak_credentials(monkeypatch):
    def broken_service(**kwargs):
        raise RuntimeError("secret-token in exception")
    monkeypatch.setitem(sys.modules, "qiskit_ibm_runtime", SimpleNamespace(
        QiskitRuntimeService=broken_service, SamplerV2=object))
    settings = SimpleNamespace(ibm_enabled=True, ibm_token="secret-token", ibm_instance="instance", ibm_backend="ibm_test")
    with pytest.raises(HardwareConfigurationError) as caught:
        create_execution_context("ibm", shots=32, seed=1, settings=settings)
    assert "secret-token" not in str(caught.value)


def test_resources_count_transpiled_gates_and_distinguish_widths():
    def instruction(name, qubits):
        return SimpleNamespace(operation=SimpleNamespace(name=name, num_qubits=len(qubits)), qubits=qubits)
    circuit = SimpleNamespace(
        metadata={"harmony_resources": {"logical_qubits": 2}},
        num_qubits=5, depth=lambda: 4,
        data=[instruction("sx", [3]), instruction("cx", [3, 4]), instruction("barrier", [3, 4])],
        find_bit=lambda bit: SimpleNamespace(index=bit),
    )
    ctx = ExecutionContext("noisy", "fake", None, None)
    result = ctx.resources(circuit, reps=1, shots=128)
    assert result.n_qubits == 2
    assert result.two_qubit_gates == 1
    assert result.extra["physical_register_width"] == 5
    assert result.extra["active_physical_qubits"] == 2
    assert ctx.estimated_runtime_seconds is None


def test_resources_reject_untranspiled_circuit():
    ctx = ExecutionContext("aer", "aer", None, None)
    with pytest.raises(ValueError, match="transpile"):
        ctx.resources(SimpleNamespace(metadata={}), 1, 128)


def test_aer_bell_smoke_if_installed():
    qiskit = pytest.importorskip("qiskit")
    pytest.importorskip("qiskit_aer")
    ctx = create_execution_context("aer", shots=64, seed=42, settings=SimpleNamespace())
    circuit = qiskit.QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure_all()
    compiled = ctx.transpile(circuit)
    counts = ctx.sampler.run([compiled], shots=64).result()[0].data.meas.get_counts()
    assert sum(counts.values()) == 64
    assert set(counts) <= {"00", "11"}
    assert ctx.resources(compiled, 1, 64).two_qubit_gates >= 1


def test_noisy_sampler_smoke_if_installed():
    qiskit = pytest.importorskip('qiskit')
    pytest.importorskip('qiskit_aer')
    pytest.importorskip('qiskit_ibm_runtime')
    ctx = create_execution_context('noisy', shots=32, seed=42,
                                   settings=SimpleNamespace(fake_backend='FakeManilaV2'))
    circuit = qiskit.QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure_all()
    compiled = ctx.transpile(circuit)
    counts = ctx.sampler.run([compiled], shots=32).result()[0].data.meas.get_counts()
    assert sum(counts.values()) == 32
    assert ctx.resources(compiled, 1, 32).n_qubits == 2
