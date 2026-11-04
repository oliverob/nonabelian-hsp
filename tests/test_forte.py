"""Native Forte compilation, ZZ folding, histogram import and the hardware lock."""
import unittest
from unittest import mock

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from circuits import build_circuit
from theory import theory
import forte
from forte import (fold_zz, import_histogram, native_as_qiskit, native_gates,
                   native_gates_with_frame, probe_bit_order, validate_native)
from simulation import simulate


class NativeTests(unittest.TestCase):
    def test_native_operator_not_only_zero_state(self):
        circuit = QuantumCircuit(2)
        circuit.u(.2, -.8, 2.3, 0); circuit.u(-1.1, .7, .9, 1)
        circuit.cx(0, 1); circuit.cx(1, 0); circuit.rz(-.3, 0)
        for angle in [np.pi/2, -np.pi/2, np.pi, -np.pi]:
            circuit.u(angle, .4, -1.3, 0); circuit.cx(0, 1); circuit.u(angle, -2.1, .6, 1)
        # Partial ZZ angles: negative, beyond pi/2, and near +-pi.
        for angle in [.7, -.4, 2.3, -2.6, 3.1, -3.1]:
            circuit.rzz(angle, 0, 1); circuit.u(.3, angle, -.2, 0)
        gates, frame = native_gates_with_frame(circuit)
        native = native_as_qiskit(dict(qubits=2, gateset="native", circuit=gates))
        for qubit, angle in enumerate(frame):
            native.rz(angle, qubit)  # The virtual frame dropped before measurement.
        self.assertTrue(Operator(circuit).equiv(Operator(native)))

    def test_native_full_sampling_circuit(self):
        circuit = build_circuit(5, 2, 4, "pairs")
        body = dict(qubits=circuit.num_qubits, gateset="native", circuit=native_gates(circuit))
        probability, _ = simulate(native_as_qiskit(body, circuit.metadata), shots=16)
        np.testing.assert_allclose(probability, theory(5, 2, 4, "pairs"), atol=1e-11)

    def test_zne_folding_preserves_the_circuit(self):
        circuit = build_circuit(5, 2, 3, "pairs")
        gates = native_gates(circuit)
        body = lambda g: dict(qubits=circuit.num_qubits, gateset="native", circuit=g)
        base = native_as_qiskit(body(gates))
        for fold in (3, 5):
            self.assertEqual(validate_native(body(fold_zz(gates, fold)))["zz"],
                             fold*validate_native(body(gates))["zz"])
            self.assertTrue(Operator(base).equiv(Operator(native_as_qiskit(body(fold_zz(gates, fold))))))
        with self.assertRaises(ValueError):
            fold_zz(gates, 2)

    def test_budget_boundary_and_unknown_gate(self):
        gate = dict(gate="zz", targets=[0, 1], angle=.25)
        cap = forte.MAX_ZZ
        self.assertEqual(validate_native(dict(qubits=2, gateset="native", circuit=[gate]*cap))["zz"], cap)
        with self.assertRaises(ValueError):
            validate_native(dict(qubits=2, gateset="native", circuit=[gate]*(cap+1)))
        with self.assertRaises(ValueError):
            validate_native(dict(qubits=2, gateset="native", circuit=[dict(gate="cx")]))


class ResultTests(unittest.TestCase):
    def test_bit_order_and_workspace_marginal(self):
        metadata = dict(x_bits=3, second_bits=1)
        # x=3, t=1, workspace=5, Qiskit integer=91, bit string 1011011.
        right = import_histogram({"1011011": 12}, metadata, 7, "q0-right")
        left = import_histogram({"1101101": 12}, metadata, 7, "q0-left")
        np.testing.assert_array_equal(right, left)
        self.assertEqual(right[3, 1], 12)
        self.assertEqual(right.sum(), 12)

    def test_probe_bit_order_needs_full_width_label(self):
        self.assertEqual(probe_bit_order({"100": 97, "000": 3}, 3), "q0-left")
        self.assertEqual(probe_bit_order({"001": 97, "101": 3}, 3), "q0-right")
        # A compacted label from dropped idle qubits cannot reveal the order.
        self.assertIsNone(probe_bit_order({"1": 100}, 3))

    def test_qpu_counts_from_probabilities(self):
        # IonQ may return a QPU job (a debiased one always) only as probabilities.
        job = dict(id="job", status="completed", shots=8, backend="qpu.forte-1",
                   results={"ionq.result.probabilities.json.v2": dict(id="combined")})
        responses = lambda probabilities: (lambda path, payload=None: job if path == "/jobs/job" else
                                           dict(probabilities=dict(registers=dict(output_all=probabilities))))
        with mock.patch("forte.api_request", side_effect=responses({"00": .625, "01": .25, "11": .125})):
            _, artifact = forte.fetch_histogram("job")
        self.assertEqual(forte.output_histogram(artifact), {"00": 5, "01": 2, "11": 1})
        # A probability that is not a whole number of the shots is refused, not rounded.
        with mock.patch("forte.api_request", side_effect=responses({"00": .6, "11": .4})):
            with self.assertRaisesRegex(ValueError, "whole number"):
                forte.fetch_histogram("job")
        # A simulator's probabilities are never turned into counts.
        with mock.patch("forte.api_request", side_effect=responses({"00": 1.0})):
            with mock.patch.dict(job, backend="simulator"):
                with self.assertRaisesRegex(ValueError, "relabeled"):
                    forte.fetch_histogram("job")


class TargetTests(unittest.TestCase):
    def test_payloads_differ_only_in_backend(self):
        body = dict(qubits=3, gateset="native", circuit=[])
        emulator = forte.job_payload("x", "emulator", 10, body, {})
        hardware = forte.job_payload("x", "hardware", 10, body, {})
        self.assertEqual(emulator["backend"], "simulator")
        self.assertEqual(emulator["noise"]["model"], "forte-1")
        self.assertEqual(hardware["backend"], "qpu.forte-1")
        self.assertNotIn("noise", hardware)
        self.assertEqual({k: v for k, v in emulator.items() if k not in ("backend", "noise")},
                         {k: v for k, v in hardware.items() if k != "backend"})

    def test_hardware_is_locked_even_for_dry_runs(self):
        self.assertFalse(forte.HARDWARE_SUBMISSION_ALLOWED)
        payload = forte.probe_payload("hardware")
        with mock.patch("forte.urlopen") as network:
            for body in (payload, dict(payload, dry_run=True)):
                with self.assertRaisesRegex(RuntimeError, "locked"):
                    forte.api_request("/jobs", body)
            network.assert_not_called()


if __name__ == "__main__":
    unittest.main()
