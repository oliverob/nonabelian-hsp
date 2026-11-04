"""Circuits against independent closed-form laws. Run: python -m unittest discover -s tests"""
import unittest

import json

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Operator, Statevector

from blocks import STORED, generic_diagonal, oracle_diagonal
from circuits import (bits, build_circuit, fourier, group_parameters,
                      oracle_angles, prepare_uniform, sparse_state, synthesize)
from simulation import simulate
from theory import theory


def zz_count(gate, width):
    circuit = QuantumCircuit(width)
    circuit.append(gate, range(width))
    return synthesize(circuit).count_ops().get("rzz", 0)


class CircuitTests(unittest.TestCase):
    def test_fingerprint_excludes_the_theory(self):
        # Editing the closed-form law must not invalidate prepared runs.
        import circuits
        self.assertNotIn("theory.py", circuits.SOURCES)
        self.assertFalse(hasattr(circuits, "theory"))

    def test_input_validation(self):
        for p, q in [(4, 2), (9, 2), (13, 5), (2, 1)]:
            with self.assertRaises(ValueError):
                group_parameters(p, q)

    def test_circuits_against_independent_formulas(self):
        for p, q in [(3, 2), (5, 4), (7, 2), (7, 3), (7, 6)]:
            for b in [0, 1, p-1]:
                for method in ["adapted", "pairs", "h_only", "forgetful", "weak"]:
                    with self.subTest(p=p, q=q, b=b, method=method):
                        probability, counts = simulate(build_circuit(p, q, b, method), shots=32)
                        np.testing.assert_allclose(probability, theory(p, q, b, method), atol=1e-12)
                        self.assertEqual(counts.sum(), 32)
                        self.assertAlmostEqual(probability.sum(), 1)
                        self.assertLess(probability[p:, :].sum(), 1e-12)

    def test_odd_q_supported_outside_pairs(self):
        for method in ["adapted", "forgetful"]:
            probability, _ = simulate(build_circuit(7, 3, 2, method), shots=16)
            np.testing.assert_allclose(probability, theory(7, 3, 2, method), atol=1e-12)

    def test_explicit_modular_oracle_reference(self):
        # Independently form |x,t,pi(x+alpha^t*b)> and Fourier-transform x.
        # Agreement verifies the omitted workspace DFT/permutation argument.
        p, q, b = 5, 2, 3
        _, powers = group_parameters(p, q)
        rng = np.random.default_rng(12)
        permutation = rng.permutation(p)
        vector = np.zeros(2**7, dtype=complex)
        for x in range(p):
            for t in range(q):
                work = permutation[(x+int(powers[t])*b) % p]
                vector[x+8*t+16*work] = 1/np.sqrt(p*q)
        transform = QuantumCircuit(7)
        transform.append(fourier(p), range(3))
        transform.t(3); transform.h(3)
        probability = Statevector(vector).evolve(transform).probabilities(range(4)).reshape(2, 8).T
        actual, _ = simulate(build_circuit(p, q, b, "pairs"), shots=16)
        np.testing.assert_allclose(actual, probability, atol=1e-12)

    def test_prepared_states_match_dense_reference(self):
        # Shukla-Vedula on the contiguous ranges, Gleinig-Hoefler on field labels.
        for p, q in [(5, 4), (7, 3), (7, 6), (13, 4)]:
            _, powers = group_parameters(p, q)
            for width, labels in [(bits(p), list(range(p))), (bits(q), list(range(q))),
                                  (bits(p), [int(value) for value in powers])]:
                with self.subTest(p=p, q=q, size=len(labels)):
                    want = np.zeros(2**width); want[labels] = 1/np.sqrt(len(labels))
                    circuit = QuantumCircuit(width)
                    prepare_uniform(circuit, circuit.qubits, labels)
                    np.testing.assert_allclose(Statevector(circuit).data, want, atol=1e-12)

    def test_field_preparation_never_costs_more_than_dense_synthesis(self):
        for p, q in [(7, 2), (7, 3), (7, 6), (13, 4)]:
            with self.subTest(p=p, q=q):
                _, powers = group_parameters(p, q)
                width = bits(p); labels = tuple(int(value) for value in powers)
                amplitudes = np.zeros(2**width); amplitudes[list(labels)] = 1/np.sqrt(q)
                dense = QuantumCircuit(width)
                dense.append(StatePreparation(amplitudes), dense.qubits)
                chosen = QuantumCircuit(width)
                chosen.append(sparse_state(width, labels), chosen.qubits)
                self.assertLessEqual(synthesize(chosen).count_ops().get("rzz", 0),
                                     synthesize(dense).count_ops().get("rzz", 0))

    def test_sparse_preparation_is_unitary(self):
        # The gate is appended to a register, so it must be unitary on every input.
        matrix = Operator(sparse_state(4, (1, 2, 7, 11, 13))).data
        np.testing.assert_allclose(matrix.conj().T@matrix, np.eye(16), atol=1e-12)

    def test_fourier_is_measurement_equivalent(self):
        # On labels < size: F_size up to one phase per outcome. Stored sizes keep their ZZ count.
        stored = json.loads(STORED.read_text())
        for size in [2, 3, 4, 5, 6, 7]:
            with self.subTest(size=size):
                unitary = Operator(fourier(size)).data
                np.testing.assert_allclose(unitary.conj().T@unitary, np.eye(len(unitary)), atol=1e-12)
                labels = np.arange(size)
                dft = np.exp(2j*np.pi*np.outer(labels, labels)/size)/np.sqrt(size)
                block = unitary[:size, :size]
                phases = np.sum(dft.conj()*block, axis=1)
                np.testing.assert_allclose(abs(phases), 1, atol=1e-12)
                np.testing.assert_allclose(block, phases[:, None]*dft, atol=1e-12)
                if str(size) in stored:
                    self.assertEqual(zz_count(fourier(size), bits(size)), stored[str(size)]["zz"])

    def test_oracle_diagonal_is_exact_and_cheaper(self):
        # Against the target phases on populated labels, up to a phase of w alone,
        # and never more ZZ than one generic DiagonalGate per work bit.
        p = 7
        for q in [2, 3, 6]:
            nt, nw = bits(q), bits(p)
            for b in [1, 3, 6]:
                with self.subTest(q=q, b=b):
                    angles = oracle_angles(p, q, b)
                    diagonal = np.diag(Operator(oracle_diagonal(angles)).data)
                    ratio = np.array([[diagonal[x + (w << nt)]*np.exp(
                        -1j*sum((w >> j & 1)*angles[j, x] for j in range(nw))) for x in range(q)]
                        for w in range(p)])
                    np.testing.assert_allclose(ratio, ratio[:, :1]*np.ones(q), atol=1e-10)
                    self.assertLessEqual(zz_count(oracle_diagonal(angles), nt+nw),
                                         zz_count(generic_diagonal(angles), nt+nw))

    def test_h_only_and_forgetful_degeneracies(self):
        np.testing.assert_allclose(theory(7, 2, 1, "h_only"), theory(7, 2, 6, "h_only"), atol=1e-12)
        alpha, _ = group_parameters(7, 3)
        np.testing.assert_allclose(theory(7, 3, 1, "forgetful"), theory(7, 3, alpha, "forgetful"), atol=1e-12)
        self.assertGreater(np.max(abs(theory(7, 2, 1, "pairs")-theory(7, 2, 6, "pairs"))), .01)


if __name__ == "__main__":
    unittest.main()
