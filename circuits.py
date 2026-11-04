"""Small, explicit circuits for the q-hedral hidden subgroup experiment.

Register order (least significant first): x, second register, oracle workspace.
Only x and the second register are measured; the workspace is traced out.
No secret-dependent state is supplied to the decoder. The Fourier transforms
and the exponent-encoding oracle come from blocks.py.
"""
from functools import lru_cache
import hashlib
from math import isqrt
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import StatePreparation
from qiskit.circuit.library.data_preparation import UniformSuperpositionGate
from pyencode import encode, SPARSE

from blocks import fourier, oracle_diagonal

METHODS = ("adapted", "pairs", "h_only", "forgetful", "weak")
SOURCES = ("circuits.py", "blocks.py", "fourier_circuits.json")


def fingerprint():
    """SHA-256 of the files that define the circuits; stamps caches and prepared runs."""
    digest = hashlib.sha256()
    for name in SOURCES:
        digest.update((Path(__file__).parent / name).read_bytes())
    return digest.hexdigest()


def bits(size):
    return max(1, (size - 1).bit_length())


def group_parameters(p, q):
    if p < 3 or any(p % d == 0 for d in range(2, isqrt(p) + 1)):
        raise ValueError("p must be an odd prime")
    if q < 2 or (p - 1) % q:
        raise ValueError("q must be a divisor of p-1, with q >= 2")
    alpha = next(a for a in range(2, p)
                 if pow(a, q, p) == 1
                 and all(pow(a, d, p) != 1 for d in range(1, q)))
    return alpha, np.array([pow(alpha, t, p) for t in range(q)])


@lru_cache(None)
def sparse_state(width, labels):
    """Gleinig-Hoefler on sparse labels, or dense state preparation if cheaper.

    PyEncode's Gleinig-Hoefler routine isolates two labels with a few controls,
    makes them differ in one qubit with CX gates, and merges them with one
    multi-controlled gate, so its cost follows the number of labels rather than
    the register width. That wins while the support is sparse, but a label set
    filling most of the register, as q approaches p does, is cheaper under
    Qiskit's generic synthesis. Both are exact, so the one that synthesizes to
    fewer two-qubit gates on this register is used.
    """
    sparse, _ = encode(SPARSE([(label, 1.0) for label in labels]), N=2 ** width)
    amplitudes = np.zeros(2 ** width)
    amplitudes[list(labels)] = 1 / np.sqrt(len(labels))
    dense = QuantumCircuit(width)
    dense.append(StatePreparation(amplitudes), dense.qubits)
    if synthesize(dense).count_ops().get("rzz", 0) < synthesize(sparse).count_ops().get("rzz", 0):
        return dense.to_gate(label=f"D{len(labels)}")
    return sparse.to_gate(label=f"S{len(labels)}")


@lru_cache(None)
def uniform_range(size, width):
    """Shukla-Vedula uniform superposition on 0,...,size-1, in u/cx form.

    Qiskit 2.4.1 panics in its basis translator whenever the controlled
    Hadamards of that synthesis share a circuit with a dense `unitary`, as
    the embedded Fourier transforms are, so the definition is expanded here.
    """
    definition = UniformSuperpositionGate(size, width).definition
    return transpile(definition, basis_gates=["u", "cx"], optimization_level=1,
                     seed_transpiler=20260923).to_gate(label=f"U{size}")


def prepare_uniform(circuit, qubits, labels):
    """Shukla-Vedula on a contiguous range; Gleinig-Hoefler on sparse labels."""
    labels = tuple(int(value) for value in labels)
    if labels == tuple(range(len(labels))):
        circuit.append(uniform_range(len(labels), len(qubits)), qubits)
    else:
        circuit.append(sparse_state(len(qubits), labels), qubits)


def oracle_angles(p, q, b):
    """Exponent-encoding oracle phases: [j, t] = 2 pi (b 2^j alpha^t mod p)/p.

    The diagonal applies phase sum_j w_j angles[j, t] on (t, w); NaN marks
    register states that never hold a label, whose phase is free.
    """
    _, powers = group_parameters(p, q)
    angles = np.full((bits(p), 2 ** bits(q)), np.nan)
    for j in range(bits(p)):
        angles[j, :q] = 2 * np.pi * ((b * 2 ** j * powers) % p) / p
    return angles


def build_circuit(p, q, b, method="pairs", measure=True):
    alpha, powers = group_parameters(p, q)
    if method not in METHODS or not 0 <= b < p:
        raise ValueError("unknown method or b outside 0,...,p-1")
    field = method == "adapted"
    x = QuantumRegister(bits(p), "x")
    second = QuantumRegister(bits(p) if field else bits(q), "a" if field else "t")
    work = QuantumRegister(bits(p), "work")
    circuit = QuantumCircuit(x, second, work, name=f"{method}_p{p}_q{q}_b{b}")
    prepare_uniform(circuit, x, range(p))
    prepare_uniform(circuit, second, powers if field else range(q))
    prepare_uniform(circuit, work, range(p))

    # Fourier-conjugated oracle. Discarding work makes its final inverse DFT
    # and arbitrary label permutation irrelevant to all measured statistics.
    for i, control in enumerate(x):
        for j, target in enumerate(work):
            circuit.cp(2 * np.pi * ((2 ** (i + j)) % p) / p, control, target)
    if field:
        for i, control in enumerate(second):
            for j, target in enumerate(work):
                angle = 2 * np.pi * ((b * 2 ** (i + j)) % p) / p
                if angle:
                    circuit.cp(angle, control, target)
    elif b:
        circuit.append(oracle_diagonal(oracle_angles(p, q, b)), [*second, *work])

    circuit.append(fourier(p), x)
    if method == "adapted":
        circuit.append(fourier(p), second)
    elif method in ("pairs", "h_only"):
        if method == "pairs":
            circuit.t(second[0])
        circuit.h(second[0])
    elif method == "forgetful":
        circuit.append(fourier(q), second)
    if measure:
        classical = ClassicalRegister(len(x) + len(second), "sample")
        circuit.add_register(classical)
        circuit.measure([*x, *second], classical)
    circuit.metadata = dict(p=p, q=q, b=b, alpha=alpha, method=method,
                            x_bits=len(x), second_bits=len(second),
                            measured_qubits=len(x) + len(second))
    return circuit


def synthesize(circuit):
    """Exact local synthesis; every arbitrary-angle RZZ is one native ZZ."""
    return transpile(circuit, basis_gates=["u", "rzz"], optimization_level=2,
                     seed_transpiler=20260923)
