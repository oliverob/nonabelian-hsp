"""Low-ZZ building blocks: the Fourier transforms and the exponent-encoding oracle.

Both blocks exploit freedoms of this experiment that a generic compiler cannot
see, and both are exact.

fourier(size)
    Every Fourier transform is applied to a register holding labels
    0,...,size-1 and is followed directly by its measurement. The circuit
    therefore only has to act as F_size on those labels up to a phase on each
    outcome. For the sizes in the plan the circuits come from
    fourier_search.py (stored in fourier_circuits.json): 2 ZZ for F_3, 8 for
    F_5, 7 for F_6 and 10 for F_7, against 3, 19, 19 and 19 for Qiskit's
    synthesis of the embedded unitary F_size + I. F_2 is the Hadamard gate.
    Sizes beyond the plan (only simulated ideally) fall back to that unitary.

oracle_diagonal(angles)
    The oracle's diagonal on (t, w): phase sum_j w_j f_j(t). Written in the
    Walsh basis, a diagonal is exp(i sum_S c_S Z_S); each term of weight two
    or more costs one ZZ once its parity is carried by a qubit. Phases at
    unpopulated t labels are free and are chosen to cancel terms, and phases
    of w alone are dropped (the workspace is discarded). The remaining terms
    are applied along the shortest CNOT walk on t that visits every parity
    they need.
"""
from functools import lru_cache
from itertools import combinations, permutations, product
import json
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import DiagonalGate, UnitaryGate

STORED = Path(__file__).with_name("fourier_circuits.json")
TOLERANCE = 1e-9


def _register_width(labels):
    """The number of qubits needed to hold the labels 0, ..., labels - 1 (at least one)."""
    return max(1, (labels - 1).bit_length())


# ======================================================================
# Fourier transforms
# ======================================================================

def dense_fourier(size):
    """The embedded unitary F_size + I as a UnitaryGate (Qiskit synthesizes it generically)."""
    width = _register_width(size)
    matrix = np.eye(2 ** width, dtype=complex)
    labels = np.arange(size)
    matrix[:size, :size] = np.exp(2j * np.pi * np.outer(labels, labels) / size) / np.sqrt(size)
    return UnitaryGate(matrix, label=f"F{size}")


@lru_cache(None)
def fourier(size):
    """F_size on labels 0,...,size-1 up to a phase on each outcome (see module doc)."""
    if size == 2:
        # F_2 is exactly the Hadamard gate.
        circuit = QuantumCircuit(1, name="F2")
        circuit.h(0)
        return circuit.to_gate(label="F2")
    stored_circuits = json.loads(STORED.read_text())
    if str(size) not in stored_circuits:
        return dense_fourier(size)
    circuit = QuantumCircuit(_register_width(size), name=f"F{size}")
    for gate in stored_circuits[str(size)]["gates"]:
        if gate[0] == "u":
            _, qubit, theta, phi, lam = gate
            circuit.u(theta, phi, lam, qubit)
        else:
            _, first_qubit, second_qubit, angle = gate
            circuit.rzz(angle, first_qubit, second_qubit)
    return circuit.to_gate(label=f"F{size}")


# ======================================================================
# The exponent-encoding oracle
# ======================================================================
#
# What the gate must do
#   There are two registers: t, with `width` qubits holding a label t, and w,
#   with one qubit per work bit j. The gate multiplies |t, w> by
#   exp(i sum_j w_j f_j(t)), where f_j = angles[j]. Two things are free:
#   labels marked NaN never occur, so the phase there can be anything, and a
#   phase that depends on w alone is harmless because w is discarded.
#
# Parities
#   A parity is a set of t qubits, stored as a bitmask S: bit i of S is set
#   when qubit i is in the set. For example S = 0b101 is {t_0, t_2}. Z_S is
#   the product of Z over those qubits. On the basis state |t> it gives the
#   sign (-1)^n, where n counts the qubits in S whose bit of t is 1.
#
# Splitting the phase into terms
#   Any function of t can be written as f(t) = sum_S c_S Z_S, where c = W f
#   are its Walsh coefficients. Since w_j = (1 - Z_wj)/2,
#       sum_j w_j f_j(t) = sum_j f_j(t)/2  -  sum_j sum_S (c_S of f_j)/2 Z_S Z_wj.
#   So the gate is a product of commuting terms, one of each kind per S >= 1:
#       work terms  exp(i a_S Z_S Z_wj),  a_S = -(c_S of f_j)/2,  for each j
#       t terms     exp(i d_S Z_S),       d_S = -(sum over j of a_S)
#   The S = 0 terms are a phase of w alone and a global phase; both are dropped.
#
# What each term costs
#   A term whose angle is zero needs nothing. Otherwise a work term needs one
#   RZZ gate between w_j and a t qubit that currently holds the parity S, and
#   a t term needs an RZ gate on such a qubit. At the start qubit i holds its
#   own bit; CNOT gates on t combine bits so that other parities appear. This
#   sequence of CNOTs is the "walk", and it must end with t back as it
#   started. A walk "serves" a set of terms when it visits every parity at
#   which they need a rotation. The cost is the number of two-qubit gates:
#   RZZ gates plus CNOT gates.
#
# Using the free labels
#   Every Walsh coefficient depends linearly on the values at the free labels.
#   With k free labels we can therefore choose values that make k chosen
#   coefficients vanish. Each such choice gives the work bit an "option": its
#   work-term angles with those k terms cancelled. Different work bits may
#   choose differently, since each f_j has its own values at the free labels.
#
# Choosing the walk
#   Every option of a work bit cancels the same number of terms, so all of
#   them need the same number of RZZ gates. (This assumes no term vanishes by
#   coincidence, which holds for every oracle at p <= 7.) The cheapest circuit
#   is therefore the shortest walk that serves some option of every work bit.
#   The t terms come from the same options, and since d_S = -(sum of a_S) they
#   need a rotation only at a parity where some work term does, so the same
#   walk serves them.


def generic_diagonal(angles):
    """The same oracle as one Qiskit DiagonalGate per work bit (free labels get phase 0)."""
    phases = np.nan_to_num(np.asarray(angles, dtype=float))
    work_bits, labels = phases.shape
    width = _register_width(labels)
    circuit = QuantumCircuit(width + work_bits, name="D")
    for j in range(work_bits):
        # On (t, w_j), with w_j the highest qubit: phase 1 when w_j = 0 and
        # exp(i f_j(t)) when w_j = 1.
        diagonal = np.exp(1j * np.concatenate([np.zeros_like(phases[j]), phases[j]]))
        circuit.append(DiagonalGate(list(diagonal)), [*range(width), width + j])
    return circuit.to_gate(label="D")


def oracle_diagonal(angles):
    """Gate on (t qubits, w qubits): phase sum_j w_j angles[j, t]; NaN marks a free t label.

    Exact on every populated label, up to a phase depending on w alone.
    """
    angles = np.asarray(angles, dtype=float)
    work_bits, labels = angles.shape
    width = _register_width(labels)
    if width > 3:
        # Beyond the plan: _shortest_walks only goes up to 3 t qubits.
        return generic_diagonal(angles)

    # Labels marked NaN never occur, so their phase is free. Start them at 0.
    free_labels = [label for label in range(labels) if np.isnan(angles[0, label])]
    phases = np.nan_to_num(angles)

    # Step 1: each work bit's options, one per way of using its free labels to cancel terms.
    options_per_work_bit = []
    for j in range(work_bits):
        options_per_work_bit.append(_options(phases[j], free_labels, width))

    # Step 2: the shortest walk that serves every work bit, and the option each work bit uses.
    walk, work_angles = _shortest_serving_walk(options_per_work_bit, width)

    # Step 3: the t terms, from the same options (d_S = -(sum over work bits of a_S)),
    # then the circuit.
    t_angles = -sum(work_angles)
    return _circuit(width, work_angles, t_angles, walk)


# ---------------------------------------------------------------- step 1: options

def _options(phases, free_labels, width):
    """One work bit's options: its work-term angles a_S = -c_S/2 for each way of
    filling its free labels that cancels one term per free label.

    `phases` holds 0 at the free labels. Writing x for the values put there,
    each Walsh coefficient is c_S = W[S] @ phases + W[S, free_labels] @ x: a
    known part plus a part linear in x. Cancelling a chosen set of terms is
    therefore a linear system for x, with one equation per free label, solved
    here for every possible set. (Cancelling fewer terms than there are free
    labels never gives a cheaper circuit, so those sets are not tried.)
    """
    walsh = _walsh_matrix(width)
    options = []
    for parities_to_cancel in combinations(_all_parities(width), len(free_labels)):
        rows = walsh[list(parities_to_cancel)]  # one row per coefficient to cancel
        known_part = rows @ phases
        effect_of_x = rows[:, free_labels]
        # Solve effect_of_x @ x = -known_part. When the system is singular,
        # these terms cannot be cancelled together, so skip this set.
        try:
            x = np.linalg.solve(effect_of_x, -known_part)
        except np.linalg.LinAlgError:
            continue
        filled = phases.copy()
        filled[free_labels] = x
        coefficients = walsh @ filled
        options.append(-coefficients / 2)
    return options


@lru_cache(None)
def _walsh_matrix(width):
    """The matrix W with c = W f the Walsh coefficients of f, so f(t) = sum_S c_S Z_S(t).

    W[S, t] = (-1)^n / 2^width, where n counts the qubits in S whose bit of t is 1.
    """
    size = 2 ** width
    matrix = np.zeros((size, size))
    for parity in range(size):
        for label in range(size):
            shared_ones = bin(parity & label).count("1")
            matrix[parity, label] = (-1) ** shared_ones / size
    return matrix


# ---------------------------------------------------------------- step 2: the walk

def _shortest_serving_walk(options_per_work_bit, width):
    """The shortest walk that serves some option of every work bit.

    Returns the walk and, for each work bit, the first of its options that the
    walk serves. _shortest_walks lists the walks shortest first, so the first
    walk that serves every work bit is the shortest. The walk through every
    parity serves any option, so one is always found.
    """
    for visited, walk in _shortest_walks(width).items():
        chosen = []
        for options in options_per_work_bit:
            served = [option for option in options if _needed_parities(option, width) <= visited]
            if served:
                chosen.append(served[0])
        if len(chosen) == len(options_per_work_bit):
            return walk, chosen


@lru_cache(None)
def _shortest_walks(width):
    """For each set of parities a CNOT walk can visit, the shortest such walk.

    Returns {frozenset of visited parities: tuple of (control, target) CNOTs}.
    Qubit i starts out holding its own bit, the parity 1 << i. A CNOT from
    control to target makes the target hold the combined parity of the two
    (their XOR). A walk visits every parity that some qubit holds along the
    way, and must end with every qubit holding its own bit again, so that t
    is unchanged.

    Every walk of 0 to 6 CNOTs is tried, shortest first, and the first one
    found for each set of visited parities is kept, so the walks come out
    listed shortest first. Six is enough: a walk of six visits all 7 parities
    of 3 qubits, so no longer walk can be cheaper.
    """
    starting_parities = [1 << qubit for qubit in range(width)]
    all_cnots = list(permutations(range(width), 2))  # every (control, target) pair
    shortest = {}
    for length in range(7):
        for walk in product(all_cnots, repeat=length):
            held = list(starting_parities)
            visited = set(starting_parities)
            for control, target in walk:
                held[target] = held[target] ^ held[control]
                visited.add(held[target])
            visited = frozenset(visited)
            if held == starting_parities and visited not in shortest:
                shortest[visited] = walk
    return shortest


# ---------------------------------------------------------------- step 3: the circuit

def _circuit(width, work_angles, t_angles, walk):
    """Follow the walk, adding every rotation for a parity the first time a t qubit holds it.

    While qubit i holds parity S, a Z on qubit i acts as Z_S. So rzz(-2a) on
    (i, w_j) gives exp(i a Z_S Z_wj), and rz(-2d) on i gives exp(i d Z_S).
    """
    circuit = QuantumCircuit(width + len(work_angles), name="D")
    held = [1 << qubit for qubit in range(width)]  # the parity each t qubit holds
    done = set()  # parities whose rotations have been added

    def add_rotations_for_new_parities():
        for qubit, parity in enumerate(held):
            if parity in done:
                continue
            done.add(parity)
            for j, angles in enumerate(work_angles):
                if _needs_rotation(angles[parity]):
                    circuit.rzz(-2 * angles[parity], qubit, width + j)
            if _needs_rotation(t_angles[parity]):
                circuit.rz(-2 * t_angles[parity], qubit)

    add_rotations_for_new_parities()
    for control, target in walk:
        circuit.cx(control, target)
        held[target] = held[target] ^ held[control]
        add_rotations_for_new_parities()

    # The walk serves every term, so every rotation has been added.
    for angles in [*work_angles, t_angles]:
        assert _needed_parities(angles, width).issubset(done)
    return circuit.to_gate(label="D")


# ---------------------------------------------------------------- parities and angles

def _all_parities(width):
    """Every non-empty parity of `width` qubits: 1, 2, ..., 2^width - 1."""
    return range(1, 2 ** width)


def _needs_rotation(angle):
    """Whether the term exp(i angle Z..Z) needs a rotation gate: every term except a zero one."""
    return abs(angle) >= TOLERANCE


def _needed_parities(term_angles, width):
    """The parities S whose term, with angle term_angles[S], needs a rotation gate.

    A walk must visit all of these to serve the terms.
    """
    needed = set()
    for parity in _all_parities(width):
        if _needs_rotation(term_angles[parity]):
            needed.add(parity)
    return needed
