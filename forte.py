"""IonQ Forte: native-gate compilation, job payloads, and the IonQ API (v0.4).

Compilation (offline)
    native_gates() turns a logical circuit into Forte's native gate list
    (GPi, GPi2, arbitrary-angle ZZ; angles in turns), fold_zz() makes the
    ZZ-folded copy used for zero-noise extrapolation, and validate_native()
    audits a gate list. native_as_qiskit() rebuilds the serialized gates as a
    Qiskit circuit so every job can be re-simulated before it is sent.

Targets
    "emulator": IonQ's cloud simulator with the forte-1 noise model.
    "hardware": the qpu.forte-1 trapped-ion computer, with IonQ's debiasing.
    The payloads for the two carry the same gate lists; they differ in their
    backend fields and in the debiasing setting (see job_payload).

API (online; needs IONQ_API_KEY)
    Every request goes through api_request(). It refuses to send any job,
    dry runs included, to real hardware while HARDWARE_SUBMISSION_ALLOWED is
    False. Set it to True only when you intend to spend hardware credit.
"""
from collections import Counter
import json
import os
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Optimize1qGatesDecomposition

from circuits import synthesize

# Safety switch: while False no job (not even a free dry run) can reach a QPU.
HARDWARE_SUBMISSION_ALLOWED = False

MAX_ZZ = 1000  # IonQ's hard cap on ZZ gates per hardware job; the emulator has none.
API = "https://api.ionq.co/v0.4"
NOISE_SEED = 20260923
TARGETS = {
    "emulator": dict(backend="simulator", noise=dict(model="forte-1", seed=NOISE_SEED)),
    "hardware": dict(backend="qpu.forte-1"),
}
# IonQ's debiasing: each hardware job runs as symmetrized variants (qubit
# assignments, gate decompositions; a 2,048-shot job ran as 32 variants of 64
# shots) whose counts are combined, and is returned only as probabilities
# (see qpu_counts). It needs at least 500 shots and is not available on
# the simulator.
# With it on, IonQ's minimum price per job rises (168.20 USD on Forte, 2026-09).
DEBIAS_HARDWARE = True
DEBIAS_MIN_SHOTS = 500
# Recorded in each fetched histogram; analysis.py refuses to mix them up.
SOURCES = dict(emulator="IonQ Forte noisy simulator histogram",
               hardware="IonQ Forte debiased hardware histogram")


# ---------------------------------------------------------------- compilation

def native_gates(circuit):
    """Forte native gate list for `circuit` (final measurements are ignored)."""
    return native_gates_with_frame(circuit)[0]


def native_gates_with_frame(circuit):
    """Native gates and the dropped final Rz angle per qubit.

    Qiskit first synthesizes the circuit exactly to U and arbitrary-angle RZZ
    gates; each RZZ becomes one native ZZ. RZZ(a) = exp(-i a/2 ZZ) is native
    ZZ(a/2pi turns), allowed in [0, 1/4]: RZZ(a) = RZZ(a -/+ pi) Z x Z folds a
    into [-pi/2, pi/2], and RZZ(-a) = X_0 RZZ(a) X_0 makes it nonnegative. The
    added Z and X gates merge into neighbouring one-qubit gates for free.

    Rz is virtual: it shifts the phases of later pulses on its qubit
    (Rz(-f) R_axis(psi) Rz(f) = R_axis(psi - f)) and commutes with ZZ. So each
    U costs at most two GPi2 pulses, using Ry(a) = Rx(-pi/2) Rz(a) Rx(pi/2).
    The final Rz frame is diagonal before a computational-basis measurement
    and is dropped: circuit == Rz(frame) after the returned gates, up to phase.
    """
    synthesized = synthesize(circuit.remove_final_measurements(inplace=False))
    if synthesized.count_ops().get("rzz", 0) > MAX_ZZ:
        raise ValueError(f"{circuit.name}: {synthesized.count_ops()['rzz']} ZZ exceeds {MAX_ZZ}")
    folded = QuantumCircuit(circuit.num_qubits)
    for instruction in synthesized.data:
        targets = [synthesized.find_bit(q).index for q in instruction.qubits]
        if instruction.operation.name != "rzz":
            folded.append(instruction.operation, targets)
            continue
        angle = (float(instruction.operation.params[0]) + np.pi) % (2*np.pi) - np.pi
        if abs(angle) > np.pi/2:
            angle -= np.copysign(np.pi, angle)
            folded.z(targets)
        if abs(angle) < 1e-12:
            continue
        if angle < 0:
            folded.x(targets[0])
        folded.rzz(abs(angle), *targets)
        if angle < 0:
            folded.x(targets[0])
    folded = PassManager([Optimize1qGatesDecomposition(basis=["u"])]).run(folded)
    gates = []
    frame = np.zeros(circuit.num_qubits)

    def pulse(name, target, phase):
        phase = (phase - frame[target]/(2*np.pi)) % 1
        gates.append(dict(gate=name, target=target, phase=float(phase)))

    def close(a, b):
        return abs((a - b + np.pi) % (2*np.pi) - np.pi) < 1e-12

    for instruction in folded.data:
        targets = [folded.find_bit(q).index for q in instruction.qubits]
        params = [float(p) for p in instruction.operation.params]
        if instruction.operation.name == "rzz":
            gates.append(dict(gate="zz", targets=targets, angle=params[0]/(2*np.pi)))
        elif instruction.operation.name == "u":
            theta, phi, lam = params
            target = targets[0]
            frame[target] += lam
            if close(theta, np.pi/2):
                pulse("gpi2", target, .25)
            elif close(theta, -np.pi/2):
                pulse("gpi2", target, .75)
            elif close(theta, np.pi):
                pulse("gpi", target, .25)
            elif not close(theta, 0):
                pulse("gpi2", target, 0)
                frame[target] += theta
                pulse("gpi2", target, .5)
            frame[target] += phi
        else:
            raise ValueError(f"Untranslated gate: {instruction.operation.name}")
    validate_native(dict(qubits=circuit.num_qubits, gateset="native", circuit=gates))
    return gates, frame


def fold_zz(gates, fold):
    """Zero-noise extrapolation: replace each ZZ by ZZ (ZZ^-1 ZZ)^((fold-1)/2).

    ZZ(a)^-1 = ZZ(-a) = GPi ZZ(a) GPi on either qubit, since GPi(phase) is
    Hermitian and anticommutes with Z; this keeps every angle in [0, 1/4].
    Each fold adds two ZZ and two GPi pulses per original ZZ, so the ZZ
    count scales by `fold` and the logical circuit is unchanged.
    """
    if fold < 1 or fold % 2 == 0:
        raise ValueError("ZNE fold must be an odd integer >= 1")
    folded = []
    for gate in gates:
        folded.append(gate)
        if gate["gate"] != "zz":
            continue
        flip = dict(gate="gpi", target=gate["targets"][0], phase=0.0)
        folded.extend([flip, dict(gate), flip, dict(gate)] * ((fold-1)//2))
    return folded


def validate_native(body, max_zz=MAX_ZZ):
    """Audit a native gate list and return its gate counts.

    max_zz=None lifts the ZZ cap; hardware jobs must respect it.
    """
    if body["gateset"] != "native":
        raise ValueError("Only audited native circuits may be submitted")
    counts = Counter()
    for gate in body["circuit"]:
        name = gate["gate"]
        if name not in ("gpi", "gpi2", "zz"):
            raise ValueError(f"Non-Forte-native gate: {name}")
        targets = gate["targets"] if name == "zz" else [gate["target"]]
        if len(targets) != (2 if name == "zz" else 1) or len(set(targets)) != len(targets):
            raise ValueError("Invalid gate arity")
        if any(not isinstance(t, int) or not 0 <= t < body["qubits"] for t in targets):
            raise ValueError("Qubit index outside circuit")
        angle = gate["angle"] if name == "zz" else gate["phase"]
        if not np.isfinite(angle) or (name == "zz" and not 0 <= angle <= .25):
            raise ValueError("Invalid native angle")
        if name != "zz" and not -1 <= angle <= 1:
            raise ValueError("Invalid native phase")
        counts[name] += 1
    if max_zz is not None and counts["zz"] > max_zz:
        raise ValueError(f"{counts['zz']} ZZ gates exceeds the hard limit {max_zz}")
    return counts


def native_as_qiskit(body, metadata=None):
    """Rebuild the *serialized* native gates as a Qiskit circuit, for checking.

    With circuit metadata (from circuits.build_circuit) the measured
    registers are added, so simulation.simulate() can run it.
    """
    validate_native(body, None)
    circuit = QuantumCircuit(body["qubits"])
    for gate in body["circuit"]:
        if gate["gate"] == "zz":
            circuit.rzz(2*np.pi*gate["angle"], *gate["targets"])
        else:
            phase = 2*np.pi*gate["phase"]
            gpi = np.array([[0, np.exp(-1j*phase)], [np.exp(1j*phase), 0]])
            matrix = gpi if gate["gate"] == "gpi" else (np.eye(2)-1j*gpi)/np.sqrt(2)
            circuit.append(UnitaryGate(matrix), [gate["target"]])
    if metadata is not None:
        circuit.metadata = metadata
        measured = metadata["measured_qubits"]
        circuit.add_register(ClassicalRegister(measured, "sample"))
        circuit.measure(range(measured), range(measured))
    return circuit


# ------------------------------------------------------------------- payloads

def job_payload(name, target, shots, body, metadata):
    """An ionq.circuit.v1 job, debiased on hardware whenever it has enough shots.

    The bit-order probe (100 shots) is below IonQ's debiasing minimum, so it
    always runs without.
    """
    debiasing = DEBIAS_HARDWARE and target == "hardware" and shots >= DEBIAS_MIN_SHOTS
    settings = dict(TARGETS[target])
    return dict(type="ionq.circuit.v1", name=name, **settings, shots=shots,
                input=body, settings={"error_mitigation": {"debiasing": debiasing}},
                metadata={key: str(value) for key, value in metadata.items()})


def probe_payload(target):
    """A 3-qubit job that flips only qubit 0, to read off the histogram bit order.

    Qubits 1 and 2 get an identity (GPi twice) so they are not dropped as idle.
    """
    gates = [dict(gate="gpi", target=0, phase=0)] + [dict(gate="gpi", target=t, phase=0)
                                                     for t in (1, 1, 2, 2)]
    return job_payload("bit_order_probe", target, 100,
                       dict(qubits=3, gateset="native", circuit=gates),
                       dict(purpose="bit_order_probe"))


def is_hardware(payload):
    return payload.get("backend") != "simulator"


# ------------------------------------------------------------------------ API

def api_request(path, payload=None):
    """GET (payload None) or POST one IonQ API request; enforces the hardware lock."""
    if payload is not None and is_hardware(payload) and not HARDWARE_SUBMISSION_ALLOWED:
        raise RuntimeError("Hardware submission is locked: set HARDWARE_SUBMISSION_ALLOWED "
                           "= True in forte.py when you intend to run on the QPU")
    key = os.environ.get("IONQ_API_KEY")
    if not key:
        raise RuntimeError("Set IONQ_API_KEY in the environment (or in .env)")
    request = Request(API+path, data=None if payload is None else json.dumps(payload).encode(),
                      headers={"Authorization": f"apiKey {key}", "Content-Type": "application/json"})
    # No automatic retries: a timed-out POST must not create a duplicate paid job.
    try:
        with urlopen(request, timeout=60) as response:
            return json.load(response)
    except HTTPError as error:
        # Keep the type (job_cost checks the code) but show IonQ's explanation.
        error.msg = f"{error.msg}: {error.read().decode(errors='replace')[:2000]}"
        raise


def job_cost(job_id):
    """IonQ's cost record for a job, or None while it is still being compiled."""
    try:
        return api_request(f"/jobs/{job_id}/cost")
    except HTTPError as error:
        if error.code in (404, 409, 425):
            return None
        raise


def fetch_histogram(job_id):
    """(job record, counts artifact or None if the job has not completed).

    The emulator returns a raw counts histogram artifact. A QPU job may have
    none, only probabilities (a debiased job always does), and qpu_counts
    turns those back into the exact counts. A simulator's probabilities are
    never turned into counts.
    """
    job = api_request(f"/jobs/{job_id}")
    if job["status"] != "completed":
        return job, None
    descriptor = job["results"].get("ionq.result.histogram.json.v2")
    if descriptor is not None:
        return job, api_request(f"/jobs/{job_id}/artifacts/{descriptor['id']}")
    if not job["backend"].startswith("qpu."):
        raise ValueError("No raw histogram artifact; probabilities must not be relabeled as counts")
    return job, qpu_counts(job)


def qpu_counts(job):
    """A histogram artifact holding a QPU job's exact counts, from its probabilities.

    A QPU job's probabilities are its measured counts divided by its shots.
    For a debiased job they are the summed counts of its variants (a
    2,048-shot job ran as 32 variants of 64 shots). Multiplying back by the
    shots gives the counts exactly; a probability that is not a whole number
    of shots is refused, never rounded. A debiased job's record also lists
    each variant's own probabilities artifact, for later use such as sharpening.
    """
    descriptor = job["results"]["ionq.result.probabilities.json.v2"]
    artifact = api_request(f"/jobs/{job['id']}/artifacts/{descriptor['id']}")
    counts = {}
    for label, probability in artifact["probabilities"]["registers"]["output_all"].items():
        shots = probability * job["shots"]
        if abs(shots - round(shots)) > 1e-6:
            raise ValueError(f"Label {label}: probability {probability} is not a whole number "
                             f"of the job's {job['shots']} shots")
        if round(shots):
            counts[label] = int(round(shots))
    if sum(counts.values()) != job["shots"]:
        raise ValueError(f"{sum(counts.values())} shots rebuilt, {job['shots']} run")
    return dict(histogram=dict(registers=dict(output_all=counts)))


def output_histogram(artifact):
    """{bitstring: shots} over all qubits, from a v2 histogram artifact."""
    return artifact["histogram"]["registers"]["output_all"]


def probe_bit_order(histogram, qubits):
    """"q0-left" or "q0-right" from the probe histogram, or None if ambiguous.

    Only qubit 0 is flipped, so the side of the dominant label's single 1
    says which end of the string is qubit 0.
    """
    label = max(histogram, key=histogram.get)
    if len(label) != qubits:
        return None
    return {"1"+"0"*(qubits-1): "q0-left", "0"*(qubits-1)+"1": "q0-right"}.get(label)


def import_histogram(histogram, metadata, qubits, bit_order):
    """Counts array [k, second] from full-width bitstrings; the workspace is summed out.

    Invalid binary labels are kept (they are reported as noise). bit_order
    "q0-left" means the first character of each label is qubit 0.
    """
    if bit_order not in ("q0-left", "q0-right"):
        raise ValueError("Specify q0-left or q0-right")
    nx, ns = int(metadata["x_bits"]), int(metadata["second_bits"])
    counts = np.zeros((2**nx, 2**ns), dtype=np.int64)
    for label, value in histogram.items():
        if len(label) != qubits or set(label)-{"0", "1"}:
            raise ValueError("Expected full-width binary output_all labels")
        if int(value) != value or value < 0:
            raise ValueError("Expected raw integer shot counts")
        little = label[::-1] if bit_order == "q0-left" else label
        measured = int(little, 2) % (2**(nx+ns))
        counts[measured % (2**nx), measured//(2**nx)] += int(value)
    return counts
