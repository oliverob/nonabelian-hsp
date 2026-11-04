"""Ideal (noiseless) Aer simulation of the logical circuits, with an on-disk cache.

For each circuit this gives the exact marginal probabilities of the measured
registers and a finite shot histogram, both as arrays indexed [k, second].
Every simulation is checked against the closed-form law theory.theory().
Results are cached in runs/ideal/aer/; the cache key hashes this file, the
circuit sources (circuits.fingerprint) and theory.py, so editing any of them
triggers a fresh (fast, ~0.1 s/circuit) run.
"""
from pathlib import Path
import hashlib

import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator

from circuits import build_circuit, fingerprint
from theory import theory
from plan import IDEAL_SHOTS, SEED

CACHE = Path("runs/ideal/aer")
CACHE_TAG = hashlib.sha256(Path(__file__).read_bytes() + fingerprint().encode()
                           + (Path(__file__).parent/"theory.py").read_bytes()).hexdigest()


def simulate(circuit, shots=IDEAL_SHOTS, seed=SEED):
    """Exact marginal probabilities and a seeded shot histogram of `circuit`.

    The circuit must carry the metadata set by circuits.build_circuit (or
    forte.native_as_qiskit), which says how many qubits are measured.
    """
    metadata = circuit.metadata
    clean = circuit.remove_final_measurements(inplace=False)
    measured = metadata["measured_qubits"]
    clean.save_probabilities(list(range(measured)))
    clean.add_register(circuit.cregs[0])
    clean.measure(range(measured), range(measured))
    simulator = AerSimulator(method="statevector", max_parallel_threads=1)
    compiled = transpile(clean, simulator, optimization_level=0)
    result = simulator.run(compiled, shots=shots, seed_simulator=seed).result()
    if not result.success:
        raise RuntimeError(result.status)
    width_x = 2 ** metadata["x_bits"]
    width_second = 2 ** metadata["second_bits"]
    probabilities = np.asarray(result.data(0)["probabilities"]).reshape(width_second, width_x).T
    counts = np.zeros(width_x * width_second, dtype=np.int64)
    for label, count in result.get_counts().items():
        counts[int(label.replace(" ", ""), 2)] = count
    return probabilities, counts.reshape(width_second, width_x).T


def cache_path(p, q, b, method):
    return CACHE / f"{method}_p{p}_q{q}_b{b}.npz"


def ideal_instance(p, q, b, method):
    """(probabilities, counts) of the ideal circuit, simulating it if not cached."""
    path = cache_path(p, q, b, method)
    if path.exists():
        with np.load(path) as stored:
            if int(stored["shots"]) == IDEAL_SHOTS and str(stored["cache_tag"]) == CACHE_TAG:
                return stored["probabilities"], stored["counts"]
    circuit = build_circuit(p, q, b, method)
    seed = SEED + p * 10000 + q * 100 + b
    probabilities, counts = simulate(circuit, seed=seed)
    error = float(np.max(abs(probabilities - theory(p, q, b, method))))
    if error > 1e-10:
        raise AssertionError(f"Circuit/theory mismatch {circuit.name}: {error}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, probabilities=probabilities, counts=counts,
                        p=p, q=q, b=b, method=method, seed=seed, shots=IDEAL_SHOTS,
                        theory_max_error=error, cache_tag=CACHE_TAG)
    print(f"Aer {circuit.name}: max error {error:.2g}", flush=True)
    return probabilities, counts


def theory_error(p, q, b, method):
    """Largest |Aer - closed form| probability difference recorded for a circuit."""
    ideal_instance(p, q, b, method)
    with np.load(cache_path(p, q, b, method)) as stored:
        return float(stored["theory_max_error"])
