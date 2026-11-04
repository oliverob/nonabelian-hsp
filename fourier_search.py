"""Offline search for the short Fourier circuits stored in fourier_circuits.json.

    python fourier_search.py 3 5 6 7

Not part of the pipeline: circuits.py only reads the stored gate lists, and
tests/test_circuits.py checks them. Needs jax (pip install jax), which the
pipeline itself does not.

Every Fourier transform in the paper acts on labels 0,...,size-1 of an
n-qubit register and is followed directly by a measurement. So a circuit V
only has to satisfy V|j> = sum_k e^{i phi_k} (F_size)_{kj} |k> for j < size:
the unused binary states and a phase on each outcome are free. The ansatz is
a layer of arbitrary one-qubit gates U, then m blocks of RZZ(theta) on a qubit
pair followed by U on both qubits. The pairs follow a repeated round-robin
schedule (pair_sequence): Forte is all-to-all, so each sweep couples every
pair once, and consecutive pairs within a round are disjoint, so their gates
can run in parallel. m still counts ZZ gates, which dominate the error on a
trapped-ion device; zz_depth records the resulting parallel ZZ depth. With 2
or 3 qubits (F3-F8) no two pairs are disjoint, so the depth equals m. For m = 1, 2, ... BFGS from random starts minimizes
1 - (sum_k |<F_k, V_k>| / size)^2 over the populated rows and columns. The
first solution is polished by minimum-norm Gauss-Newton steps on the
residual V - diag(e^{i phi}) F, so the stored angles are exact to machine
precision.
"""
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import sys

import numpy as np

OUTPUT = Path(__file__).with_name("fourier_circuits.json")
RESTARTS = 48
SEED = 20260923


def bits(size):
    return max(1, (size - 1).bit_length())


def dft(size):
    labels = np.arange(size)
    return np.exp(2j * np.pi * np.outer(labels, labels) / size) / np.sqrt(size)


def rounds(n):
    """Round-robin (circle method) schedule: n-1 rounds of disjoint pairs for
    even n, n rounds of (n-1)//2 pairs for odd n; every pair appears once."""
    players = list(range(n)) + [None] * (n % 2)
    half = len(players) // 2
    schedule = []
    for _ in range(len(players) - 1):
        schedule.append([tuple(sorted(pair)) for pair in zip(players[:half], players[:half-1:-1])
                         if None not in pair])
        players = [players[0], players[-1], *players[1:-1]]
    return schedule


def pair_sequence(n, m):
    """First m pairs of the repeated round-robin schedule. Pairs within a round
    are disjoint, so their ZZ gates (and the U gates after them) can run in
    parallel, and each full sweep entangles every pair once (all-to-all)."""
    pairs = [pair for layer in rounds(n) for pair in layer]
    return [pairs[i % len(pairs)] for i in range(m)]


def zz_depth(pairs):
    """Parallel ZZ layers when each gate is scheduled as early as possible."""
    last = {}
    for i, j in pairs:
        last[i] = last[j] = max(last.get(i, 0), last.get(j, 0)) + 1
    return max(last.values(), default=0)


def model(n, pairs):
    """V(params) for the ansatz, little-endian like Qiskit; jax is imported lazily."""
    import jax.numpy as jnp

    def u(t):
        return jnp.array([[jnp.cos(t[0]/2), -jnp.exp(1j*t[2])*jnp.sin(t[0]/2)],
                          [jnp.exp(1j*t[1])*jnp.sin(t[0]/2), jnp.exp(1j*(t[1]+t[2]))*jnp.cos(t[0]/2)]])

    def on(qubit, gate):
        out = jnp.eye(1)
        for k in reversed(range(n)):
            out = jnp.kron(out, gate if k == qubit else jnp.eye(2))
        return out

    states = np.arange(2 ** n)
    zz = [jnp.array((1 - 2 * (states >> i & 1)) * (1 - 2 * (states >> j & 1))) for i, j in pairs]

    def V(x):
        matrix = jnp.eye(2 ** n, dtype=complex)
        for q in range(n):
            matrix = on(q, u(x[3*q:3*q+3])) @ matrix
        k = 3 * n
        for (i, j), diagonal in zip(pairs, zz):
            matrix = jnp.exp(-0.5j * x[k] * diagonal)[:, None] * matrix
            matrix = on(i, u(x[k+1:k+4])) @ on(j, u(x[k+4:k+7])) @ matrix
            k += 7
        return matrix
    return V, 3 * n + 7 * len(pairs)


def attempt(args):
    """One random start; returns (infidelity, params)."""
    size, m, seed = args
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from scipy.optimize import minimize
    n = bits(size)
    V, count = model(n, pair_sequence(n, m))
    target = jnp.array(dft(size))

    def cost(x):
        block = V(x)[:size, :size]
        overlap = jnp.abs(jnp.sum(target.conj() * block, axis=1)).sum() / size
        return 1 - overlap ** 2
    f, g = jax.jit(cost), jax.jit(jax.grad(cost))
    x0 = np.random.default_rng(seed).uniform(0, 2 * np.pi, count)
    result = minimize(lambda x: float(f(x)), x0, jac=lambda x: np.asarray(g(x)),
                      method="BFGS", options=dict(maxiter=4000, gtol=1e-12))
    return float(result.fun), result.x


def polish(size, m, x):
    """Gauss-Newton on V - diag(e^{i phi}) F; returns params and max error."""
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    n = bits(size)
    V, count = model(n, pair_sequence(n, m))
    target = jnp.array(dft(size))

    def residual(z):
        block = V(z[:count])[:size, :size]
        error = block - jnp.exp(1j * z[count:])[:, None] * target
        return jnp.concatenate([error.real.ravel(), error.imag.ravel()])
    block = np.asarray(V(x))[:size, :size]
    phases = np.angle(np.sum(dft(size).conj() * block, axis=1))
    r, jacobian = jax.jit(residual), jax.jit(jax.jacfwd(residual))
    z = np.concatenate([x, phases])
    best = (np.max(np.abs(np.asarray(r(z)))), z)
    for _ in range(20):
        # Truncated pseudo-inverse: the ansatz has redundant directions.
        z = z - np.linalg.lstsq(np.asarray(jacobian(z)), np.asarray(r(z)), rcond=1e-8)[0]
        best = min(best, (np.max(np.abs(np.asarray(r(z)))), z), key=lambda b: b[0])
    return best[1][:count], float(best[0])


def gate_list(size, m, x):
    """[["u", qubit, theta, phi, lam] | ["rzz", i, j, theta], ...] in circuit order."""
    n = bits(size)
    gates = [["u", q, *map(float, x[3*q:3*q+3])] for q in range(n)]
    k = 3 * n
    for i, j in pair_sequence(n, m):
        gates += [["rzz", i, j, float(x[k])], ["u", i, *map(float, x[k+1:k+4])],
                  ["u", j, *map(float, x[k+4:k+7])]]
        k += 7
    return gates


def search(size, pool):
    for m in range(1, 4 ** bits(size)):
        seeds = [SEED + 1000 * m + r for r in range(RESTARTS)]
        results = list(pool.map(attempt, [(size, m, s) for s in seeds]))
        best, x = min(results, key=lambda r: r[0])
        print(f"F{size}: m={m} best infidelity {best:.1e}", flush=True)
        if best < 1e-10:
            x, error = polish(size, m, x)
            if error < 1e-12:
                return dict(qubits=bits(size), zz=m, zz_depth=zz_depth(pair_sequence(bits(size), m)),
                            restarts=RESTARTS, max_error=error,
                            note=f"no solution found with {m - 1} ZZ in {RESTARTS} starts each",
                            gates=gate_list(size, m, x))
    raise RuntimeError(f"no circuit found for F{size}")


def main():
    sizes = [int(arg) for arg in sys.argv[1:]] or [3, 5, 6, 7]
    stored = json.loads(OUTPUT.read_text()) if OUTPUT.exists() else {}
    with ProcessPoolExecutor(mp_context=multiprocessing.get_context("spawn")) as pool:
        for size in sizes:
            stored[str(size)] = search(size, pool)
            OUTPUT.write_text(json.dumps(stored, indent=1) + "\n")
            print(f"F{size}: {stored[str(size)]['zz']} ZZ, max error {stored[str(size)]['max_error']:.1e}")


if __name__ == "__main__":
    main()
