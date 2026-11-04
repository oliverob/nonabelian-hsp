"""The experiment plan: every circuit the figures use, and the fixed run settings.

This is the one place that says *what* is run. `circuits()` lists the
(method, p, q, b) circuits that the six figures read, and `zne_circuits()` the
subset whose ZZ-folded copies the zero-noise extrapolated Figures 2-4 read.
`jobs()` is everything sent to the emulator or to hardware.

`jobs()` reproduces the 75 hardware jobs of the paper: the circuits the
figures read, the 13 folded copies, and the 11 NOT_PLOTTED circuits that were
run but appear in no figure.

Methods (see circuits.py): "adapted" (field-encoded second register, F_p on
both registers), "pairs" (T then H on adjacent-pair labels), "h_only" (H
only; a control that cannot tell b from -b), "forgetful" (F_q on the second
register) and "weak" (Fourier label k only). Figure 1 does not use "weak",
since k mixes the irrep with its row: it keeps only the irrep of each
forgetful outcome instead.

To run your own experiment, change the figure parameters below; `circuits()`
and `jobs()` follow from them.
"""
from circuits import group_parameters

SEED = 20260923          # Base seed for every random number in the project.
SHOTS = 2048             # Shots per circuit on the emulator and on hardware.
ZNE_FOLD = 3             # ZZ noise-scale factor of the folded ZNE copies.
IDEAL_SHOTS = 480        # Shots per circuit in the ideal Aer reference run.
TRIALS = 20000           # Monte Carlo trials per learning curve.
BOOTSTRAPS = 1000        # Resamples for ZNE and spectrum intervals.
# Measured runs: learning curves recomputed on this many multinomial resamples of
# every secret's histogram, so intervals include the finite-shot error of the data.
LEARNING_BOOTSTRAPS = 1000
EXACT_MAX = 4            # Adapted mode-decoder recovery is computed exactly up to this budget.

# Largest total query count in Figure 6, for the ideal reference and for measured runs.
# The ideal reference needs fewer queries than the noisy runs to saturate.
QUERY_LIMITS = dict(ideal=1024, sampled=2048)

# Figure parameters. Every prime is <= 7 so every circuit stays under 100 ZZ.
WEAK = (7, [2, 6])                 # Figure 1: p and the q values, every b (forgetful circuits).
ADAPTED = (7, 6, [0, 1, 5])        # Figure 2.
Q_SWEEP = (7, [2, 3, 6])           # Figure 3 (a): p and the q values, b = 1.
PRIMES = [3, 5, 7]                 # Figure 3 (b): the primes, q = 2, b = 1.
FFT = (7, 2, [1, 3, 6])            # Figure 4: pairs (top row) and H only (bottom row).
FORGETFUL = (7, 3)                 # Figure 5.
LEARNING = (7, 2)                  # Figure 6 (top row).
AFFINE = (7, 6)                    # Figure 6 (bottom row): the full affine case q = p-1.

# Run on hardware with the rest, but shown in no figure: adjacent pairs at q = 2
# for every secret at p = 3 and 5, and the k-only circuit at p = 7, q = 2.
NOT_PLOTTED = sorted({("pairs", p, 2, b) for p in (3, 5) for b in range(p)}
                     | {("weak", 7, 2, b) for b in (0, 1, 5)})


def forgetful_secrets(p, q):
    """b = 1, b = alpha (same orbit, identical law) and one b outside the orbit."""
    alpha, powers = group_parameters(p, q)
    outsider = next(b for b in range(1, p) if b not in powers)
    return [1, alpha, outsider]


def zne_circuits():
    """The 13 circuits behind the zero-noise extrapolated Figures 2-4."""
    p, q, secrets = ADAPTED
    jobs = {("adapted", p, q, b) for b in secrets}
    sweep_p, sweep_q = Q_SWEEP
    jobs |= {("adapted", sweep_p, q, 1) for q in sweep_q}
    jobs |= {("adapted", p, 2, 1) for p in PRIMES}
    p, q, secrets = FFT
    jobs |= {(method, p, q, b) for method in ("pairs", "h_only") for b in secrets}
    return sorted(jobs)


def circuits():
    """Every circuit read by Figures 1-6."""
    jobs = set(zne_circuits())
    p, sizes = WEAK
    jobs |= {("forgetful", p, q, b) for q in sizes for b in range(p)}    # Figure 1.
    p, q = FORGETFUL
    jobs |= {("forgetful", p, q, b) for b in forgetful_secrets(p, q)}   # Figure 5.
    for p, q in (LEARNING, AFFINE):                                      # Figure 6.
        jobs |= {(method, p, q, b) for method in ("adapted", "pairs", "forgetful")
                 for b in range(p)}
    return sorted(jobs)


def jobs():
    """Every circuit job as (method, p, q, b, fold): each circuit of circuits() and
    NOT_PLOTTED at fold 1, and a fold-ZNE_FOLD copy of each zne_circuits() circuit.
    Sorted so that a circuit's copies are sent one after the other, which keeps
    drift between them small."""
    zne = set(zne_circuits())
    return [(*circuit, fold) for circuit in sorted(set(circuits()) | set(NOT_PLOTTED))
            for fold in [1] + [ZNE_FOLD]*(circuit in zne)]
