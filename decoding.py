"""Decoders: turn measured labels into a score for every candidate secret b.

Decoders see only samples and the public group parameters (p, q, alpha),
never the secret. A sample is the integer k + 2**bits(p) * second, where k is
the measured Fourier label of x and `second` the label of the second
register. k = 0 carries no information about b and binary labels outside the
valid ranges are noise; both are discarded (they still count as queries).

All functions work on whole arrays, so analysis.learning_curve can score
thousands of Monte Carlo trials at once; the highest score is the estimate.
"""
import numpy as np

from circuits import bits, group_parameters
from theory import theory


def unpack(samples, p):
    """Split samples into (k, second)."""
    samples = np.asarray(samples, dtype=np.int64)
    return samples % 2 ** bits(p), samples // 2 ** bits(p)


def valid(k, second, p, support):
    """Informative shots: nonzero Fourier label and in-range second label."""
    return (k > 0) & (k < p) & (second < support)


def adapted_candidates(k, l, p):
    """Adapted method: each valid shot (k, l) votes for the candidate l/k mod p.

    The correct b receives probability q/p of the votes, every other
    candidate (p-q)/(p(p-1)), so the mode of the votes is the estimate.
    """
    inverse = np.array([0] + [pow(value, -1, p) for value in range(1, p)])
    return (np.asarray(l) * inverse[np.asarray(k)]) % p


def pair_label(k, t, p, q):
    """Adjacent-pair methods: the frequency m = (1-alpha) alpha^(t - t%2) k mod p.

    The pair (t, t+1) with t even differs by a factor alpha, so the measured
    bit r = t%2 has E[(-1)^r | m] = cos(2 pi m b / p + pi/4) after T then H,
    and cos(2 pi m b / p) after H alone.
    """
    alpha, powers = group_parameters(p, q)
    return ((1 - alpha) * powers[t - t % 2] * k) % p


def pairs_scores(signed):
    """Correlation of the signed histogram with cos(2 pi m b / p + pi/4), every b.

    signed[..., m] = sum over valid shots at frequency m of (-1)^r. With
    numpy's negative-exponent DFT the correlation is Re(exp(-i pi/4) FFT).
    Every candidate, b = 0 included, has the same template norm, so the plain
    argmax is least squares and stays correct under depolarizing noise.
    """
    return np.real(np.exp(-1j * np.pi / 4) * np.fft.fft(signed, axis=-1))


def h_only_scores(signed, visits):
    """Negative squared residual against cos(2 pi m b / p), up to a constant.

    visits[..., m] counts valid shots at frequency m. The b = 0 template has
    twice the norm of the others, so the cos**2 term (a second FFT, of the
    visits at frequency 2b) is kept. b and -b always tie: the H-only
    measurement cannot distinguish them.
    """
    p = signed.shape[-1]
    second_harmonic = np.fft.fft(visits, axis=-1).real[..., (2 * np.arange(p)) % p]
    return 2 * np.fft.fft(signed, axis=-1).real - visits.sum(axis=-1)[..., None] / 2 - second_harmonic / 2


def log_likelihood_table(p, q, method):
    """log P(sample | b) for every b, for the maximum-likelihood (forgetful) decoder.

    Rows are candidates b, columns the flattened samples k + 2**bits(p) * second.
    """
    return np.array([np.log(np.maximum(theory(p, q, b, method).T.ravel(), 1e-15))
                     for b in range(p)])
