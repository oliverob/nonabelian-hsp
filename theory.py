"""The closed-form measured law P(k, second) of every circuit, independent of its synthesis.

Kept apart from circuits.py so that changing it does not change
circuits.fingerprint(), which stamps prepared job files: the law is used
only after compilation, to check the Aer simulations (simulation.py), by
the maximum-likelihood decoders (decoding.py) and in the analysis.
"""
import numpy as np

from circuits import bits, group_parameters


def theory(p, q, b, method):
    """Independent analytic joint P(k, second), including k=0 and padding."""
    alpha, powers = group_parameters(p, q)
    width = 2 ** (bits(p) if method == "adapted" else bits(q))
    probability = np.zeros((2 ** bits(p), width))
    k = np.arange(p)
    if method == "adapted":
        for l in range(p):
            amplitudes = np.exp(2j * np.pi * np.outer(l - k * b, powers) / p).sum(axis=1)
            probability[:p, l] = abs(amplitudes) ** 2 / (p * p * q)
    elif method in ("pairs", "h_only"):
        for j in range(q // 2):
            m = ((1 - alpha) * int(powers[2 * j]) * k) % p
            phase = 2 * np.pi * m * b / p
            signal = np.cos(phase + (np.pi / 4 if method == "pairs" else 0))
            probability[:p, 2 * j] = (1 + signal) / (p * q)
            probability[:p, 2 * j + 1] = (1 - signal) / (p * q)
        if q % 2:
            # Odd q: label q-1 is paired with the empty padding label q, so H
            # splits it evenly and that pair carries no information about b.
            probability[:p, q - 1:q + 1] = 1 / (2 * p * q)
    elif method == "forgetful":
        for l in range(q):
            amplitude = np.exp(-2j * np.pi * np.outer(k * b, powers) / p
                               + 2j * np.pi * l * np.arange(q) / q).sum(axis=1)
            probability[:p, l] = abs(amplitude) ** 2 / (p * q * q)
    elif method == "weak":
        probability[:p, :q] = 1 / (p * q)
    else:
        raise ValueError(method)
    return probability
