"""Decoders, learning-curve statistics and zero-noise extrapolation."""
import unittest

import numpy as np

from unittest import mock

import analysis
from analysis import (adapted_exact, extrapolate, hist_candidates, irrep_bins,
                      learning_curve, threshold_crossings, weak_law)
from circuits import group_parameters
from theory import theory
from decoding import (adapted_candidates, h_only_scores, pair_label, pairs_scores,
                      unpack, valid)


def decode(samples, p, q, method):
    """Score one batch of samples the way analysis.learning_curve does, return argmax."""
    k, second = unpack(samples, p)
    keep = valid(k, second, p, p if method == "adapted" else q)
    k, second = k[keep], second[keep]
    if method == "adapted":
        return int(np.argmax(np.bincount(adapted_candidates(k, second, p), minlength=p)))
    m = pair_label(k, second, p, q)
    signed = np.bincount(m, weights=1-2*(second % 2), minlength=p)
    return int(np.argmax(pairs_scores(signed)))


class WeakSamplingTests(unittest.TestCase):
    def test_forgetful_irreps_follow_the_weak_law_for_every_secret(self):
        for p in (3, 5, 7, 11, 13):
            for q in (q for q in range(2, p) if (p-1) % q == 0):
                bins, names = irrep_bins(p, q)
                # Each q-dimensional irrep holds q of the nonzero labels k.
                np.testing.assert_array_equal(np.bincount(bins[1:p, 0])[q:], q)
                for b in range(p):
                    law = np.bincount(bins.ravel(), theory(p, q, b, "forgetful").ravel(), len(names))
                    np.testing.assert_allclose(law, weak_law(p, q), atol=1e-12)


class DecoderTests(unittest.TestCase):
    def test_pair_scores_equal_direct_correlations(self):
        p, q = 7, 6
        rng = np.random.default_rng(3)
        k = rng.integers(1, p, 200); t = rng.integers(q, size=200)
        m = pair_label(k, t, p, q)
        alpha, powers = group_parameters(p, q)
        np.testing.assert_array_equal(m, ((1-alpha)*powers[t-t % 2]*k) % p)
        z = 1-2*(t % 2)
        phase = 2*np.pi*np.outer(np.arange(p), m)/p
        signed = np.bincount(m, weights=z, minlength=p)
        visits = np.bincount(m, minlength=p).astype(float)
        # Pairs: correlation with cos(phase + pi/4). H only: least-squares residual.
        np.testing.assert_allclose(pairs_scores(signed), (z*np.cos(phase+np.pi/4)).sum(axis=1), atol=1e-10)
        model = np.cos(phase)
        np.testing.assert_allclose(h_only_scores(signed, visits), (2*z*model-model**2).sum(axis=1), atol=1e-10)

    def test_pairs_decoder_recovers_zero_and_sign(self):
        rng = np.random.default_rng(4)
        for b in range(7):
            probability = theory(7, 2, b, "pairs").T.ravel()
            samples = rng.choice(len(probability), 10000, p=probability)
            self.assertEqual(decode(samples, 7, 2, "pairs"), b)

    def test_pairs_decoder_survives_depolarizing_noise(self):
        # b=0 is a real peak under T then H, so no noise-dependent threshold is needed.
        rng = np.random.default_rng(5)
        for b in range(7):
            ideal = theory(7, 2, b, "pairs")
            probability = (.3*ideal+.7/ideal.size).T.ravel()
            samples = rng.choice(len(probability), 8000, p=probability)
            self.assertEqual(decode(samples, 7, 2, "pairs"), b)

    def test_adapted_mode_corrects_each_k_and_discards_invalids(self):
        # Raw l values 4, 1, 3 all vote for b=2 after multiplying by 1/k.
        samples = [2+8*4, 3+8*1, 4+8*3, 0+8*2, 6+8*2, 1+8*7]
        self.assertEqual(decode(samples, 5, 2, "adapted"), 2)
        k, l = unpack(samples, 5)
        np.testing.assert_array_equal(valid(k, l, 5, 5), [1, 1, 1, 0, 0, 0])

    def test_adapted_candidate_marginal_has_unique_mode(self):
        # Marginalize the independently checked circuit law over measured k.
        for p, q in [(5, 2), (7, 3), (7, 6)]:
            for b in range(p):
                probability = hist_candidates(theory(p, q, b, "adapted"), p)
                probability /= probability.sum()
                expected = np.full(p, (p-q)/(p*(p-1)))
                expected[b] = q/p
                np.testing.assert_allclose(probability, expected, atol=1e-12)


class StatisticsTests(unittest.TestCase):
    def test_threshold_interpolation_and_censoring(self):
        rates = [[.1, .8, 1], [.95, .8, 1], [.1, .8, .89], [.1, .9, .85]]
        np.testing.assert_allclose(threshold_crossings([0, 1, 2], rates), [1.5, 0, np.inf, 1])

    def test_adapted_exact_matches_closed_form_and_sampling(self):
        p, q = 5, 4
        ideal = [theory(p, q, b, "adapted") for b in range(p)]
        # One ideal query: the shot names b with probability (p-1)q/p^2, and k = 0 guesses.
        np.testing.assert_allclose(adapted_exact(ideal, p, [0, 1]), [1/p, ((p-1)*q+1)/p**2], atol=1e-12)
        noisy = [.6*law+.4/law.size for law in ideal]
        exact = adapted_exact(noisy, p, [1, 2, 3, 4])
        trials = 100000
        with mock.patch.object(analysis.plan, "EXACT_MAX", -1):  # Force sampling at every budget.
            sampled, _, _ = learning_curve(p, q, "adapted", [1, 2, 3, 4], trials=trials, seed=3,
                                           save=False, law=lambda b: noisy[b])
        np.testing.assert_allclose(sampled, exact, atol=4*np.sqrt(.25/trials))

    def test_shorter_grid_reproduces_the_same_trials(self):
        p, q = 7, 2
        law = lambda b: .7*theory(p, q, b, "pairs")+.3/theory(p, q, b, "pairs").size
        sizes = np.array([0, 1, 2, 3, 5, 8, 13, 21, 34])
        full, _, _ = learning_curve(p, q, "pairs", sizes, trials=500, seed=5, save=False, law=law)
        short, _, _ = learning_curve(p, q, "pairs", sizes[:5], trials=500, seed=5, save=False, law=law)
        np.testing.assert_array_equal(full[:5], short)

    def test_zne_extrapolation_is_exact_for_depolarizing_noise(self):
        ideal = theory(7, 2, 3, "pairs")
        uniform = np.full(ideal.shape, 1/ideal.size)
        noisy = lambda f: f*ideal+(1-f)*uniform
        # Clean fraction .6 per unit noise scale; fold 3 has .6**3.
        np.testing.assert_allclose(extrapolate(noisy(.6), noisy(.6**3), 3), ideal, atol=1e-12)
        self.assertTrue(np.isnan(extrapolate(noisy(.6), uniform, 3)).all())


if __name__ == "__main__":
    unittest.main()
