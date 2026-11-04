"""Check the optimal classical curve against explicit oracle queries and an
exhaustive search over adaptive strategies."""
from functools import lru_cache
from itertools import combinations
import unittest
import numpy as np
from classical import covered_secrets, guarantee, optimal_curve, optimal_queries
from circuits import group_parameters


def adaptive_optimum(p, q, budget):
    """Best recovery over all adaptive deterministic strategies, by brute force.

    Only the collision pattern is informative, so a state is the query history
    and the secrets consistent with it; each query splits them by which earlier
    query (if any) it collides with.
    """
    _, powers = group_parameters(p, q)
    points = [(x, t) for x in range(p) for t in range(q)]

    @lru_cache(None)
    def value(history, secrets, left):
        best = 1/len(secrets)
        if left == 0 or len(secrets) == 1:
            return best
        for x, t in points:
            outcomes = {}
            for b in secrets:
                hit = next((i for i, (x1, t1) in enumerate(history)
                            if (x1+powers[t1]*b-x-powers[t]*b) % p == 0), None)
                outcomes.setdefault(hit, []).append(b)
            best = max(best, sum(len(part)*value(history+((x, t),), frozenset(part), left-1)
                                 for part in outcomes.values())/len(secrets))
        return best

    return value((), frozenset(range(p)), budget)


class ClassicalTests(unittest.TestCase):
    def test_matches_adaptive_brute_force(self):
        for p, q, budget in [(3, 2, 3), (5, 2, 4), (5, 4, 3)]:
            for n in range(budget+1):
                self.assertAlmostEqual(optimal_curve(p, q, [n])[0], adaptive_optimum(p, q, n))

    def test_maximizes_coverage_over_all_query_sets(self):
        for p, q in [(5, 2), (5, 4), (7, 2), (7, 3), (7, 6)]:
            points = [(x, t) for x in range(p) for t in range(q)]
            for n in range(guarantee(p, q)+1):
                best = max((len(covered_secrets(p, q, list(chosen)))
                            for chosen in combinations(points, n)), default=0)
                found = len(covered_secrets(p, q, list(optimal_queries(p, q, n))))
                self.assertEqual(min(found, p-1), min(best, p-1))

    def test_queries_recover_every_secret(self):
        rng = np.random.default_rng(93)
        for p, q in [(5, 2), (7, 2), (7, 6), (13, 4), (13, 12)]:
            _, powers = group_parameters(p, q)
            n = guarantee(p, q)
            points = optimal_queries(p, q, n)
            self.assertEqual(len(points), len(set(points)))
            unresolved = 0
            for b in range(p):
                permutation = rng.permutation(p)
                seen = {}
                for x, t in points:
                    label = permutation[(x+int(powers[t])*b) % p]
                    if label in seen:
                        xp, tp = seen[label]
                        denominator = (int(powers[t])-int(powers[tp])) % p
                        self.assertEqual((xp-x)*pow(denominator, -1, p) % p, b)
                        break
                    seen[label] = (x, t)
                else:
                    unresolved += 1
            # At most one secret is left, and it is then known by elimination.
            self.assertLessEqual(unresolved, 1)
            before, at = optimal_curve(p, q, [n-1, n])
            self.assertLess(before, 1)
            self.assertEqual(at, 1)

    def test_paper_values(self):
        np.testing.assert_allclose(optimal_curve(7, 2, range(7)), np.array([1, 1, 2, 3, 5, 7, 7])/7)
        np.testing.assert_allclose(optimal_curve(7, 6, range(6)), np.array([1, 1, 2, 4, 7, 7])/7)

    def test_endpoints_and_large_budgets(self):
        np.testing.assert_allclose(optimal_curve(13, 4, [0, 1, 52, 256]), [1/13, 1/13, 1, 1])
        self.assertEqual(len(optimal_curve(13, 4, [])), 0)
        for invalid in [[-1], [1.5], [[1, 2]]]:
            with self.assertRaises(ValueError):
                optimal_curve(13, 4, invalid)


if __name__ == "__main__":
    unittest.main()
