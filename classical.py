"""Exact optimal classical baseline for the paper's permutation-hidden oracle.

A classical query (x, t) returns pi(x + alpha^t b mod p) for an unknown uniform
label permutation pi, so the answers carry information only through which
queries collide. Queries (x, t) and (x', t') collide exactly when
b = (x'-x)(alpha^t - alpha^t')^-1, which needs t != t'; each such pair covers one
secret. For a fixed secret the collision pattern is deterministic, so until the
first collision every algorithm sees the same answer and an adaptive algorithm
is a fixed query sequence (randomizing cannot beat the best sequence against a
uniform secret). After N queries covering a set D of secrets, a collision
identifies b in D, and otherwise b is uniform on the p-|D| secrets left, so the
optimal success probability is max_D min(|D|+1, p)/p. The maximum over N-point
query sets is found by exhaustive search, which is fast for the paper's p <= 7.
"""
from functools import lru_cache
import numpy as np
from circuits import group_parameters


def covered_secrets(p, q, points):
    """Secrets b for which some pair of the queried points collides."""
    _, powers = group_parameters(p, q)
    return {(x2-x1)*pow(int(powers[t1]-powers[t2]), -1, p) % p
            for i, (x1, t1) in enumerate(points) for x2, t2 in points[:i] if t1 != t2}


@lru_cache(None)
def optimal_queries(p, q, n):
    """A set of n queries covering the most secrets, found by branch and bound.

    Translating x and shifting t map covered sets to scaled copies of the same
    size, so the first query can be fixed at (0, 0). Covering p-1 secrets
    already guarantees recovery, so the search stops there.
    """
    _, powers = group_parameters(p, q)
    if n <= 1:
        return ((0, 0),)[:n]
    inverse = {d: pow(d, -1, p) for d in range(1, p)}
    domain = [(x, t) for t in range(q) for x in range(p)][1:]
    best = [set(), None]

    def search(chosen, covered, start):
        if len(covered) > len(best[0]) or best[1] is None:
            best[:] = [covered, chosen]
        remaining = n-len(chosen)
        if remaining == 0 or len(best[0]) >= p-1:
            return
        # Each added query pairs with every earlier one at most once.
        if len(covered)+remaining*len(chosen)+remaining*(remaining-1)//2 <= len(best[0]):
            return
        for index in range(start, len(domain)-remaining+1):
            x2, t2 = domain[index]
            new = {(x2-x1)*inverse[(powers[t1]-powers[t2]) % p] % p
                   for x1, t1 in chosen if t1 != t2}
            search(chosen+(domain[index],), covered | new, index+1)

    search(((0, 0),), set(), 0)
    # Pad with unused points if full coverage was reached early.
    chosen = best[1]
    return chosen+tuple(point for point in domain if point not in chosen)[:n-len(chosen)]


def optimal_curve(p, q, query_counts):
    """Return the optimal exact-secret recovery probability at each query budget."""
    group_parameters(p, q)
    counts = np.asarray(query_counts)
    if counts.ndim != 1 or np.any(counts < 0) or np.any(counts != counts.astype(int)):
        raise ValueError("Query counts must be a one-dimensional array of nonnegative integers")
    recovery = []
    for count in counts.astype(int):
        covered = len(covered_secrets(p, q, optimal_queries(p, q, min(count, p*q))))
        recovery.append(min(covered+1, p)/p)
    return np.array(recovery, dtype=float)


def guarantee(p, q):
    """Smallest budget at which the optimal algorithm always recovers b."""
    n = 0
    while len(covered_secrets(p, q, optimal_queries(p, q, n))) < p-1:
        n += 1
    return n
