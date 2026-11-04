"""Figures 1-6 of the paper, from ideal Aer or from a run's histograms.

Two kinds of analysis, both driven by pipeline.py:

    analyse_ideal()     Noiseless Aer reference: runs/ideal/{analysis,figures}.
    analyse_run(run)    A run's histograms in runs/<target>/results, into
                        runs/<target>/figures. Figures 1, 5 and 6 use the raw
                        (fold 1) histograms. Figures 2-4 show zero-noise
                        extrapolated estimates, from the fold-1 and ZZ-folded
                        (fold 3) histograms, with the raw data overlaid.

    fig1_weak             weak Fourier sampling: the irrep law is the same for every secret
    fig2_adapted          adapted-basis candidates l k^-1 peak at the secret
    fig3_success_scaling  adapted-basis success against q and against p
    fig4_fft              adjacent-pair spectra, with T then H and with H only
    fig5_forgetful        forgetful abelian sampling: joint P(k, l)
    fig6_learning         recovery of the secret against total oracle queries

Figure 6's learning curves draw N shots per trial from a circuit's *finite*
measured histogram (so on hardware they are an empirical resampling), decode
them with decoding.py, and record how often the exact secret is recovered,
averaged over a uniform random secret. Each shot counts as one oracle query.
The exact optimal classical baseline (classical.py) is drawn for comparison.
For measured runs, histogram_bootstrap repeats every curve on multinomial
resamples of the histograms, so intervals include their finite-shot error;
the ideal run uses the exact Aer law, so its only error is Monte Carlo.

Each analysis also writes summary.json (the numbers behind the figures) and
the learning-curve arrays learning_*.npz into analysis/raw and analysis/zne.
This module keeps the current configuration in module-level variables set by
_configure().
"""
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import itertools
import json
import math
import multiprocessing
import os
from pathlib import Path
import platform
import zlib

os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import qiskit
import qiskit_aer

from circuits import bits, group_parameters
from theory import theory
from classical import guarantee, optimal_curve
from decoding import (adapted_candidates, h_only_scores, log_likelihood_table,
                      pair_label, pairs_scores, valid)
from forte import SOURCES
import plan
import style
from style import COLORS, COLUMN, PAGE, panel_headings, panel_labels, row_label, setting
from plan import SEED, TRIALS
from simulation import ideal_instance, theory_error

LEARNING_XMAX = 512  # Figure 6 x-axis end; every curve has saturated by then.
LEARNING_COLORS = {False: COLORS[0], True: COLORS[1]}  # Figure 6: native decoder, maximum likelihood.
PLOT_LABELS = dict(emulator="IonQ Forte noise-model simulator", hardware="IonQ Forte")
ZNE_FIGURES = [2, 3, 4]  # Only the distribution figures are mitigated.

# Current configuration, set by _configure().
MODE = "ideal"       # "ideal", "emulator" or "hardware".
RESULTS = None       # Directory of fetched histograms (sampled modes only).
ZNE = False          # True: zero-noise extrapolated figures 2-4.
FIGURES = DATA = None
OUTPUT = SHOTS = None  # The run directory and shots per circuit, passed on to bootstrap workers.
SHOT_NOTE = ""       # e.g. ", 2,048 shots/circuit" in the summary's source.
LEARNING_MAX = plan.QUERY_LIMITS["ideal"]


# ------------------------------------------------------------ loading results

def load_counts(p, q, b, method, fold=1):
    """Measured histogram [k, second] of one circuit at one ZZ fold."""
    suffix = "" if fold == 1 else f"_fold{fold}"
    path = RESULTS / f"{method}_p{p}_q{q}_b{b}{suffix}.npz"
    with np.load(path) as saved:
        if str(saved["source"]) != SOURCES[MODE]:
            raise ValueError(f"Not a {MODE} result: {path}")
        if int(saved["zne_fold"]) != fold:
            raise ValueError(f"{path} records fold {int(saved['zne_fold'])}, expected {fold}")
        return saved["counts"]


def extrapolate(base, folded, fold):
    """Exponential zero-noise estimate from fold-1 and fold-`fold` histograms.

    Assumes each unit of noise scale shrinks the deviation from the uniform
    distribution by the same factor, as global depolarizing noise does; that
    factor is fitted per circuit across all labels, which is exact for
    depolarizing noise. Returns a quasi-probability array: it sums to one but
    may have small negative entries. NaN if the folded run has no signal left.
    """
    one, scaled = base/base.sum(), folded/folded.sum()
    uniform = 1/one.size
    deviation, scaled_deviation = one-uniform, scaled-uniform
    shrink = (scaled_deviation*deviation).sum()/(deviation*deviation).sum()
    if shrink <= 0:
        return np.full(one.shape, np.nan)
    return uniform + deviation/shrink**(1/(fold-1))


@lru_cache(None)
def zne_replicates(p, q, b, method):
    """Mitigated estimates from multinomial resamples of both histograms."""
    base, folded = load_counts(p, q, b, method), load_counts(p, q, b, method, plan.ZNE_FOLD)
    rng = np.random.default_rng([SEED, p, q, b, zlib.crc32(method.encode())])
    draws = []
    for _ in range(plan.BOOTSTRAPS):
        one = rng.multinomial(base.sum(), (base/base.sum()).ravel()).reshape(base.shape)
        many = rng.multinomial(folded.sum(), (folded/folded.sum()).ravel()).reshape(folded.shape)
        draws.append(extrapolate(one, many, plan.ZNE_FOLD))
    return np.array(draws)


def run_instance(p, q, b, method, raw=False):
    """(probabilities, counts) of one circuit under the current configuration.

    Ideal: exact Aer probabilities and the Aer shot histogram. Sampled: the
    measured frequencies and counts. Under ZNE both are the mitigated
    quasi-distribution (counts scaled to the fold-1 shot total), unless
    raw=True asks for the unmitigated fold-1 histogram.
    """
    if MODE == "ideal":
        return ideal_instance(p, q, b, method)
    counts = load_counts(p, q, b, method)
    if ZNE and not raw:
        mitigated = extrapolate(counts, load_counts(p, q, b, method, plan.ZNE_FOLD), plan.ZNE_FOLD)
        return mitigated, mitigated*counts.sum()
    return counts/counts.sum(), counts


def zne_noise_check():
    """Secret-free clean-fraction estimate at each fold, for the ZNE circuits.

    If a fraction c of shots follows the ideal law and the rest is uniform,
    the share of labels the ideal circuit never produces determines c.
    Folding must lower it, or ZNE has nothing to extrapolate.
    """
    rows = []
    for method, p, q, b in plan.zne_circuits():
        base, folded = load_counts(p, q, b, method), load_counts(p, q, b, method, plan.ZNE_FOLD)
        impossible = theory(p, q, b, method) < 1e-12
        share = impossible.mean()
        clean = lambda c: float(1 - c[impossible].sum()/c.sum()/share) if share else None
        rows.append(dict(circuit=f"{method}_p{p}_q{q}_b{b}", clean_base=clean(base), clean_folded=clean(folded)))
    return [row for row in rows if row["clean_base"] is not None]


def interval(p, q, b, method, statistic, raw=False):
    """Estimate and 95% interval of numerator/denominator = statistic(counts).

    Shot counts use Wilson intervals. ZNE estimates are not shot counts, so
    they use percentile intervals over resamples of both histograms.
    """
    _, counts = run_instance(p, q, b, method, raw)
    numerator, denominator = statistic(counts)
    value = np.asarray(numerator)/denominator
    if not ZNE or raw:
        low, high = wilson(numerator, denominator)
        return value, low, high, denominator
    ratios = np.array([np.divide(*statistic(draw)) for draw in zne_replicates(p, q, b, method)])
    low, high = np.nanquantile(ratios, [.025, .975], axis=0)
    return value, low, high, denominator


def wilson(success, total):
    """95% Wilson score interval for a binomial proportion."""
    value = np.asarray(success) / total
    z = 1.959963984540054
    center = (value + z*z/(2*total)) / (1+z*z/total)
    radius = z*np.sqrt(value*(1-value)/total+z*z/(4*total*total))/(1+z*z/total)
    return center-radius, center+radius


# ------------------------------------------------------------ plotting helpers

def shot_label(raw=False):
    """Legend label for plotted sampled data; the caption names the source."""
    if MODE == "ideal":
        return "Aer shots"
    if raw and ZNE:
        return "Raw (fold 1)"
    return f"ZNE (folds 1, {plan.ZNE_FOLD})" if ZNE else "Measured"


def paired_bars(ax, x, value, low, high, color, raw):
    """Bars with 95% interval endpoints. Under ZNE, raw bars are hollow on the
    left of each label and ZNE bars filled on the right; otherwise one filled
    bar per label."""
    if not ZNE:
        bars = ax.bar(x, value, color=color, alpha=.85)
        ax.vlines(x, low, high, color="black", lw=.7)
        return bars
    width = .4
    x = np.asarray(x) + (-width/2 if raw else width/2)
    style = dict(facecolor="white", edgecolor=color, hatch="////", lw=.8) if raw else dict(color=color, alpha=.85)
    bars = ax.bar(x, value, width, **style)
    ax.vlines(x, low, high, color="0.45" if raw else "black", lw=.7)
    return bars


def comparison_handles(reference="Ideal peak $q/p$", marker="--"):
    """Neutral legend entries for raw-vs-ZNE bar figures."""
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    handles = [Patch(facecolor="white", edgecolor="0.3", hatch="////", label=shot_label(raw=True)),
               Patch(facecolor="0.45", label=shot_label()),
               Line2D([], [], color="black", ls="none" if marker == "_" else marker,
                      marker=marker if marker == "_" else None, ms=7, lw=1, label=reference)]
    return handles, [h.get_label() for h in handles]


def save(figure, name):
    """Write figure `name` as PDF and PNG. No title: the caption names the data."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    figure.savefig(FIGURES / f"{name}.png", bbox_inches="tight")
    plt.close(figure)


def hist_candidates(counts, p):
    """Adapted histogram [k, l] -> votes for each candidate l k^-1 mod p (k > 0)."""
    histogram = np.zeros(p)
    for k in range(1, p):
        for l in range(p):
            histogram[(l * pow(k, -1, p)) % p] += counts[k, l]
    return histogram


# -------------------------------------------------- Figures 1-4: distributions

def irrep_bins(p, q):
    """Weak-sampling bin of each forgetful outcome (k, l), and the bin names.

    k = 0 carries the one-dimensional irreps, and F_q on the second register
    gives sigma_l as l. For k != 0 the relabelling of Sec. 2 writes
    k = g^(c+1) alpha^j with alpha = g^((p-1)/q): k lies in the q-dimensional
    irrep rho_c, c = (log_g k - 1) mod (p-1)/q, and j is a row index that weak
    sampling discards. Labels outside the group form a last, invalid bin.
    """
    alpha, _ = group_parameters(p, q)
    r = (p-1)//q
    g = next(g for g in range(2, p)
             if pow(g, r, p) == alpha and len({pow(g, e, p) for e in range(p-1)}) == p-1)
    log = {pow(g, e, p): e for e in range(p-1)}
    bins = np.full((2**bits(p), 2**bits(q)), q+r)
    bins[0, :q] = range(q)
    for k in range(1, p):
        bins[k, :q] = q + (log[k]-1) % r
    names = ([rf"$\sigma_{{{l}}}$" for l in range(q)] + [rf"$\rho_{{{c}}}$" for c in range(r)]
             + ["invalid"])
    return bins, names


def weak_law(p, q):
    """Ideal weak-sampling law for every H_b: 1/p on sigma_0, q/p on each rho_c."""
    return np.r_[1/p, np.zeros(q-1), np.full((p-1)//q, q/p), 0]


def plot_weak(summary):
    """Figure 1: weak Fourier sampling; the irrep law is the same for every secret."""
    p, sizes = plan.WEAK
    fig, axes = plt.subplots(len(sizes), 1, figsize=(COLUMN, 1.45*len(sizes)+.3), sharey=True, layout="constrained")
    errors, invalid, summary["weak_spread"] = [], [], []
    for ax, q in zip(axes, sizes):
        bins, names = irrep_bins(p, q)
        histogram = lambda c: (np.bincount(bins.ravel(), c.ravel(), len(names)), c.sum())
        x, law, laws = np.arange(len(names)), weak_law(p, q), []
        for b, offset in enumerate(np.linspace(-.3, .3, p)):
            probabilities, _ = run_instance(p, q, b, "forgetful")
            laws.append(histogram(probabilities)[0])
            value, low, high, _ = interval(p, q, b, "forgetful", histogram)
            ax.errorbar(x+offset, value, yerr=[value-low, high-value], fmt="o", ms=2, lw=.6,
                        color=COLORS[0], label=rf"$b=0,\ldots,{p-1}$" if b == 0 else None)
        laws = np.array(laws)
        errors.append(abs(laws[:, :-1]-law[:-1]).max())
        invalid.append(laws[:, -1].max())
        summary["weak_spread"].append(dict(p=p, q=q, max_pairwise_tv=float(max(
            .5*abs(laws[a]-laws[c]).sum() for a in range(p) for c in range(a)))))
        ax.hlines(law, x-.4, x+.4, color="black", lw=1, label="Ideal, every $b$")
        for edge in x[:-1]+.5:
            ax.axvline(edge, color="0.9", lw=.6, zorder=0)
        ax.set(xticks=x, xticklabels=names, xlim=(-.6, len(names)-.4), ylabel="Probability")
        ax.tick_params(axis="x", which="both", top=False, bottom=False)
    axes[-1].set_xlabel("Irrep")
    axes[0].set_ylim(-.02, 1)
    panel_labels(axes)
    panel_headings(axes, [setting("q", q) for q in sizes])
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=2)
    save(fig, "fig1_weak")
    summary["weak_max_error"] = float(max(errors))
    summary["weak_max_invalid"] = float(max(invalid))


def plot_adapted(summary):
    """Figure 2: adapted candidates l k^-1 peak at the secret with height q/p."""
    p, q, secrets = plan.ADAPTED
    fig, axes = plt.subplots(1, 3, figsize=(COLUMN, 1.95 if ZNE else 1.7), sharey=True, layout="constrained")
    peaks = []
    candidates = lambda c: (lambda h: (h, h.sum()))(hist_candidates(c, p))
    for index, (ax, b) in enumerate(zip(axes, secrets)):
        value, low, high, total = interval(p, q, b, "adapted", candidates)
        if ZNE:
            raw_value, raw_low, raw_high, _ = interval(p, q, b, "adapted", candidates, raw=True)
            paired_bars(ax, range(p), raw_value, raw_low, raw_high, COLORS[index], raw=True)
        paired_bars(ax, range(p), value, low, high, COLORS[index], raw=False)
        ax.axhline(q/p, color="black", ls="--", lw=.8)
        ax.set_xticks(range(0, p, 3))
        peaks.append(dict(b=b, conditional_success=float(value[b]), retained=float(total)))
        if ZNE:
            peaks[-1]["raw_conditional_success"] = float(raw_value[b])
    axes[0].set_ylabel("Conditional probability")
    axes[1].set_xlabel(r"Candidate $l k^{-1}\ (\mathrm{mod}\ p)$")  # Centered: one label for all three.
    panel_labels(axes)
    panel_headings(axes, [setting("b", b) for b in secrets])
    if ZNE:
        fig.legend(*comparison_handles(), loc="outside lower center", ncol=3)
    save(fig, "fig2_adapted")
    summary["adapted_peaks"] = peaks


def plot_success(summary):
    """Figure 3: adapted one-shot success P(k>0, l k^-1 = b) against q and p."""
    fig, axes = plt.subplots(1, 2, figsize=(COLUMN, 2.2), layout="constrained")
    records = []
    primes = plan.PRIMES
    sweep_p, sweep_q = plan.Q_SWEEP
    direct = lambda p: (lambda c: (hist_candidates(c, p)[1], c.sum()))
    for ax, cases, varying in [(axes[0], [(sweep_p, q) for q in sweep_q], "q"),
                               (axes[1], [(p, 2) for p in primes], "p")]:
        xvalues, values, lower, upper = [], [], [], []
        for p, q in cases:
            value, low, high, _ = interval(p, q, 1, "adapted", direct(p))
            xvalues.append(q if varying == "q" else p)
            values.append(value); lower.append(value-low); upper.append(high-value)
            records.append(dict(p=p, q=q, success=float(value), theory=(p-1)*q/p**2))
        ax.errorbar(xvalues, values, yerr=[lower, upper], fmt="o", color=COLORS[0], label=f"{shot_label()}, $b=1$")
        ax.plot(xvalues, [(p-1)*q/p**2 for p, q in cases], color=COLORS[0], label=r"$(p-1)q/p^2$")
        ax.plot(xvalues, [1/p for p, q in cases], "--", color=COLORS[1], label="One-query random guess")
        ax.set(xlabel=f"${varying}$", xticks=xvalues, ylim=(0, None))
    axes[1].margins(x=.05)
    axes[0].set_ylabel("Success per circuit execution")
    panel_labels(axes)
    panel_headings(axes, [setting("p", sweep_p), setting("q", 2)])
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2)
    save(fig, "fig3_success_scaling")
    summary["single_shot_success"] = records


def spectrum(probabilities, p, q):
    """Fourier transform over frequencies m of the signed pair histogram.

    Component b is E[(-1)^r e^{-2 pi i m b/p}], so the pairs peak sits at the
    secret after a pi/4 rotation. Scaled by P(m) = 1/p; the empty m = 0 bin is zero.
    """
    alpha, powers = group_parameters(p, q)
    signed = np.zeros(p)
    for k in range(1, p):
        for t in range(q):
            m = ((1-alpha)*int(powers[t-t % 2])*k) % p
            signed[m] += (1-2*(t % 2))*probabilities[k, t]
    return np.fft.fft(p*signed)/p


def spectrum_interval(p, q, b, method, turn, raw=False):
    """95% percentile interval of the plotted spectrum component.

    Shots: multinomial resamples of the histogram. ZNE: the resamples of
    both histograms used by the other mitigated figures.
    """
    if ZNE and not raw:
        draws = zne_replicates(p, q, b, method)
    else:
        _, counts = run_instance(p, q, b, method, raw)
        rng = np.random.default_rng([SEED, p, q, b, zlib.crc32(method.encode())])
        total = counts.sum()
        draws = rng.multinomial(total, (counts/total).ravel(), size=plan.BOOTSTRAPS).reshape(-1, *counts.shape)/total
    values = np.array([(turn*spectrum(draw, p, q)).real for draw in draws])
    return np.nanquantile(values, [.025, .975], axis=0)


def plot_fft(summary):
    """Figure 4: pair spectra, top row with T then H (peak at b), bottom row H only (b, -b tie)."""
    p, q, secrets = plan.FFT
    fig, grid = plt.subplots(2, 3, figsize=(COLUMN, 3.1), sharex=True, sharey="row", layout="constrained")
    # Top row: T then H before measurement (HT); bottom row: H only.
    for axes, method, basis in [(grid[0], "pairs", "HT"), (grid[1], "h_only", "H")]:
        for ax, b, color in zip(axes, secrets, COLORS):
            probabilities, counts = run_instance(p, q, b, method)
            exact = spectrum(probabilities if MODE == "ideal" else theory(p, q, b, method), p, q)
            observed = spectrum(counts/counts.sum(), p, q)
            # Rotating by exp(-i pi/4) makes the pairs peak real, as in the decoder.
            turn = np.exp(-1j*np.pi/4) if method == "pairs" else 1
            y = (turn*observed).real
            reference = (turn*exact).real
            low, high = spectrum_interval(p, q, b, method, turn)
            if ZNE:
                _, raw_counts = run_instance(p, q, b, method, raw=True)
                raw_y = (turn*spectrum(raw_counts/raw_counts.sum(), p, q)).real
                raw_low, raw_high = spectrum_interval(p, q, b, method, turn, raw=True)
                paired_bars(ax, range(p), raw_y, raw_low, raw_high, color, raw=True)
            # Draw endpoints directly: percentile intervals need not contain the estimate.
            paired_bars(ax, range(p), y, low, high, color, raw=False).set_label(shot_label())
            ax.plot(range(p), reference, "k_", ms=10 if ZNE else 7, mew=1,
                    label="Exact Aer" if MODE == "ideal" else "Ideal reference")
            ax.axhline(0, color="0.5", lw=.5)
            ax.set_xticks(range(0, p, 3))
        axes[0].set_ylabel("Fourier coefficient")
        row_label(axes[0], rf"$\boldsymbol{{{basis}}}$")
    grid[1, 1].set_xlabel("Candidate $b'$")  # Centered: one label for all three.
    fig.align_ylabels(grid[:, 0])
    panel_labels(grid.flat)
    panel_headings(grid[0], [setting("b", b) for b in secrets])  # One secret per column.
    # Below the panels: a long source label inside one would squeeze it.
    legend = (comparison_handles("Ideal reference", "_") if ZNE
              else grid[0, -1].get_legend_handles_labels())
    fig.legend(*legend, loc="outside lower center", ncol=3 if ZNE else 2)
    save(fig, "fig4_fft")
    a = run_instance(p, q, 1, "h_only")[0]
    b = run_instance(p, q, p-1, "h_only")[0]
    summary["h_sign_max_difference"] = float(np.max(abs(a-b)))


# --------------------------------------------------- learning-curve machinery

def learning_curve(p, q, method, sample_sizes, trials=TRIALS, seed=SEED,
                   save_trials=False, *, law=None, save=True, likelihood=False):
    """Exact-secret recovery rate after each number of shots in sample_sizes.

    Each trial draws a uniform secret b (0 included), then a sequence of
    shots from law(b), P(k, second). By default that is the exact Aer law
    in the ideal run and the circuit's measured shot frequencies otherwise.
    After each prefix length the decoder scores every candidate and picks
    the best, breaking exact ties at random. likelihood=True replaces the
    adapted and pair decoders by maximum likelihood under the ideal law
    (forgetful always uses it); the same seed draws the same shots, so the
    two decoders are paired. Shots are drawn by inverse CDF from the same
    uniforms for any law, so a resampled histogram with the same seed gives
    nearly the same shots (common random numbers for histogram_bootstrap).
    For the adapted mode decoder, budgets up to plan.EXACT_MAX are computed
    exactly (adapted_exact) instead of sampled.
    Returns (rate, Wilson low, Wilson high) per sample size. The Wilson
    interval is Monte Carlo error given the law; it has zero width where exact.
    """
    rng = np.random.default_rng(seed)
    truths = rng.integers(p, size=trials)
    paths = np.zeros((trials, max(sample_sizes)), dtype=np.int32)
    width_x = 2**bits(p)
    support = p if method == "adapted" else q
    width_second = 2**(bits(p) if method == "adapted" else bits(q))
    law = law or (lambda b: run_instance(p, q, b, method)[0])
    laws = []
    for b in range(p):
        probabilities = np.maximum(law(b), 0)  # ZNE quasi-probabilities may dip below 0.
        probabilities = probabilities/probabilities.sum()
        laws.append(probabilities)
        rows = np.flatnonzero(truths == b)
        flat = probabilities.T.ravel()
        # One stream per secret, drawn shot by shot, so a shorter grid gives the same prefix.
        stream = np.random.default_rng([seed, b])
        paths[rows] = stream.choice(len(flat), size=(paths.shape[1], len(rows)), p=flat).T
    # Running per-trial statistics, updated with each batch of new shots.
    observed = np.zeros((trials, width_x*width_second))
    votes = np.zeros((trials, p), dtype=np.int64)
    signed = np.zeros((trials, p)); visits = np.zeros((trials, p))
    likelihood = likelihood or method not in ("adapted", "pairs", "h_only")
    if likelihood:
        loglaws = log_likelihood_table(p, q, method)
    correct = []
    trial_success = np.zeros((trials, len(sample_sizes)), dtype=bool)
    previous = 0
    for column, size in enumerate(sample_sizes):
        batch = paths[:, previous:size]
        k = batch % width_x; second = batch//width_x
        keep = valid(k, second, p, support)
        rows = np.broadcast_to(np.arange(trials)[:, None], batch.shape)[keep]
        if likelihood:
            np.add.at(observed, (rows, batch[keep]), 1)
            scores = observed@loglaws.T
        elif method == "adapted":
            np.add.at(votes, (rows, adapted_candidates(k[keep], second[keep], p)), 1)
            scores = votes
        elif method in ("pairs", "h_only"):
            t = second[keep]
            m = pair_label(k[keep], t, p, q)
            np.add.at(signed, (rows, m), 1-2*(t % 2))
            np.add.at(visits, (rows, m), 1)
            scores = pairs_scores(signed) if method == "pairs" else h_only_scores(signed, visits)
        tied = np.isclose(scores, scores.max(axis=1)[:, None], atol=1e-8, rtol=0)
        predictions = np.argmax(np.where(tied, rng.random(tied.shape), -1), axis=1)
        trial_success[:, column] = predictions == truths
        correct.append(int(trial_success[:, column].sum()))
        previous = size
    correct = np.array(correct)
    rate = correct/trials
    low, high = wilson(correct, trials)
    sizes = np.asarray(sample_sizes)
    small = sizes <= plan.EXACT_MAX
    if method == "adapted" and not likelihood and small.any():
        rate[small] = low[small] = high[small] = adapted_exact(laws, p, sizes[small])
    if not save:
        return rate, low, high
    decoder = "maximum_likelihood" if likelihood else dict(adapted="mode", pairs="rotated_fft",
                                                           h_only="signed_fft")[method]
    extra = dict(trial_success=trial_success, truths=truths) if save_trials else {}
    tag = "_ml" if likelihood and method in ("adapted", "pairs", "h_only") else ""
    np.savez_compressed(DATA/f"learning_{method}{tag}_p{p}_q{q}.npz", sample_sizes=sample_sizes,
                        successes=correct, rate=rate, trials=trials, seed=seed, low=low, high=high,
                        decoder=decoder, source=MODE, exact_max=plan.EXACT_MAX, **extra)
    return rate, low, high


def adapted_exact(laws, p, sizes):
    """Exact recovery of the adapted mode decoder after each budget in sizes.

    laws[b] is P(k, l) for secret b. Each shot votes for l/k mod p or, if
    k = 0 or l >= p, for nobody; the decoder takes the most-voted candidate
    and breaks ties uniformly, as in learning_curve. Enumerates every vote
    count vector, C(n+p, p) of them at budget n, so only small n is practical.
    """
    k, l = np.indices(laws[0].shape)
    # k mod p only keeps the lookup in range; invalid cells (category p) cast no vote.
    votes_for = np.where(valid(k, l, p, p), adapted_candidates(k % p, l, p), p)
    categories = np.array([np.bincount(votes_for.ravel(), law.ravel(), p+1) for law in laws])
    result = []
    for n in map(int, sizes):
        total = 0.
        for multiset in itertools.combinations_with_replacement(range(p+1), n):
            counts = np.bincount(multiset, minlength=p+1).astype(int)
            weight = math.factorial(n)/math.prod(math.factorial(c) for c in counts)
            tied = counts[:p] == counts[:p].max()
            # P(these counts | b) for every b, times the chance the tie-break picks b.
            total += (weight*np.prod(categories**counts, axis=1)*tied/tied.sum()).sum()
        result.append(total/p)
    return np.array(result)


def threshold_crossings(sample_sizes, rates, target=.9):
    """First crossing of `target` per row, interpolated linearly; inf if never."""
    sizes = np.asarray(sample_sizes)
    rates = np.atleast_2d(rates)
    above = rates >= target
    first = above.argmax(axis=1)
    result = np.full(len(rates), np.inf)
    result[above[:, 0]] = sizes[0]
    rows = np.flatnonzero(above.any(axis=1) & (first > 0))
    right = first[rows]
    left_rate = rates[rows, right-1]
    fraction = (target-left_rate)/(rates[rows, right]-left_rate)
    result[rows] = sizes[right-1] + fraction*(sizes[right]-sizes[right-1])
    return result


def _bootstrap_chunk(spec, replicates):
    """Recovery curves of one learning-curve spec on histogram resamples `replicates`."""
    p, q, method = spec["p"], spec["q"], spec["method"]
    counts = [np.asarray(run_instance(p, q, b, method)[1]) for b in range(p)]
    curves = []
    for replicate in replicates:
        # Keyed by circuit and replicate only, so every decoder of a circuit sees the same resamples.
        rng = np.random.default_rng([SEED, p, q, zlib.crc32(method.encode()), replicate])
        draws = [rng.multinomial(int(c.sum()), (c/c.sum()).ravel()).reshape(c.shape) for c in counts]
        rate, _, _ = learning_curve(p, q, method, spec["sizes"], seed=spec["seed"], save=False,
                                    likelihood=spec["likelihood"], law=lambda b: draws[b])
        curves.append(rate)
    return np.array(curves)


def histogram_bootstrap(specs, repeats=None, chunk=25):
    """Learning curves recomputed on multinomial resamples of the measured histograms.

    Each replicate redraws every secret's fold-1 histogram (same shot total)
    and reruns learning_curve with the spec's own seed, so the replicate
    curves differ only through the histograms. Their spread is the
    finite-shot uncertainty of the data. It treats shots as independent and
    identically distributed within a job and the jobs of different secrets
    as independent, so it does not capture device drift or variation
    between the debiasing variants pooled in each job. specs: dicts with
    p, q, method, likelihood, sizes, seed. Returns one (repeats, len(sizes))
    array per spec. HSP_WORKERS sets the number of processes.
    """
    repeats = repeats or plan.LEARNING_BOOTSTRAPS
    workers = int(os.environ.get("HSP_WORKERS", max(1, min(8, (os.cpu_count() or 2)-2))))
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(name, "1")  # One thread per worker process.
    blocks = [range(start, min(start+chunk, repeats)) for start in range(0, repeats, chunk)]
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(workers, mp_context=context, initializer=_configure,
                             initargs=(MODE, RESULTS, False, OUTPUT, SHOTS)) as pool:
        futures = [[pool.submit(_bootstrap_chunk, spec, block) for block in blocks] for spec in specs]
        return [np.concatenate([future.result() for future in row]) for row in futures]


def classical_curve(p, q, query_counts):
    """Optimal classical recovery curve, saved apart from the sampled quantum data."""
    recovery = optimal_curve(p, q, query_counts)
    np.savez_compressed(DATA/f"learning_classical_p{p}_q{q}.npz",
                        query_counts=query_counts, recovery=recovery, p=p, q=q,
                        source="exact optimal classical algorithm (classical.py)",
                        fallback="uniform guess over the secrets not ruled out")
    return recovery


# ------------------------------------------------- Figures 5 and 6

def plot_forgetful(summary):
    """Figure 5: forgetful joint P(k, l); secrets in one orbit give identical laws."""
    p, q = plan.FORGETFUL
    secrets = plan.forgetful_secrets(p, q)
    fig, axes = plt.subplots(1, 3, figsize=(COLUMN, 1.4), sharey=True, layout="constrained")
    probabilities = []
    # The largest ideal joint probability is 1/p (at k=0).
    vmax = max(.08, 1.05/p)
    for ax, b in zip(axes, secrets):
        prob, counts = run_instance(p, q, b, "forgetful")
        probabilities.append(prob)
        artist = ax.imshow((counts/counts.sum())[:p, :q].T, origin="lower", aspect="auto", vmin=0, vmax=vmax,
                           cmap="viridis", extent=(-.5, p-.5, -.5, q-.5))
        ax.set(xlabel="$k$", xticks=range(0, p, 3), yticks=range(q))
        ax.tick_params(which="both", length=0)  # Tick marks would sit on the colored cells.
    axes[0].set_ylabel("$l$")
    panel_labels(axes)
    panel_headings(axes, [setting("b", b) for b in secrets])
    bar = fig.colorbar(artist, ax=axes, label="Joint probability", pad=.02, aspect=12, ticks=[0, .05, .1, .15])
    bar.outline.set_linewidth(.5)
    save(fig, "fig5_forgetful")
    summary["forgetful_same_orbit_max_difference"] = float(np.max(abs(probabilities[0]-probabilities[1])))
    summary["forgetful_other_orbit_tv"] = float(np.sum(abs(probabilities[0]-probabilities[2]))/2)


def learning_row(summary, axes, *, p, q, name, seed):
    """One row of Figure 6: recovery vs total queries at (p, q), one panel per
    measurement. Solid: the measured shots; dashed: the same decoder and trials
    on the noiseless law. Adapted and pairs are decoded both natively (mode,
    FFT) and by maximum likelihood on the same shots; forgetful only by maximum
    likelihood. `name` keys the row's numbers in the summary."""
    # Every budget to four, then powers of two and their midpoints (x1.5), to LEARNING_MAX.
    grid = [0, 1, 2, 3, 4]+[m*2**e for e in range(3, 40) for m in (1, 1.5)]
    sizes = np.array([n for n in grid if n <= LEARNING_MAX]+[LEARNING_MAX], dtype=int)
    sizes = np.unique(sizes)
    one = np.flatnonzero(sizes == 1)[0]
    final, at_one, noiseless = {}, {}, {}
    query_counts = np.arange(LEARNING_MAX+1)
    classical = classical_curve(p, q, query_counts)
    cap = (p+q-1)/(p*q)  # Forgetful can at best identify the secret's orbit.
    # Maximum likelihood first, so the native decoder is drawn on top of it.
    curves = dict(adapted=[("adapted_ml", True), ("adapted", False)],
                  pairs=[("pairs_ml", True), ("pairs", False)],
                  forgetful=[("forgetful", True)])
    specs = [dict(p=p, q=q, method=method, likelihood=ml, sizes=sizes, seed=seed+index, key=key)
             for index, (method, decoders) in enumerate(curves.items()) for key, ml in decoders]
    points = {s["key"]: learning_curve(p, q, s["method"], sizes, seed=s["seed"], likelihood=s["likelihood"])
              for s in specs}
    bootstrap = {}
    if MODE != "ideal":
        # Measured histograms: intervals over histogram resamples (Monte Carlo error alone
        # would understate them). Ideal law: the Wilson Monte Carlo interval is complete.
        for spec, replicates in zip(specs, histogram_bootstrap(specs)):
            key, (mean, _, _) = spec["key"], points[spec["key"]]
            low, high = np.quantile(replicates, [.025, .975], axis=0)
            points[key] = (mean, low, high)
            n90 = threshold_crossings(sizes, replicates)
            bootstrap[key] = dict(max_bias=float(np.max(abs(replicates.mean(axis=0)-mean))),
                                  max_monte_carlo_se=float(np.max(np.sqrt(mean*(1-mean)/TRIALS))),
                                  censored_n90=int(np.isinf(n90).sum()))
            np.savez_compressed(DATA/f"learning_{key}_p{p}_q{q}_bootstrap.npz", sample_sizes=sizes,
                                curves=replicates, low=low, high=high, n90=n90, repeats=len(replicates),
                                resampling="multinomial, every secret's fold-1 histogram", source=MODE)
    for index, (ax, (method, decoders)) in enumerate(zip(axes, curves.items())):
        ax.plot(query_counts, classical, color="0.55", ls="-.", lw=1)
        ax.axhline(1/p, color="0.75", ls=":", lw=.9)
        if method == "forgetful":
            ax.axhline(cap, color="0.4", ls=":", lw=1)
        for key, ml in decoders:
            color = LEARNING_COLORS[ml]
            mean, low, high = points[key]
            ax.fill_between(sizes, low, high, color=color, alpha=.35, lw=0)
            ax.plot(sizes, mean, color=color, lw=1)
            final[key] = dict(success=float(mean[-1]), low=float(low[-1]), high=float(high[-1]))
            at_one[key] = dict(success=float(mean[one]), low=float(low[one]), high=float(high[one]))
            if MODE != "ideal":
                # Noiseless reference: the exact Aer law with the same seed and trials,
                # so the gap to the sampled curve is the effect of noise.
                ideal, _, _ = learning_curve(p, q, method, sizes, seed=seed+index, save=False, likelihood=ml,
                                             law=lambda b, method=method: ideal_instance(p, q, b, method)[0])
                ax.plot(sizes, ideal, color=color, ls="--", lw=.8)
                noiseless[key] = dict(success=ideal.tolist(), final=float(ideal[-1]),
                                      at_one=float(ideal[one]))
    if bootstrap:
        summary[f"{name}_bootstrap"] = dict(
            repeats=plan.LEARNING_BOOTSTRAPS, trials=TRIALS, exact_max=plan.EXACT_MAX,
            interval="pointwise 95% percentile over multinomial resamples of every secret's histogram",
            assumptions="shots i.i.d. within a job, secrets independent; excludes drift and "
                        "variation between pooled debiasing variants",
            curves=bootstrap)
    axes[0].set_ylabel("Exact-secret recovery")
    classical_summary = dict(p=p, q=q, guarantee=guarantee(p, q),
        n90_integer=int(query_counts[np.flatnonzero(classical >= .9)[0]]),
        recovery=classical[:guarantee(p, q)+1].tolist())
    if noiseless:
        summary[f"{name}_noiseless"] = dict(p=p, q=q, seed=seed, trials=TRIALS, sample_sizes=sizes.tolist(),
            law="exact ideal Aer statevector probabilities", curves=noiseless)
    summary[name] = dict(p=p, q=q, alpha=int(group_parameters(p, q)[0]),
        trials=TRIALS, seed=seed, sample_sizes=sizes.tolist(),
        final=final, at_one=at_one, classical=classical_summary, forgetful_cap=cap,
        adapted_one_query_theory=((p-1)*q+1)/p**2)


def learning_handles(data, noiseless):
    """Legend entries for Figure 6."""
    handles = [plt.Line2D([], [], color=LEARNING_COLORS[False], label="Native decoder (mode / FFT)"),
               plt.Line2D([], [], color=LEARNING_COLORS[True], label="Maximum likelihood"),
               plt.Line2D([], [], color="black", label=data)]
    if noiseless:
        handles.append(plt.Line2D([], [], color="black", ls="--", lw=.8, label="Noiseless"))
    return handles + [plt.Line2D([], [], color="0.55", ls="-.", label="Optimal classical"),
                      plt.Line2D([], [], color="0.4", ls=":", label="Forgetful orbit cap"),
                      plt.Line2D([], [], color="0.75", ls=":", lw=.9, label="Random guess")]


def plot_learning(summary):
    """Figure 6: learning curves, rows LEARNING (q = 2) and AFFINE (q = p-1),
    columns adapted, adjacent pairs and forgetful; optimal classical in each."""
    fig, axes = plt.subplots(2, 3, figsize=(PAGE, 4.1), sharex=True, sharey=True, layout="constrained")
    learning_row(summary, axes[0], p=plan.LEARNING[0], q=plan.LEARNING[1], name="learning", seed=SEED)
    learning_row(summary, axes[1], p=plan.AFFINE[0], q=plan.AFFINE[1], name="learning_affine",
                 seed=SEED+plan.AFFINE[0]*10000+plan.AFFINE[1])
    panel_labels(axes.flat)
    panel_headings(axes[0], ["Adapted", "Adjacent pairs", "Forgetful abelian"])
    for row, q in zip(axes, [plan.LEARNING[1], plan.AFFINE[1]]):
        row_label(row[0], setting("q", q))
    xmax = min(LEARNING_XMAX, LEARNING_MAX)
    # Linear from 0 to 2 so the zero-query point shows; then powers of four.
    ticks = [0, 1, 2, 4]+[4**e for e in range(2, 20) if 4**e <= xmax/4]+[xmax]
    for ax in axes.flat:
        ax.set(xscale="symlog", xlim=(0, xmax), ylim=(0, 1.03))
    for ax in axes[1]:
        ax.set_xticks(ticks, labels=[str(n) for n in ticks])
        ax.set_xlabel("Total oracle queries")
    fig.legend(handles=learning_handles("Ideal" if MODE == "ideal" else "Measured (raw)", MODE != "ideal"),
               loc="outside lower center", ncol=4)
    save(fig, "fig6_learning")


# -------------------------------------------------------------------- drivers

def _configure(mode, results=None, zne=False, output=None, shots=None):
    """Point the module at a data source and an output directory."""
    global MODE, RESULTS, ZNE, FIGURES, DATA, SHOT_NOTE, LEARNING_MAX, OUTPUT, SHOTS
    MODE, RESULTS, ZNE, OUTPUT, SHOTS = mode, results, zne, output, shots
    FIGURES = output/"figures"
    DATA = output/"analysis" if mode == "ideal" else output/"analysis"/("zne" if zne else "raw")
    DATA.mkdir(parents=True, exist_ok=True)
    LEARNING_MAX = plan.QUERY_LIMITS["ideal" if mode == "ideal" else "sampled"]
    SHOT_NOTE = f", {shots:,} shots/circuit" if mode != "ideal" else ""
    zne_replicates.cache_clear()
    style.apply()


def _base_summary():
    return dict(seed=SEED, trials=TRIALS,
                versions=dict(python=platform.python_version(), qiskit=qiskit.__version__,
                              qiskit_aer=qiskit_aer.__version__, numpy=np.__version__,
                              matplotlib=matplotlib.__version__))


def _raw_figures(summary):
    plot_weak(summary)
    plot_forgetful(summary)
    plot_learning(summary)


def _zne_figures(summary):
    plot_adapted(summary)
    plot_success(summary)
    plot_fft(summary)


def _write(summary):
    (DATA/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(f"Wrote {DATA/'summary.json'} and figures in {FIGURES}", flush=True)


def analyse_ideal(output=Path("runs/ideal")):
    """Ideal Aer reference: all six figures from IDEAL_SHOTS-shot noiseless histograms."""
    _configure("ideal", output=output)
    summary = _base_summary()
    summary.update(shots_per_circuit=plan.IDEAL_SHOTS,
                   source="ideal Aer statevector circuits; NumPy resampling for learning curves")
    _raw_figures(summary)
    _zne_figures(summary)
    errors = [theory_error(p, q, b, method) for method, p, q, b in plan.circuits()]
    summary["aer_circuits"] = len(errors)
    summary["max_theory_error"] = max(errors)
    _write(summary)


def analyse_run(run):
    """All six figures from runs/<target>/results, into runs/<target>/figures."""
    settings = json.loads((run/"run.json").read_text())
    for zne in (False, True):
        _configure(settings["target"], results=run/"results", zne=zne, output=run, shots=settings["shots"])
        summary = _base_summary()
        summary.update(source=f"{PLOT_LABELS[MODE]}{SHOT_NOTE} raw histograms; learning curves "
                              "resample finite empirical distributions",
                       interval_scope="learning-curve intervals: percentiles over multinomial resamples "
                                      "of the histograms (finite-shot error); excluding device drift and "
                                      "variation between pooled debiasing variants")
        if zne:
            summary["zne"] = dict(method="exponential", fold=plan.ZNE_FOLD, bootstraps=plan.BOOTSTRAPS,
                                  figures=ZNE_FIGURES, noise_check=zne_noise_check(),
                                  interval="95% percentile over multinomial resamples of both histograms")
            for row in summary["zne"]["noise_check"]:
                if row["clean_folded"] >= row["clean_base"]:
                    print(f"WARNING {row['circuit']}: fold {plan.ZNE_FOLD} is not noisier than fold 1; "
                          "the backend may have cancelled the folded gates", flush=True)
            _zne_figures(summary)
        else:
            _raw_figures(summary)
            summary["shot_accounting"] = []
            for method, p, q, b in plan.circuits():
                with np.load(RESULTS/f"{method}_p{p}_q{q}_b{b}.npz") as stored:
                    summary["shot_accounting"].append(dict(
                        file=f"{method}_p{p}_q{q}_b{b}.npz", shots=int(stored["shots"]),
                        invalid=int(stored["invalid"]), k_zero=int(stored["k_zero"])))
        _write(summary)
