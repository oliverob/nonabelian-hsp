# Non-abelian hidden subgroup problem on IonQ Forte: code and data

Code and hardware data for

> O. O'Brien, *Solving a non-abelian Hidden Subgroup Problem on a trapped-ion
> quantum computer: the power of measurement basis* (2026).

The experiment runs the full hidden subgroup algorithm for the q-hedral
groups ℤ_p ⋊ ℤ_q (p ≤ 7) on the IonQ Forte trapped-ion computer. It compares
weak Fourier sampling, strong Fourier sampling in an adapted basis, strong
Fourier sampling of adjacent pairs, and forgetful abelian sampling.

This repository lets you:

1. **Regenerate the paper's figures** (Figs. 1–6) from the published hardware
   histograms, with no IonQ account needed.
2. **Run the experiment yourself** on IonQ's noise-model emulator or on Forte
   hardware, or change the plan to run your own circuits. You get the same
   figures from your own data.

## Installation

Python 3.12 and `git` (to install PyEncode) are needed.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests        # 44 tests, about 10 s
```

Run every command from the repository root.

## 1. Regenerate the paper's figures

```sh
.venv/bin/python pipeline.py figures hardware
```

This reads `runs/hardware/results/` and writes PDF and PNG figures into
`runs/hardware/figures/`, along with the numbers behind them in
`runs/hardware/analysis/{raw,zne}/summary.json`. Figs. 1 and 5 take seconds.
The Fig. 6 learning curves are recomputed on 1,000 resamples of the
histograms, which takes about 75 minutes with 8 worker processes; set
`HSP_WORKERS` to change the number. The copies already in
`runs/hardware/figures/` are this command's output.

| Paper | File | Shows | Data |
| --- | --- | --- | --- |
| Fig. 1 | `fig1_weak` | Weak Fourier sampling, p = 7, q = 2 and 6 | raw |
| Fig. 2 | `fig2_adapted` | Adapted basis, p = 7, q = 6 | ZNE, with raw overlaid |
| Fig. 3 | `fig3_success_scaling` | Adapted-basis success against q and p | ZNE, with raw overlaid |
| Fig. 4 | `fig4_fft` | Adjacent pairs with HT and with H, p = 7, q = 2 | ZNE, with raw overlaid |
| Fig. 5 | `fig5_forgetful` | Forgetful abelian sampling, p = 7, q = 3 | raw |
| Fig. 6 | `fig6_learning` | Recovery against oracle queries, p = 7, q = 2 and 6 | raw |

"Raw" means IonQ's debiased counts without further mitigation. "ZNE" is
zero-noise extrapolation from the fold-1 and fold-3 runs of each circuit.
Figs. 7–10 of the paper are circuit diagrams, not data, and are not produced
here.

## 2. Run the experiment yourself

The plan of circuits lives in [plan.py](plan.py). As shipped, it reproduces
the paper's 75 hardware jobs: 62 circuits at 2,048 shots, plus ZZ-folded
copies of the 13 circuits behind the zero-noise extrapolated figures. Edit
the figure parameters there to run different primes, divisors q or secrets.

**Noiseless preview (free, offline).** Draws the same six figures from ideal
Aer simulation, into `runs/ideal/figures/`:

```sh
.venv/bin/python pipeline.py ideal
```

**IonQ emulator or hardware.** Put your IonQ API key in `.env`; it is read
automatically and git-ignored.

```sh
echo "IONQ_API_KEY=..." > .env
.venv/bin/python pipeline.py run emulator     # prepare, submit, wait, fetch, figures
```

`run` performs these steps, which are also available on their own:

| Command | What it does |
| --- | --- |
| `prepare TARGET` | Compiles every job to Forte native gates in `runs/TARGET/jobs/`, offline. Each gate list is re-simulated and must match the ideal circuit. |
| `estimate TARGET` | Prices the jobs with free IonQ dry runs (`runs/TARGET/costs.json`). |
| `submit TARGET` | Sends every job that has no receipt yet. Rerunning never sends a job twice. |
| `fetch TARGET [--wait]` | Downloads finished histograms into `runs/TARGET/results/`. |
| `status TARGET` | Counts prepared, submitted and fetched jobs. |
| `figures TARGET` | Draws Figs. 1–6 from the results. |

`TARGET` is `emulator` (IonQ's cloud simulator with the `forte-1` noise model)
or `hardware` (`qpu.forte-1`). Every step can be interrupted and rerun.

**Hardware is locked by default.** `HARDWARE_SUBMISSION_ALLOWED = False` in
[forte.py](forte.py) blocks every request to a QPU, dry runs included. Set it
to `True` only when you intend to spend hardware credit. Hardware jobs use
IonQ's debiasing: each 2,048-shot job runs as 32 variants of 64 shots over
different qubit assignments. The emulator cannot debias, so its jobs run
without it; otherwise emulator and hardware jobs carry identical gate lists.

`runs/hardware/` already holds the published run. To start a new hardware
run, move that folder aside first.

## The published hardware data

`runs/hardware/` contains every job of the run on IonQ Forte, 9–12 October
2026:

```
runs/hardware/
  run.json        target, backend, shots, ZNE fold, fingerprint of the circuit sources
  jobs/           the exact job payloads sent to IonQ (native gate lists), and
                  manifest.json with each job's gate counts and SHA-256
  results/
    <job>.npz           counts[k, second] for the measured registers (workspace
                        summed out), plus p, q, b, method, zne_fold, shots,
                        invalid and k_zero shot counts, IonQ job id and source
    <job>_raw.json      the histogram returned by IonQ (counts, recovered from
                        the debiased probabilities)
    <job>_job.json      IonQ's job record: timings, compilation, debiasing variants
    <job>_receipt.json  the submission response
    bit_order.json      the bit order read from the bit-order probe job
  figures/        Figs. 1-6 (PDF and PNG)
  analysis/       summary.json and learning-curve arrays behind the figures
```

Job names are `<method>_p<p>_q<q>_b<b>` with `_fold3` for ZZ-folded copies.
The run also includes 11 circuits that appear in no figure (`NOT_PLOTTED` in
[plan.py](plan.py)). IonQ account and project identifiers have been removed
from the job records.

`pipeline.py prepare hardware` regenerates the 76 files in `jobs/`
byte for byte. `figures` refuses to analyse a run whose `run.json`
fingerprint does not match the current `circuits.py`, `blocks.py` and
`fourier_circuits.json`, so the data and the circuits cannot drift apart.

## How the code fits together

```
plan.py ──► which circuits, shots, seeds, figure parameters
   │
circuits.py ──► logical circuits (theory.py: their closed-form outcome laws)
   │   └── blocks.py ──► Fourier transforms (fourier_circuits.json) + oracle diagonals
   │                     │
   │               simulation.py ──► ideal Aer probabilities (cache in runs/ideal/aer)
   ▼
forte.py ──► native GPi/GPi2/ZZ gate lists, ZZ folding, job payloads, IonQ API, hardware lock
   │
pipeline.py ──► prepare → submit → fetch → figures   (runs/<target>/)
   │
analysis.py ──► Figs. 1-6   (uses decoding.py, classical.py, style.py)
```

| File | What it does |
| --- | --- |
| [plan.py](plan.py) | The circuits (method, p, q, b) each figure reads, the folded subset, and every fixed setting: shots, seeds, trial and bootstrap counts. |
| [circuits.py](circuits.py) | Builds each sampling circuit: state preparation, the oracle, and the measurement basis for each method. |
| [theory.py](theory.py) | The exact outcome law of each circuit, used to check simulations and by the maximum-likelihood decoders. |
| [blocks.py](blocks.py) | Low-ZZ building blocks: the stored Fourier circuits and the exponent-encoding oracle as a phase polynomial. |
| [fourier_search.py](fourier_search.py) | The offline numerical search that found the stored F₃, F₅, F₆, F₇ circuits in `fourier_circuits.json`. Needs `jax`; only rerun it to regenerate the JSON. |
| [simulation.py](simulation.py) | Ideal Aer statevector simulation, checked against `theory.py`. |
| [forte.py](forte.py) | Exact compilation to Forte native gates (one ZZ per two-qubit gate, virtual Z), ZZ folding, gate-list checks, job payloads, IonQ API calls and histogram import. |
| [pipeline.py](pipeline.py) | The command-line workflow above. |
| [decoding.py](decoding.py) | Turns shots into a score per candidate secret. Adapted: mode of l·k⁻¹. Adjacent pairs: π/4-rotated FFT. H only: signed FFT. Forgetful: maximum likelihood. |
| [classical.py](classical.py) | Exact recovery curve of the optimal classical algorithm. |
| [analysis.py](analysis.py) | Draws the figures, with intervals, zero-noise extrapolation and learning curves. |
| [style.py](style.py) | Figure style (Physical Review two-column sizes). |
| `tests/` | Circuit laws, native compilation, folding, decoders, statistics, the classical baseline, and the mocked prepare → submit → fetch workflow, including the hardware lock. |

## Method notes

- **Methods.** `adapted` stores the second register as field elements α^t
  and applies F_p to both registers. `pairs` measures adjacent labels
  (t, t+1) in the T-then-H basis. `h_only` uses H alone, a control that cannot
  tell b from −b. `forgetful` applies F_q, which only identifies the orbit of
  b. Fig. 1's weak Fourier sampling keeps only the irrep of each forgetful
  outcome (`analysis.irrep_bins`).
- **Two-qubit cost.** Each Fourier transform is followed by its measurement
  and sees only labels below its size, so it only has to be correct there, up
  to a phase per outcome. The stored circuits use 2, 8, 7 and 10 ZZ for F₃, F₅,
  F₆, F₇. The exponent-encoding oracle is one phase polynomial on (t, w).
- **Queries.** Every quantum shot counts as one oracle query, including shots
  the decoder discards (k = 0 or an invalid label). Classical curves are exact
  calculations, not Monte Carlo estimates.
- **Learning curves.** Each point uses 20,000 trials with a uniformly random
  secret, drawing shots from that circuit's measured histogram. Bands are 95%
  intervals over 1,000 multinomial resamples of the histograms. Dashed curves
  apply the same decoder to the exact ideal law.
- **ZNE.** Each ZZ gate is replaced by ZZ·ZZ⁻¹·ZZ, which triples the
  two-qubit noise without changing the circuit. The estimate is extrapolated
  exponentially, which is exact for global depolarizing noise. Intervals are
  95% percentiles over 1,000 resamples of both histograms.
- **Reproducibility.** Every random number derives from `plan.SEED`, so the
  figures are deterministic given the histograms.

## Acknowledgements

I am grateful to FormationQ and IonQ for providing access to the IonQ Forte
trapped-ion quantum computer.

The code for circuit generation and compilation, the classical baseline,
decoding, uncertainty analysis and figure generation was written with the
assistance of Claude Opus 5.5 (Anthropic). The author designed the study,
checked and validated all code and results, and takes full responsibility
for them.

## Citation

If you use this code or data, please cite the paper above (see also
[CITATION.cff](CITATION.cff)).

## License

[MIT](LICENSE).
