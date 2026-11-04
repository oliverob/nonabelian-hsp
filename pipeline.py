"""Run the experiment on the IonQ Forte emulator or on Forte hardware, and draw the figures.

The whole workflow, for TARGET = emulator or hardware:

    python pipeline.py run TARGET

That runs each step below in turn and can be rerun at any point; finished
steps are skipped. Each target keeps everything in its own folder runs/TARGET:

    prepare TARGET    Compile the plan's circuits to native job files in runs/TARGET/jobs,
                      re-simulating each serialized gate list against the ideal circuit.
                      Offline and free.
    submit TARGET     Send every job that has no receipt yet (results/*_receipt.json).
    fetch TARGET      Download completed histograms into results/*.npz. With --wait,
                      keep polling until every job is back.
    status TARGET     Count prepared, submitted and fetched jobs.
    figures TARGET    Figures 1-6 from the results, into runs/TARGET/figures.
    estimate TARGET   Price the hardware jobs with free dry runs (writes runs/hardware/costs.json).
    ideal             The same figures from noiseless Aer simulation, into runs/ideal/figures.

runs/hardware holds the published hardware data, so `python pipeline.py figures
hardware` redraws the paper's figures without any network access.

To start a target again from scratch, delete (or move) its runs/TARGET folder.
prepare records circuits.fingerprint() in run.json, and submit and figures refuse
a run prepared from different circuit sources.
Hardware jobs, dry runs included, stay blocked until HARDWARE_SUBMISSION_ALLOWED
is set to True in forte.py. The API key is read from IONQ_API_KEY or from .env.
"""
import argparse
from datetime import date
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from circuits import build_circuit, fingerprint
import forte
import plan
from simulation import ideal_instance, simulate

RUNS = Path("runs")


def load_env():
    """Read KEY=VALUE lines from .env into the environment (never printed)."""
    path = Path(".env")
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        key, sep, value = line.strip().partition("=")
        if sep and not key.startswith("#"):
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2)+"\n")


# --------------------------------------------------------------------- steps

def ideal():
    """Figures from the noiseless circuits (Aer simulation) in runs/ideal."""
    import analysis
    analysis.analyse_ideal()


def prepare(target):
    """Write every job of the plan, plus the bit-order probe, to runs/TARGET/jobs.

    One job per plan.jobs() row, ZZ-folded for fold > 1. Each serialized gate list is rebuilt and simulated, and must reproduce
    the ideal probabilities to 1e-9. The manifest records gate counts, that
    error, and each file's SHA-256, which submit() re-checks.
    """
    run = RUNS/target
    jobs = run/"jobs"
    if (jobs/"manifest.json").exists():
        raise FileExistsError(f"{jobs} is already prepared; delete {run} to start {target} again")
    jobs.mkdir(parents=True, exist_ok=True)
    cap = forte.MAX_ZZ if target == "hardware" else None
    manifest = []

    def add(filename, payload, row):
        encoded = json.dumps(payload, indent=2)+"\n"
        (jobs/filename).write_text(encoded)
        manifest.append(dict(file=filename, **row, sha256=hashlib.sha256(encoded.encode()).hexdigest()))

    add("bit_order_probe.json", forte.probe_payload(target),
        dict(purpose="bit_order_probe", qubits=3, zz=0, shots=100))
    compiled = {}
    for method, p, q, b, fold in plan.jobs():
        circuit = build_circuit(p, q, b, method)
        reference, _ = ideal_instance(p, q, b, method)
        if (method, p, q, b) not in compiled:
            compiled[method, p, q, b] = forte.native_gates(circuit)
        gates = compiled[method, p, q, b]
        body = dict(qubits=circuit.num_qubits, gateset="native",
                    circuit=forte.fold_zz(gates, fold) if fold != 1 else gates)
        counts = forte.validate_native(body, cap)
        # Check the exact JSON gate list, not just its source circuit.
        actual, _ = simulate(forte.native_as_qiskit(body, circuit.metadata), shots=16)
        error = float(np.max(abs(actual-reference)))
        if error > 1e-9:
            raise AssertionError(f"Native translation mismatch {circuit.name}: {error}")
        name = circuit.name + (f"_fold{fold}" if fold != 1 else "")
        metadata = dict(circuit.metadata, **({"zne_fold": fold} if fold != 1 else {}))
        add(f"{name}.json", forte.job_payload(name, target, plan.SHOTS, body, metadata),
            dict(circuit.metadata, zne_fold=fold, qubits=circuit.num_qubits,
                 zz=counts["zz"], single_qubit=counts["gpi"]+counts["gpi2"], shots=plan.SHOTS,
                 max_probability_error=error))
        print(f"{target} {name}: {counts['zz']} ZZ, verified {error:.2g}", flush=True)
    write_json(jobs/"manifest.json", manifest)
    write_json(run/"run.json", dict(target=target, **forte.TARGETS[target], shots=plan.SHOTS,
                                    zne_fold=plan.ZNE_FOLD,
                                    prepared=date.today().isoformat(), circuits=fingerprint()))
    print(f"Prepared {len(manifest)} jobs in {jobs}; nothing was submitted")


def require_current(run):
    """Refuse a run whose jobs were compiled from different circuit sources."""
    prepared = read_json(run/"run.json").get("circuits")
    if prepared != fingerprint():
        raise RuntimeError(f"{run} was prepared from different circuits (circuits.py, blocks.py or "
                           f"fourier_circuits.json changed); move {run} aside and prepare again")


def audited_payload(run, row):
    """Load a prepared job, refusing anything changed since prepare() audited it."""
    encoded = (run/"jobs"/row["file"]).read_bytes()
    if hashlib.sha256(encoded).hexdigest() != row["sha256"]:
        raise ValueError(f"{row['file']} changed since it was prepared; delete the run and prepare again")
    payload = json.loads(encoded)
    cap = forte.MAX_ZZ if forte.is_hardware(payload) else None
    forte.validate_native(payload["input"], cap)
    return payload


def receipt_path(run, row):
    return run/"results"/(Path(row["file"]).stem+"_receipt.json")


def is_data(row):
    """Every manifest row except the bit-order probe has a results/<job>.npz."""
    return row.get("purpose") != "bit_order_probe"


def submit(target):
    """Send every job without a receipt, in manifest order (the bit-order probe first,
    then each circuit's copies one after another). Rerunning never duplicates a job."""
    run = RUNS/target
    require_current(run)
    manifest = read_json(run/"jobs"/"manifest.json")
    (run/"results").mkdir(exist_ok=True)
    sent = 0
    for row in manifest:
        receipt = receipt_path(run, row)
        if receipt.exists():
            continue
        response = forte.api_request("/jobs", audited_payload(run, row))
        write_json(receipt, dict(response=response, file=row["file"]))
        sent += 1
        print(f"Submitted {row['file']}: {response['id']}", flush=True)
    print(f"Submitted {sent} new jobs; {len(manifest)-sent} already had receipts")


def fetch(target, wait=False, poll=60):
    """Download finished histograms; convert them to arrays once the bit order is known.

    Raw artifacts are kept as results/<job>_raw.json. The bit-order probe is
    read first: its dominant label says which end of each bitstring is qubit
    0. Each circuit is then saved as results/<job>.npz with its counts
    [k, second] (workspace summed out), job id, and shot accounting.
    """
    run = RUNS/target
    manifest = read_json(run/"jobs"/"manifest.json")
    results = run/"results"
    while True:
        waiting, failed = [], []
        for row in manifest:
            stem = Path(row["file"]).stem
            if (results/f"{stem}_raw.json").exists() or not receipt_path(run, row).exists():
                continue
            job_id = read_json(receipt_path(run, row))["response"]["id"]
            job, artifact = forte.fetch_histogram(job_id)
            write_json(results/f"{stem}_job.json", job)
            if artifact is not None:
                write_json(results/f"{stem}_raw.json", artifact)
            elif job["status"] in ("failed", "canceled"):
                failed.append(f"{stem} ({job['status']})")
            else:
                waiting.append(stem)
        if failed:
            raise RuntimeError(f"Jobs did not complete: {', '.join(failed)}. Delete their receipts "
                               "and run submit again to resend them")
        order = bit_order(run)
        if order:
            for row in manifest:
                convert(run, row, order)
        unsubmitted = sum(not receipt_path(run, row).exists() for row in manifest)
        done = sum((results/(Path(row["file"]).stem+".npz")).exists() for row in manifest if is_data(row))
        total = sum(is_data(row) for row in manifest)
        print(f"{done}/{total} jobs fetched, {len(waiting)} still running, "
              f"{unsubmitted} not submitted", flush=True)
        if not wait or not waiting:
            return done == total
        time.sleep(poll)


def bit_order(run):
    """'q0-left'/'q0-right' from the fetched probe (saved in results/bit_order.json)."""
    saved = run/"results"/"bit_order.json"
    if saved.exists():
        return read_json(saved)["bit_order"]
    raw = run/"results"/"bit_order_probe_raw.json"
    if not raw.exists():
        return None
    histogram = forte.output_histogram(read_json(raw))
    order = forte.probe_bit_order(histogram, 3)
    if order is None:
        raise ValueError(f"Bit-order probe is ambiguous: {histogram}")
    write_json(saved, dict(bit_order=order, histogram=histogram))
    print(f"Bit order from probe: {order}")
    return order


def convert(run, row, order):
    """results/<job>_raw.json -> results/<job>.npz (the file analysis.py reads)."""
    stem = Path(row["file"]).stem
    raw, output = run/"results"/f"{stem}_raw.json", run/"results"/f"{stem}.npz"
    if not is_data(row) or output.exists() or not raw.exists():
        return
    payload = read_json(run/"jobs"/row["file"])
    counts = forte.import_histogram(forte.output_histogram(read_json(raw)), payload["metadata"],
                                    payload["input"]["qubits"], order)
    if counts.sum() != payload["shots"]:
        raise ValueError(f"{stem}: {counts.sum()} shots returned, {payload['shots']} requested")
    target = "hardware" if forte.is_hardware(payload) else "emulator"
    common = dict(shots=int(counts.sum()), job_id=read_json(receipt_path(run, row))["response"]["id"],
                  source=forte.SOURCES[target], noise_model=payload.get("noise", {}).get("model", "none"),
                  bit_order=order)
    p, q = row["p"], row["q"]
    support = p if row["method"] == "adapted" else q
    np.savez_compressed(output, counts=counts, p=p, q=q, b=row["b"], method=row["method"],
                        zne_fold=row["zne_fold"], invalid=int(counts.sum()-counts[:p, :support].sum()),
                        k_zero=int(counts[0, :support].sum()), **common)


def status(target):
    run = RUNS/target
    if not (run/"jobs"/"manifest.json").exists():
        print(f"{target}: not prepared")
        return
    manifest = read_json(run/"jobs"/"manifest.json")
    stems = [Path(row["file"]).stem for row in manifest]
    count = lambda suffix: sum((run/"results"/f"{stem}{suffix}").exists() for stem in stems)
    print(f"{target}: {len(manifest)} jobs prepared, {count('_receipt.json')} submitted, "
          f"{count('_raw.json')} completed, {count('.npz')} converted (the probe has no .npz)")


def estimate(target):
    """Price every job with free dry runs (IonQ compiles and prices, never runs them).

    The first dry run must be confirmed as a dry run by IonQ before any other
    is sent. Dry-run receipts live in runs/TARGET/estimates, never beside real
    receipts. Rerunning resumes.
    """
    run = RUNS/target
    require_current(run)
    manifest = read_json(run/"jobs"/"manifest.json")
    receipts = run/"estimates"
    receipts.mkdir(exist_ok=True)
    confirmed = False
    for row in manifest:
        receipt = receipts/(Path(row["file"]).stem+"_receipt.json")
        if receipt.exists():
            continue
        response = forte.api_request("/jobs", dict(audited_payload(run, row), dry_run=True))
        write_json(receipt, dict(response=response, file=row["file"], dry_run=True))
        print(f"Dry run {row['file']}: {response['id']}", flush=True)
        if not confirmed:
            for _ in range(24):
                cost = forte.job_cost(response["id"])
                if cost is not None:
                    break
                time.sleep(5)
            else:
                raise TimeoutError("Could not confirm the first dry run; rerun estimate later")
            if cost.get("dry_run") is not True:
                raise RuntimeError(f"Job {response['id']} is NOT a dry run and may be charged; "
                                   "cancel it in the IonQ console. No further jobs were sent.")
            confirmed = True
    rows, pending = [], []
    for row in manifest:
        job_id = read_json(receipts/(Path(row["file"]).stem+"_receipt.json"))["response"]["id"]
        cost = forte.job_cost(job_id)
        if cost is None or "estimated_cost" not in cost:
            pending.append(row["file"])
            continue
        if cost.get("dry_run") is not True:
            raise RuntimeError(f"{row['file']} ({job_id}) is NOT a dry run; check the IonQ console")
        rows.append(dict(file=row["file"], job_id=job_id, zz=row["zz"], shots=row["shots"],
                         **{key: cost["estimated_cost"][key] for key in ("value", "unit")}))
    total = sum(row["value"] for row in rows)
    write_json(run/"costs.json", dict(backend=forte.TARGETS[target]["backend"], jobs=len(manifest),
                                      priced=len(rows), pending=pending, total=total,
                                      unit=rows[0]["unit"] if rows else None, rows=rows))
    print(f"Estimated {total:.2f} {rows[0]['unit'] if rows else ''} for {len(rows)}/{len(manifest)} "
          f"jobs (non-binding); {len(pending)} not priced yet, rerun to retry")


def figures(target):
    """Figures 1-6 from runs/TARGET/results, into runs/TARGET/figures."""
    import analysis
    run = RUNS/target
    require_current(run)
    missing = [f"{method}_p{p}_q{q}_b{b}" + (f"_fold{fold}" if fold != 1 else "")
               for method, p, q, b, fold in plan.jobs()]
    missing = [name for name in missing if not (run/"results"/f"{name}.npz").exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} results missing (e.g. {missing[0]}); run fetch first")
    analysis.analyse_run(run)


def run_all(target):
    """Every step, skipping those already done."""
    if not (RUNS/target/"jobs"/"manifest.json").exists():
        prepare(target)
    submit(target)
    if fetch(target, wait=True):
        figures(target)


def main():
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    load_env()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["run", "ideal", "prepare", "submit", "fetch",
                                            "status", "figures", "estimate"])
    parser.add_argument("target", nargs="?", choices=["emulator", "hardware"])
    parser.add_argument("--wait", action="store_true", help="fetch: poll until every job is back")
    args = parser.parse_args()
    if args.command == "ideal":
        return ideal()
    if args.target is None:
        parser.error(f"{args.command} needs a target: emulator or hardware")
    steps = dict(run=run_all, prepare=prepare, submit=submit, status=status,
                 figures=figures, estimate=estimate)
    if args.command == "fetch":
        fetch(args.target, wait=args.wait)
    else:
        steps[args.command](args.target)


if __name__ == "__main__":
    main()
