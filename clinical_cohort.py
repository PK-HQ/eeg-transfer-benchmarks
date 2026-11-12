"""Run the frozen baseline cohort protocol in cohort_design.md."""

import argparse
import csv
import json
import platform
from importlib.metadata import version
from pathlib import Path

import matplotlib
import numpy as np

from clinical_data import (
    BASE, COHORT_HEALTHY, COHORT_STROKE, MANIFEST_SHA256, ROOT, manifest, pilot_paths, verify,
)
from clinical_pilot import evaluate, extract

NONMOTOR = ("FP1", "FPZ", "FP2", "AF3", "AF4", "P3", "PZ", "P4",
            "PO3", "POZ", "PO4", "O1", "OZ", "O2", "PO8")
GROUPS = ("motor", "nonmotor", "whole_scalp", "pre_cue", "early_visual", "ocular",
          "visual_ocular", "motor_visual_ocular")
RECORDINGS = COHORT_HEALTHY | COHORT_STROKE


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def analyze(rows):
    scores = {(r["artifact"], r["subject"], r["feature"], r["method"],
               r["labeled_trials"]): r["balanced_accuracy"] for r in rows}
    patients = list(COHORT_STROKE)
    draws = np.random.default_rng(2026).integers(len(patients), size=(10_000, len(patients)))

    def values(artifact, group, method="zero_shot", n=0):
        return np.array([scores[artifact, s, group, method, n] for s in patients])

    summaries = []
    def record(artifact, contrast, group, n, data):
        lo, hi = np.percentile(data[draws].mean(axis=1), [2.5, 97.5])
        summaries.append(dict(artifact=artifact, contrast=contrast, feature=group,
                              labeled_trials=n, mean=data.mean(), low=lo, high=hi))

    for artifact in ("raw", "clipped"):
        for group in GROUPS:
            zero = values(artifact, group)
            record(artifact, "zero_shot", group, 0, zero)
            for n in (4, 10, 20, 40):
                healthy = values(artifact, group, "source_prior", n)
                other = values(artifact, group, "zero_prior", n)
                record(artifact, "source_prior", group, n, healthy)
                record(artifact, "zero_prior", group, n, other)
                record(artifact, "source_prior_minus_zero_shot", group, n, healthy - zero)
                record(artifact, "source_prior_minus_zero_prior", group, n, healthy - other)
        for method, n in [("zero_shot", 0), *(("source_prior", n) for n in (4, 10, 20, 40))]:
            for a, b in (("motor", "nonmotor"), ("motor", "early_visual"),
                         ("motor_visual_ocular", "visual_ocular")):
                record(artifact, f"{method}:{a}_minus_{b}", a, n,
                       values(artifact, a, method, n) - values(artifact, b, method, n))
    for group in GROUPS:
        for method, n in [("zero_shot", 0), *(("source_prior", n) for n in (4, 10, 20, 40))]:
            record("paired", f"clipped_minus_raw:{method}", group, n,
                   values("clipped", group, method, n) - values("raw", group, method, n))
    return summaries


def plot(output, summaries):
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    def series(ax, rows, x, label, linestyle="-"):
        y = np.array([r["mean"] for r in rows])
        lo, hi = (np.array([r[key] for r in rows]) for key in ("low", "high"))
        ax.errorbar(x, y, yerr=[y - lo, hi - y], marker="o", capsize=3,
                    label=label, linestyle=linestyle)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
    for ax, artifact in zip(axes, ("raw", "clipped"), strict=True):
        rows = [r for r in summaries if r["artifact"] == artifact and r["feature"] == "motor"]
        for method, label in (("source_prior", "Healthy prior"), ("zero_prior", "Zero prior")):
            selected = [r for r in rows if r["contrast"] == method
                        or (method == "source_prior" and r["contrast"] == "zero_shot")]
            series(ax, selected, [r["labeled_trials"] for r in selected], label)
        ax.set(title=artifact.capitalize(), xlabel="Labeled patient trials", xticks=[0, 4, 10, 20, 40])
        ax.axhline(0.5, color="0.5", linestyle="--")
    axes[0].set_ylabel("Patient mean balanced accuracy (95% bootstrap interval)")
    axes[1].legend()
    fig.savefig(output / "calibration.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    groups = GROUPS[:6]
    x = np.arange(len(groups))
    for artifact, offset in (("raw", -0.10), ("clipped", 0.10)):
        rows = {r["feature"]: r for r in summaries
                if r["artifact"] == artifact and r["contrast"] == "zero_shot"}
        series(ax, [rows[g] for g in groups], x + offset, artifact.capitalize(), "none")
    ax.set(title="Healthy-to-patient zero-shot transfer", ylabel="Balanced accuracy",
           xticks=x, xticklabels=[g.replace("_", " ") for g in groups])
    ax.axhline(0.5, color="0.5", linestyle="--")
    ax.legend()
    fig.savefig(output / "features.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=Path("results/clinical_cohort"))
    args = parser.parse_args()
    entries = manifest(args.data)
    paths = pilot_paths(RECORDINGS)
    for relative in paths:
        if not verify(args.data / relative, entries[relative]):
            raise ValueError(f"Missing or invalid input {relative}. Run clinical_data.py --cohort.")
    provenance = json.loads((args.data / "sourcedata/sourcedata_provenance.json").read_bytes())
    source = {r["file"]: r for r in provenance["files"]}
    for subject, stem in RECORDINGS.items():
        for ext in ("vhdr", "vmrk", "eeg"):
            filename = f"{stem}.{ext}"
            entry = entries[f"sourcedata/{filename}"]
            if (source[filename]["subject"] != subject[4:]
                    or source[filename]["bytes"] != entry["size"]
                    or entry["checksum_algorithm"] != "sha256"
                    or source[filename]["sha256"] != entry["checksum"]):
                raise ValueError(f"Unexpected provenance for {subject}: {filename}")

    recordings, audit, rows, predictions, splits = {}, {}, [], [], {}
    for artifact, limit in (("raw", None), ("clipped", 150)):
        for subject, stem in RECORDINGS.items():
            print(artifact, subject, stem, flush=True)
            rec = extract(args.data, subject, stem, NONMOTOR, limit)
            x = rec[2]
            x["visual_ocular"] = np.c_[x["early_visual"], x["ocular"]]
            x["motor_visual_ocular"] = np.c_[x["motor"], x["visual_ocular"]]
            recordings[subject] = rec
            audit[subject] = rec[3]
        scalp = recordings[next(iter(RECORDINGS))][4]
        if any(rec[4] != scalp for rec in recordings.values()):
            raise ValueError("Whole-scalp channel order differs across recordings")
        scores, trials, partition = evaluate(recordings, COHORT_HEALTHY, COHORT_STROKE,
                                             GROUPS, full_recording=False)
        if splits and splits != partition:
            raise ValueError("Artifact handling changed the partition")
        splits = partition
        for s in scores:
            (pp, pf), (fp, ff) = s["confusion_pinch_fist"]
            rows.append(dict(artifact=artifact, **s, pinch_recall=pp / (pp + pf),
                             fist_recall=ff / (fp + ff), fist_fraction=(pf + ff) / s["n_test"]))
        predictions.extend(dict(artifact=artifact, **p) for p in trials)
    summaries = analyze(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "cohort_results.csv", rows)
    write_csv(args.output / "predictions.csv", predictions)
    write_csv(args.output / "paired_summary.csv", summaries)
    write_csv(args.output / "artifact_counts.csv", [
        dict(subject=s, recording=r["recording"], trials=sum(r["trials"].values()),
             **r["artifact_trial_counts"], scalp_max_uv=r["late_epoch_max_scalp_peak_to_peak_uv"][-1])
        for s, r in audit.items()
    ])
    run = dict(snapshot=BASE, manifest_sha256=MANIFEST_SHA256, inputs=[entries[p] for p in paths],
               python=platform.python_version(), versions={name: version(name) for name in
               ("numpy", "scipy", "scikit-learn", "mne", "matplotlib")}, recordings=audit,
               healthy=COHORT_HEALTHY, stroke=COHORT_STROKE, stroke_splits=splits)
    (args.output / "run.json").write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    plot(args.output, summaries)


if __name__ == "__main__":
    main()
