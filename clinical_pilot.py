"""Run the prespecified four-recording pinch/fist feasibility pilot."""

import argparse
import csv
import json
import platform
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import matplotlib
import mne
import numpy as np
from scipy.optimize import minimize
from scipy.signal import periodogram
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

from clinical_data import (
    BASE, HEALTHY, MANIFEST_SHA256, RECORDINGS, ROOT, STROKE, events_path,
    manifest, pilot_paths, verify,
)

FS = 1000
WINDOW = (500, 1500)
LABELS = ("pinch", "fist")
MOTOR = ("FC3", "FC1", "FCZ", "FC2", "FC4", "C3", "C1", "CZ",
         "C2", "C4", "CP3", "CP1", "CPZ", "CP2", "CP4")
POSTERIOR = ("O1", "OZ", "O2", "PO3", "POZ", "PO4", "PO7", "PO8")
EOG = ("HEOGN", "HEOGP", "VEOGN", "VEOGP")
GROUPS = ("motor", "whole_scalp", "posterior", "ocular", "early_visual", "pre_cue")
BUDGETS = (2, 5, 10, 20)
PENALTY = 10.0


def audited_events(root, subject, raw):
    source = mne.events_from_annotations(
        raw, event_id={f"Stimulus/S {code:>2}": code for code in (1, 2, 11, 22)},
        verbose="ERROR",
    )[0][:, [0, 2]]
    if len(source) == 0 or len(source) % 2:
        raise ValueError(f"{subject}: expected paired cue and go markers")
    cues, go = source[::2], source[1::2]
    samples, codes = cues.T
    delays = go[:, 0] - samples
    if (not np.isin(codes, [1, 2]).all() or not np.array_equal(go[:, 1], 11 * codes)
            or np.any(delays < WINDOW[1]) or np.any(np.diff(samples) < 2500)
            or samples[0] < FS or samples[-1] + WINDOW[1] > raw.n_times):
        raise ValueError(f"{subject}: unsafe trial boundaries or invalid source markers")
    with (root / events_path(subject)).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if len(rows) != len(cues):
        raise ValueError(f"{subject}: BIDS/source cue counts differ")
    for row, sample, code in zip(rows, samples, codes, strict=True):
        if (int(row["sample"]) != sample or abs(float(row["onset"]) * FS - sample) > 0.01
                or row["trial_type"] != LABELS[code - 1] or float(row["duration"]) != 1.5):
            raise ValueError(f"{subject}: BIDS/source cue mismatch at sample {sample}")
    return samples, codes - 1, {
        "trials": dict(Counter(row["trial_type"] for row in rows)),
        "bids_value_by_label": {
            label: sorted({int(row["value"]) for row in rows if row["trial_type"] == label})
            for label in LABELS
        },
        "go_delay_ms": [int(delays.min()), int(delays.max())],
        "minimum_cue_spacing_ms": int(np.diff(samples).min()),
        "minimum_previous_go_to_cue_ms": int((samples[1:] - go[:-1, 0]).min()),
    }


def log_power(data, bands):
    frequencies, density = periodogram(data, fs=FS, window="hann", detrend="constant")
    power = [
        density[..., (frequencies >= low) & (frequencies < high)].sum(axis=-1)
        * (frequencies[1] - frequencies[0]) for low, high in bands
    ]
    return np.log(np.maximum(power, 1e-24)).ravel()


def extract(root, subject, stem, nonmotor=(), clip_uv=None):
    raw = mne.io.read_raw_brainvision(root / "sourcedata" / f"{stem}.vhdr", verbose="ERROR")
    raw.rename_channels(str.upper)
    raw.set_channel_types(dict.fromkeys(EOG, "eog"), verbose="ERROR")
    scalp = sorted(name for name, kind in zip(raw.ch_names, raw.get_channel_types())
                   if kind == "eeg" and name != "A1")
    if raw.info["sfreq"] != FS or len(raw.ch_names) != 63 or len(scalp) != 58:
        raise ValueError(f"{subject}: unexpected acquisition layout")
    if not set(MOTOR + POSTERIOR) <= set(scalp):
        raise ValueError(f"{subject}: missing designated scalp channels")
    samples, y, audit = audited_events(root, subject, raw)
    bands = ((8, 12), (12, 30))
    features = {group: [] for group in GROUPS + (("nonmotor",) if nonmotor else ())}
    peak_to_peak = []
    artifact_counts = Counter()

    def robust(data):
        if clip_uv is None:
            return data
        centered = data - np.median(data, axis=-1, keepdims=True)
        limit = clip_uv * 1e-6
        return np.clip(centered, -limit, limit)
    for cue in samples:
        start, stop = cue + WINDOW[0], cue + WINDOW[1]
        data = raw.get_data(picks=scalp + list(EOG), start=start, stop=stop)
        raw_eeg, raw_eog = data[:len(scalp)], data[len(scalp):]
        ranges = np.ptp(raw_eeg, axis=-1) * 1e6
        artifact_counts["scalp_gt_300_uv"] += int(np.any(ranges > 300))
        artifact_counts["scalp_lt_1_uv"] += int(np.any(ranges < 1))
        artifact_counts["bipolar_eog_gt_300_uv"] += int(
            np.any(np.ptp(raw_eog[[1, 3]] - raw_eog[[0, 2]], axis=-1) > 300e-6))
        eeg = robust(raw_eeg)
        eog = raw_eog
        if nonmotor:
            features["nonmotor"].append(log_power(eeg[[scalp.index(n) for n in nonmotor]], bands))
        features["motor"].append(log_power(eeg[[scalp.index(n) for n in MOTOR]], bands))
        features["whole_scalp"].append(log_power(eeg, bands))
        features["posterior"].append(log_power(eeg[[scalp.index(n) for n in POSTERIOR]], bands))
        bipolar = robust(eog[[1, 3]] - eog[[0, 2]])
        features["ocular"].append(np.r_[log_power(bipolar, ((1, 8), (8, 12), (12, 30))),
                                         np.log(np.maximum(np.ptp(bipolar, axis=-1), 1e-12))])
        visual = robust(raw.get_data(picks=POSTERIOR, start=cue - 200, stop=cue + 450))
        reference = visual[:, :200].mean(axis=-1)
        features["early_visual"].append(np.concatenate([
            visual[:, 200 + a:200 + b].mean(axis=-1) - reference
            for a, b in ((100, 200), (200, 300), (300, 450))
        ]))
        pre = robust(raw.get_data(picks=MOTOR, start=cue - 1000, stop=cue))
        features["pre_cue"].append(log_power(pre, bands))
        peak_to_peak.append(float(ranges.max()))
    x = {name: np.asarray(values) for name, values in features.items()}
    if not all(np.isfinite(values).all() for values in x.values()):
        raise ValueError(f"{subject}: nonfinite features")
    audit.update({
        "recording": stem, "channels": len(raw.ch_names), "sample_rate_hz": FS,
        "late_epoch_max_scalp_peak_to_peak_uv": [
            round(float(v), 1) for v in np.percentile(peak_to_peak, [50, 95, 100])
        ],
    })
    if nonmotor:
        audit["artifact_trial_counts"] = dict(artifact_counts)
    return samples, y, x, audit, scalp


def source_fit(recordings, subjects, group):
    x = np.concatenate([recordings[s][2][group] for s in subjects])
    ys = [recordings[s][1] for s in subjects]
    weights = np.concatenate([1.0 / np.bincount(y, minlength=2)[y] for y in ys])
    weights /= weights.mean()
    scaler = StandardScaler().fit(x, sample_weight=weights)
    model = LogisticRegression(C=1 / PENALTY, max_iter=1000, tol=1e-9)
    model.fit(scaler.transform(x), np.concatenate(ys), sample_weight=weights)
    return scaler, np.r_[model.coef_[0], model.intercept_[0]]


def calibrated_weights(x, y, prior):
    def objective(beta):
        score = x @ beta
        change = beta - prior
        loss = (np.logaddexp(0, score) - y * score).sum()
        return (loss + PENALTY / 2 * (change @ change),
                x.T @ (expit(score) - y) + PENALTY * change)

    result = minimize(objective, prior, jac=True, method="L-BFGS-B",
                      options={"ftol": 1e-13, "gtol": 1e-8})
    if not result.success:
        # Use the exact curvature when the line search stalls near the optimum.
        def hessian(beta):
            score = x @ beta
            curvature = expit(score) * expit(-score)
            return (x.T * curvature) @ x + PENALTY * np.eye(x.shape[1])

        result = minimize(objective, prior, jac=True, hess=hessian,
                          method="trust-exact", options={"gtol": 1e-8})
        if not np.isfinite(result.x).all() or np.linalg.norm(objective(result.x)[1], np.inf) > 1e-6:
            raise RuntimeError(result.message)
    return result.x


def evaluate(recordings, healthy=HEALTHY, stroke=STROKE, groups=GROUPS, full_recording=True):
    scores, predictions, splits = [], [], {}

    def score(subject, group, method, train_subjects, indices, weights, x, used=()):
        samples, y = recordings[subject][:2]
        logits = x[indices] @ weights
        pred = (logits > 0).astype(int)
        tag = {"subject": subject, "feature": group, "method": method,
               "labeled_trials": len(used)}
        scores.append({
            **tag, "train_subjects": list(train_subjects), "n_test": len(indices),
            "test_counts": np.bincount(y[indices], minlength=2).tolist(),
            "calibration_samples": [int(samples[i]) for i in used],
            "balanced_accuracy": balanced_accuracy_score(y[indices], pred),
            "confusion_pinch_fist": confusion_matrix(y[indices], pred, labels=[0, 1]).tolist(),
        })
        predictions.extend({
            **tag, "cue_sample": int(samples[i]), "label": LABELS[y[i]],
            "prediction": LABELS[p], "logit_fist": float(logit),
        } for i, p, logit in zip(indices, pred, logits, strict=True))

    for subject in stroke:
        samples, y = recordings[subject][:2]
        midpoint = len(y) // 2
        pool = np.arange(midpoint)
        test = np.arange(midpoint + 1, len(y))
        if min(np.bincount(y[test], minlength=2)) == 0:
            raise ValueError(f"{subject}: fixed test block needs both classes")
        splits[subject] = {
            "calibration_pool": samples[pool].tolist(), "guard": int(samples[midpoint]),
            "fixed_test": samples[test].tolist(),
        }
        for group in groups:
            scaler, prior = source_fit(recordings, healthy, group)
            x = np.c_[scaler.transform(recordings[subject][2][group]), np.ones(len(y))]
            if full_recording:
                score(subject, group, "zero_shot_full", healthy, np.arange(len(y)), prior, x)
            score(subject, group, "zero_shot", healthy, test, prior, x)
            for budget in BUDGETS:
                by_class = [pool[y[pool] == label][:budget] for label in range(2)]
                if any(len(indices) < budget for indices in by_class):
                    if not full_recording:
                        raise ValueError(f"{subject}: insufficient early trials for budget {budget}")
                    continue
                used = np.sort(np.concatenate(by_class))
                for method, anchor in (("source_prior", prior),
                                       ("zero_prior", np.zeros_like(prior))):
                    weights = calibrated_weights(x[used], y[used], anchor)
                    score(subject, group, method, healthy, test, weights, x, used)

    for subject in healthy:
        train = [other for other in healthy if other != subject]
        n = len(recordings[subject][1])
        for group in groups:
            scaler, prior = source_fit(recordings, train, group)
            x = np.c_[scaler.transform(recordings[subject][2][group]), np.ones(n)]
            score(subject, group, "healthy_loso", train, np.arange(n), prior, x)
    return scores, predictions, splits


def plot_pilot(output, scores):
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for method, label in (("source_prior", "Healthy prior"), ("zero_prior", "Zero prior")):
        rows = [r for r in scores if r["subject"] == "sub-1" and r["feature"] == "motor"
                and r["method"] in ("zero_shot", method)]
        if method == "zero_prior":
            rows = [r for r in rows if r["labeled_trials"] > 0]
        axes[0].plot([r["labeled_trials"] for r in rows],
                     [r["balanced_accuracy"] for r in rows], "o-", label=label)
    axes[0].set(title="Stroke sub-1 · later trial block", xlabel="Labeled calibration trials",
                ylabel="Balanced accuracy", xticks=[0, 4, 10, 20, 40])
    axes[0].legend()
    for subject in RECORDINGS:
        method = "zero_shot" if subject in STROKE else "healthy_loso"
        rows = [r for r in scores if r["subject"] == subject and r["method"] == method]
        axes[1].plot(range(len(GROUPS)), [r["balanced_accuracy"] for r in rows], "o",
                     label=subject + (" · stroke" if subject in STROKE else " · healthy"))
    axes[1].set(title="Feature sets · individual test participants",
                xticks=range(len(GROUPS)),
                xticklabels=["Motor", "Scalp", "Posterior", "Ocular", "Early visual", "Pre-cue"])
    axes[1].tick_params(axis="x", labelrotation=30)
    axes[1].legend(fontsize=8, ncol=2)
    for ax in axes:
        ax.axhline(0.5, color="0.55", linestyle="--", linewidth=1)
        ax.set_ylim(0.25, 0.85)
    fig.savefig(output / "clinical_pilot.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=Path("results/clinical_pilot"))
    args = parser.parse_args()
    entries = manifest(args.data)
    paths = pilot_paths()
    for relative in paths:
        if not verify(args.data / relative, entries[relative]):
            raise ValueError(f"Missing or invalid input {relative}. Run clinical_data.py.")
    source_map = json.loads((args.data / "sourcedata/sourcedata_provenance.json").read_bytes())
    for subject, stem in RECORDINGS.items():
        for ext in ("vhdr", "vmrk", "eeg"):
            if not any(r["file"] == f"{stem}.{ext}" and r["subject"] == subject[4:]
                       for r in source_map["files"]):
                raise ValueError(f"Unverified source mapping for {subject}, {stem}.{ext}")
    recordings = {}
    for subject, stem in RECORDINGS.items():
        recordings[subject] = extract(args.data, subject, stem)
        print(subject, recordings[subject][3], flush=True)
    scalp = recordings[next(iter(HEALTHY))][4]
    if any(rec[4] != scalp for rec in recordings.values()):
        raise ValueError("Whole-scalp feature order differs across recordings")
    scores, predictions, splits = evaluate(recordings)
    args.output.mkdir(parents=True, exist_ok=True)
    audit = {
        "snapshot": BASE, "manifest_sha256": MANIFEST_SHA256,
        "inputs": [entries[p] for p in paths],
        "versions": {"python": platform.python_version(), **{
            name: version(name) for name in ("numpy", "scipy", "scikit-learn", "mne", "matplotlib")
        }},
        "labels": LABELS, "window_samples_relative_to_cue": WINDOW,
        "channels": {"motor": MOTOR, "whole_scalp": scalp, "posterior": POSTERIOR, "eog": EOG},
        "healthy": HEALTHY, "stroke": STROKE, "l2_penalty": PENALTY,
        "recordings": {s: rec[3] for s, rec in recordings.items()}, "stroke_splits": splits,
        "scores": scores,
    }
    (args.output / "run.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    with (args.output / "predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(predictions[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(predictions)
    plot_pilot(args.output, scores)


if __name__ == "__main__":
    main()
