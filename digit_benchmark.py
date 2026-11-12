"""Run the frozen Damm2026 EEG and confound benchmark."""

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
from scipy.signal import periodogram
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

from digit_data import ROOT, STIMULI, SUBJECTS, annex_entry, check_blob, runs, snapshot_tree, valid
from digit_timing_audit import FINGERS, FS, SNAPSHOT, reconstruct

LABELS = tuple(FINGERS.values())
MOTOR = ("FC3", "FC1", "FCz", "FC2", "FC4", "C3", "C1", "Cz", "C2", "C4",
         "CP3", "CP1", "CPz", "CP2", "CP4")
POSTERIOR = ("P5", "P3", "P1", "Pz", "P2", "P4", "P6", "PO7", "PO3", "POz",
             "PO2", "PO8", "O1", "Oz", "O2")
BANDS = ((8, 12), (12, 30))
WINDOWS = {"baseline": (-3, -1), "precue": (-2, 0), "early": (0.5, 2.5), "late": (3.5, 5.5)}
CUE_BINS = ((0.08, 0.18), (0.18, 0.28), (0.28, 0.38), (0.38, 0.48))
TIME_STARTS = tuple(range(-3, 6))
DB = 10 / np.log(10)
SHORT_RUN = "sub-18/eeg/sub-18_task-random_run-01"
EEG = ("motor_late", "posterior_late", "scalp_late", "motor_early", "posterior_early",
       "precue_scalp", "early_cue")
MODELS = EEG + ("run_position", "previous_cue")


def window(data, sample, start, end):
    lo, hi = (sample + round(s * FS) for s in (start, end))
    if lo < 0 or hi > data.shape[-1] or hi <= lo:
        raise ValueError("Feature window exceeds recording boundary")
    return data[..., lo:hi]


def log_power(data):
    freq, psd = periodogram(data, fs=FS, window="hann", detrend="constant")
    power = [psd[..., (freq >= lo) & (freq < hi)].sum(axis=-1) * (freq[1] - freq[0])
             for lo, hi in BANDS]
    return np.log(np.maximum(np.stack(power, axis=-1), 1e-24))


def extract(root, run, channels=None):
    raw = mne.io.read_raw_edf(root / (run + "_eeg.edf"), preload=False, verbose="ERROR")
    if (raw.info["sfreq"] != FS or len(raw.ch_names) != 65
            or raw.ch_names[-3:] != ["Ref1", "Ref2", "Marker"]):
        raise ValueError(f"Unexpected EDF channel layout for {run}")
    scalp = raw.ch_names[:62]
    if channels is not None and scalp != channels:
        raise ValueError(f"Scalp channel names or order changed for {run}")
    marker = raw.get_data(picks=["Marker"])[0] * 1e6
    rounded = np.rint(marker)
    if not np.isfinite(marker).all() or np.max(np.abs(marker - rounded)) > 0.01:
        raise ValueError(f"Unexpected EDF marker encoding for {run}")
    marker_codes = None if np.ptp(marker) == 0 else rounded.astype(np.int8)
    try:
        cues, audit = reconstruct((root / (run + "_events.tsv")).read_bytes(), int(raw.n_times),
                                  shortened=run == SHORT_RUN, marker_codes=marker_codes)
    except ValueError as err:
        raise ValueError(f"{run}: {err}") from err
    if marker_codes is not None and marker_codes[:audit["first_marker_sample"]].any():
        raise ValueError(f"Nonzero EDF markers before TSV starts for {run}")
    audit["edf_marker_verification"] = "samplewise match" if marker_codes is not None else "flat"
    x = {key: [] for key in EEG}
    erd, timecourse = [], []
    motor, posterior = ([scalp.index(ch) for ch in group] for group in (MOTOR, POSTERIOR))
    for start, _ in cues:
        # Read only the trial span required by the frozen windows.
        trial_start = start - 3 * FS
        trial_end = start + 6 * FS
        if trial_start < audit["first_marker_sample"] or trial_end > raw.n_times:
            raise ValueError(f"Incomplete trial at {run}:{start}")
        data = raw.get_data(picks=scalp, start=trial_start, stop=trial_end)
        cue = 3 * FS
        bands = {key: log_power(window(data, cue, *times)) for key, times in WINDOWS.items()}
        for name, picks in (("motor", motor), ("posterior", posterior)):
            for period in ("early", "late"):
                x[f"{name}_{period}"].append(bands[period][picks].ravel())
        x["scalp_late"].append(bands["late"].ravel())
        x["precue_scalp"].append(bands["precue"].ravel())
        ref = window(data, cue, -0.25, 0).mean(axis=-1)
        x["early_cue"].append(np.concatenate([
            window(data, cue, a, b).mean(axis=-1) - ref for a, b in CUE_BINS]))
        erd.append([[((bands[t] - bands["baseline"])[picks]).mean(axis=0) * DB
                     for t in ("early", "late")] for picks in (motor, posterior)])
        temporal = log_power(data.reshape(len(scalp), len(TIME_STARTS), FS))
        timecourse.append([np.stack((temporal[picks].mean(axis=0),
                                     (temporal - bands["baseline"][:, None])[picks].mean(axis=0)),
                                    axis=-1) * DB for picks in (motor, posterior)])
    samples, codes = np.asarray(cues).T
    x = {name: np.asarray(rows) for name, rows in x.items()}
    if not all(np.isfinite(v).all() for v in x.values()):
        raise ValueError(f"Nonfinite feature in {run}")
    return (samples, codes - 3, x, np.asarray(erd), np.asarray(timecourse)), audit, scalp


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def keys(run, y, model):
    slot = run[-2:]
    if model == "run_position":
        return [(slot, i) for i in range(len(y))]
    return list(zip([slot] * len(y), [-1, *y[:-1]], strict=True))


def scores_for(y, p):
    matrix = confusion_matrix(y, p, labels=range(5))
    counts = matrix.sum(axis=1)
    if (counts == 0).any():
        raise ValueError("Evaluation missing a finger class")
    return float((matrix.diagonal() / counts).mean()), matrix.tolist()


def evaluate(data):
    predictions, scores = [], []
    for condition in ("random", "sequential"):
        names = [run for run in data if f"task-{condition}_" in run]
        for subject in sorted({run.split("/")[0] for run in names}):
            train = [run for run in names if not run.startswith(subject + "/")]
            test = [run for run in names if run.startswith(subject + "/")]
            y = np.concatenate([data[run][1] for run in train])
            weights = []
            for run in train:
                target = data[run][1]
                num_runs = sum(r.split("/")[0] == run.split("/")[0] for r in train)
                weights.extend(1 / (num_runs * np.bincount(target, minlength=5)[target]))
            weights = np.asarray(weights) / np.mean(weights)
            for model in MODELS:
                if model in EEG:
                    x = np.concatenate([data[run][2][model] for run in train])
                    scaler = StandardScaler().fit(x, sample_weight=weights)
                    clf = LogisticRegression(C=0.1, max_iter=2000, tol=1e-8)
                    clf.fit(scaler.transform(x), y, sample_weight=weights)
                    if clf.n_iter_.max() >= clf.max_iter:
                        raise RuntimeError(f"Solver limit for {subject}: {model}")
                else:
                    table = {}
                    for run in train:
                        labels = data[run][1]
                        for k, label in zip(keys(run, labels, model), labels, strict=True):
                            table.setdefault(k, Counter())[label] += 1
                all_y, all_p = [], []
                for run in test:
                    samples, truth, features, _, _ = data[run]
                    if model in EEG:
                        pred = clf.predict(scaler.transform(features[model]))
                    else:
                        pred = np.asarray([
                            min(range(5), key=lambda label: (-table.get(k, {}).get(label, 0), label))
                            for k in keys(run, truth, model)])
                    accuracy, matrix = scores_for(truth, pred)
                    tag = dict(subject=subject, condition=condition, model=model)
                    scores.append(dict(**tag, run=run[-2:], trials=len(truth),
                                       balanced_accuracy=accuracy, confusion=matrix))
                    predictions.extend(dict(**tag, run=run[-2:], trial=i, cue_sample=int(s),
                                            label=LABELS[t], prediction=LABELS[p])
                                       for i, (s, t, p) in enumerate(zip(samples, truth, pred, strict=True)))
                    all_y.extend(truth)
                    all_p.extend(pred)
                accuracy, matrix = scores_for(all_y, all_p)
                scores.append(dict(**tag, run="pooled", trials=len(all_y),
                                   balanced_accuracy=accuracy, confusion=matrix))
    return scores, predictions


def summarize(scores, data):
    rng = np.random.default_rng(2026)
    subjects = sorted({s["subject"] for s in scores})
    draws = rng.integers(len(subjects), size=(10_000, len(subjects)))
    summary, matrices, erd_rows = [], {}, []

    def add(condition, name, values):
        low, high = np.percentile(values[draws].mean(axis=1), [2.5, 97.5])
        summary.append(dict(condition=condition, model_or_contrast=name, mean=values.mean(),
                            low=low, high=high, sd=values.std(ddof=1)))

    lookup = {(s["condition"], s["model"], s["subject"]): s for s in scores if s["run"] == "pooled"}
    conditions = sorted({s["condition"] for s in scores})
    values = {}
    for condition in conditions:
        matrices[condition] = {}
        for model in MODELS:
            rows = [lookup[condition, model, subject] for subject in subjects]
            values[condition, model] = np.asarray([r["balanced_accuracy"] for r in rows])
            add(condition, model, values[condition, model])
            matrices[condition][model] = np.sum([r["confusion"] for r in rows], axis=0).tolist()
        for a, b in (("motor_late", "posterior_late"), ("motor_late", "motor_early"),
                     ("motor_late", "early_cue"), ("motor_early", "posterior_early")):
            add(condition, f"{a} minus {b}", values[condition, a] - values[condition, b])
        for subject in subjects:
            trials = np.concatenate([v[3] for k, v in data.items()
                                     if k.startswith(subject + "/") and f"task-{condition}_" in k])
            for i, group in enumerate(("motor", "posterior")):
                for j, timing in enumerate(("early", "late")):
                    for k, band in enumerate(("mu", "beta")):
                        erd_rows.append(dict(subject=subject, condition=condition, group=group,
                                             window=timing, band=band,
                                             change_db=trials[:, i, j, k].mean()))
    if len(conditions) == 2:
        for model in MODELS:
            add("sequential minus random", model,
                values["sequential", model] - values["random", model])
    return summary, matrices, erd_rows


def timecourse_rows(data):
    for run, (samples, labels, _, _, values) in data.items():
        condition = "random" if "_task-random_" in run else "sequential"
        for sample, label, trial in zip(samples, labels, values, strict=True):
            for group, traces in zip(("motor", "posterior"), trial, strict=True):
                for start, (mu, beta) in zip(TIME_STARTS, traces, strict=True):
                    yield dict(subject=run.split("/")[0], condition=condition, run=run[-2:],
                               cue_sample=int(sample), label=LABELS[label], group=group,
                               start_s=start, mu_db_v2=mu[0], beta_db_v2=beta[0],
                               mu_change_db=mu[1], beta_change_db=beta[1])


def figures(output, summary, matrices, scores, data):
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(1, len(matrices), figsize=(6 * len(matrices), 5),
                             layout="constrained", squeeze=False)
    for ax, (condition, models) in zip(axes[0], matrices.items(), strict=True):
        mat = np.asarray(models["motor_late"])
        rate = mat / mat.sum(axis=1, keepdims=True)
        ax.imshow(rate, vmin=0, vmax=0.5, cmap="Blues")
        for (i, j), count in np.ndenumerate(mat):
            ax.text(j, i, str(count), ha="center", va="center",
                    color="white" if rate[i, j] > 0.3 else "black")
        ax.set(title=f"{condition.capitalize()} · late motor", xlabel="Predicted", ylabel="Observed",
               xticks=range(5), yticks=range(5), xticklabels=LABELS, yticklabels=LABELS)
    fig.savefig(output / "confusion.png", dpi=150)
    plt.close(fig)

    performance_figure(output, summary, matrices, scores)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained", sharey=True)
    subjects = sorted({run.split("/")[0] for run in data})
    traces = np.stack([np.concatenate([v[4] for run, v in data.items()
                                      if run.startswith(s + "/") and "_task-random_" in run]
                                     ).mean(axis=0) for s in subjects])
    for b, (ax, band) in enumerate(zip(axes, ("mu", "beta"), strict=True)):
        for i, group in enumerate(("motor", "posterior")):
            y = traces[:, i, :, b, 1]
            mean = y.mean(axis=0)
            sem = y.std(axis=0, ddof=1) / np.sqrt(len(y))
            x = np.array(TIME_STARTS) + 0.5
            ax.plot(x, mean, "o-", label=group)
            ax.fill_between(x, mean - sem, mean + sem, alpha=0.18)
        ax.set(title=f"{band.capitalize()} · random order", xlabel="Seconds from cue onset",
               ylabel="Change from fixation (dB)")
        ax.axhline(0, color="0.5", linestyle="--")
        ax.axvline(0, color="0.5", linestyle=":")
        ax.legend()
    fig.savefig(output / "erd.png", dpi=150)
    plt.close(fig)


def performance_figure(output, summary, conditions, scores):
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, (ax, individuals) = plt.subplots(1, 2, figsize=(13, 5), layout="constrained",
                                          gridspec_kw={"width_ratios": [1.3, 1]})
    for k, condition in enumerate(conditions):
        rows = [r for r in summary if r["condition"] == condition and r["model_or_contrast"] in EEG]
        x = np.arange(len(rows)) + (k - 0.5) * 0.23
        y, lo, hi = (np.array([r[key] for r in rows]) for key in ("mean", "low", "high"))
        ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="o", capsize=3, label=condition)
        subject_rows = [r for r in scores if r["condition"] == condition and
                        r["model"] == "motor_late" and r["run"] == "pooled"]
        individuals.plot([r["subject"][4:] for r in subject_rows],
                         [r["balanced_accuracy"] for r in subject_rows], "o", label=condition)
    ax.set(xticks=range(len(EEG)), xticklabels=[m.replace("_", " ") for m in EEG],
           ylabel="Balanced accuracy", title="EEG models · mean and 95% bootstrap interval")
    individuals.set(title="Late motor by held-out participant", xlabel="Participant",
                    ylabel="Balanced accuracy")
    ax.tick_params(axis="x", rotation=45)
    for panel in (ax, individuals):
        panel.axhline(0.2, color="0.5", linestyle="--")
        panel.legend()
    fig.savefig(output / "performance.png", dpi=150)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cohort", action="store_true")
    args = parser.parse_args()
    output = args.output or Path("results") / ("digit_cohort" if args.cohort else "digit_pilot")
    tree = snapshot_tree(args.data)
    people = (args.data / "participants.tsv").read_bytes()
    check_blob(people, tree["participants.tsv"])
    identities = tuple(r["participant_id"] for r in csv.DictReader(
        people.decode().splitlines(), delimiter="\t"))
    if identities != SUBJECTS:
        raise ValueError("Unexpected participant roster")
    data, audits, inputs, channels = {}, {}, [], None
    stimuli = [annex_entry(args.data, path, tree) for path in STIMULI]
    if any(not valid(args.data / e["path"], e) for e in stimuli):
        raise ValueError("Stimulus files failed verification. Run digit_data.py --stimuli-only.")
    for run in runs(args.cohort):
        for suffix in ("_eeg.edf", "_events.tsv"):
            entry = annex_entry(args.data, run + suffix, tree)
            if not valid(args.data / entry["path"], entry):
                raise ValueError(f"Missing or invalid annex bytes for {entry['path']}")
            inputs.append(entry)
        data[run], audits[run], channels = extract(args.data, run, channels)
        print(run, len(data[run][1]), flush=True)
    scores, predictions = evaluate(data)
    summary, matrices, erd = summarize(scores, data)
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("scores", scores), ("predictions", predictions),
                       ("summary", summary), ("erd", erd),
                       ("trial_power_timecourse", list(timecourse_rows(data)))):
        write_csv(output / f"{name}.csv", rows)
    (output / "confusion.json").write_text(json.dumps(matrices, indent=2) + "\n", encoding="utf-8")
    configuration = dict(snapshot="ds008446 v1.0.1", commit=SNAPSHOT, inputs=inputs, stimuli=stimuli,
                         participant_blob=tree["participants.tsv"]["sha"], runs=audits,
                         sample_rate_hz=FS, scalp_channels=channels, motor_channels=MOTOR,
                         posterior_channels=POSTERIOR, labels=LABELS, bands_hz=BANDS, windows=WINDOWS,
                         early_cue_bins=CUE_BINS, early_cue_reference=(-0.25, 0),
                         timecourse_bin_starts_s=TIME_STARTS, timecourse_bin_width_s=1,
                         C=0.1, split="leave one participant out within condition",
                         bootstrap_seed=2026, bootstrap_replicates=10_000,
                         python=platform.python_version(),
                         dependencies={p: version(p) for p in
                                       ("mne", "numpy", "scipy", "scikit-learn", "matplotlib")})
    (output / "run.json").write_text(json.dumps(configuration, indent=2) + "\n", encoding="utf-8")
    figures(output, summary, matrices, scores, data)


if __name__ == "__main__":
    main()
