import unittest
from unittest.mock import patch

import numpy as np

from digit_benchmark import EEG, FS, evaluate, keys, scores_for, window
from digit_timing_audit import reconstruct


def stream(codes, truncate=0):
    labels = [code for marker in codes for code, n in ((1, 6), (2, 6), (marker, 12))
              for _ in range(n)] + [1] * 6 + [2] * 6
    labels = labels[:len(labels) - truncate]
    rows = ["onset\ttrial_type", *(f"{sample / 4}\t{code}" for sample, code in enumerate(labels))]
    return "\n".join(rows).encode(), len(labels)


class DigitTests(unittest.TestCase):
    @patch("digit_timing_audit.FS", 2)
    def test_reconstructs_sample_scaling_and_class_counts(self):
        events, n = stream(list(range(3, 8)) * 15)
        cues, audit = reconstruct(events, n)
        self.assertEqual(cues[:2], [(12, 3), (36, 4)])
        self.assertEqual(len(cues), 75)
        self.assertEqual(audit["last_marker_sample"], n - 1)
        self.assertEqual(set(audit["fingers"].values()), {15})

    @patch("digit_timing_audit.FS", 2)
    def test_shortened_run_keeps_only_full_trials(self):
        events, n = stream(list(range(3, 8)) * 2, truncate=13)
        codes = [int(row.split(b"\t")[1]) for row in events.splitlines()[1:]]
        cues, audit = reconstruct(events, n + 2, shortened=True, marker_codes=codes + [0, 0])
        self.assertEqual(len(cues), 9)
        self.assertEqual(audit["edf_trailing_padding_samples"], 2)
        with self.assertRaises(ValueError):
            reconstruct(events, n)
        with self.assertRaises(ValueError):
            reconstruct(events, n + 2, shortened=True, marker_codes=codes + [0, 1])

    @patch("digit_timing_audit.FS", 2)
    def test_rejects_missing_sample(self):
        events, n = stream(list(range(3, 8)) * 15)
        rows = events.splitlines()
        with self.assertRaises(ValueError):
            reconstruct(b"\n".join(rows[:20] + rows[21:]), n)

    @patch("digit_timing_audit.FS", 2)
    def test_rejects_edf_marker_disagreement(self):
        events, n = stream(list(range(3, 8)) * 15)
        codes = [int(row.split(b"\t")[1]) for row in events.splitlines()[1:]]
        reconstruct(events, n, marker_codes=codes)
        codes[12] = 4
        with self.assertRaisesRegex(ValueError, "EDF/TSV marker disagreement"):
            reconstruct(events, n, marker_codes=codes)

    def test_windows_are_half_open_and_checked(self):
        data = np.arange(FS * 3)
        segment = window(data, FS, -1, 0)
        np.testing.assert_array_equal(segment, data[:FS])
        with self.assertRaises(ValueError):
            window(data, 0, -1, 0)

    def test_history_uses_previous_cue_and_resets_per_run(self):
        self.assertEqual(keys("run-02", np.array([4, 0, 2]), "previous_cue"),
                         [("02", -1), ("02", 4), ("02", 0)])

    def test_fixed_label_order_and_balanced_chance(self):
        score, matrix = scores_for([0, 1, 2, 3, 4, 4], [4] * 6)
        self.assertEqual(score, 0.2)
        self.assertEqual(matrix[-1], [0, 0, 0, 0, 2])

    def test_scaler_and_classifier_never_fit_held_out_subject(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        seen = []

        class TrackingScaler(StandardScaler):
            def fit(self, x, y=None, sample_weight=None):
                seen.append(set(x[:, 0]))
                return super().fit(x, y=y, sample_weight=sample_weight)

        data = {
            f"sub-{s:02}/eeg/sub-{s:02}_task-random_run-0{run}": (
                np.arange(5), np.arange(5),
                {key: np.c_[np.full(5, s), np.arange(5)] for key in EEG}, None, None)
            for s in (1, 2, 3) for run in (1, 2)
        }
        with patch("digit_benchmark.StandardScaler", TrackingScaler), patch.object(
            LogisticRegression, "fit", autospec=True, side_effect=LogisticRegression.fit,
        ) as fit:
            scores, _ = evaluate(data)
        self.assertEqual(seen, [{2, 3}] * len(EEG) + [{1, 3}] * len(EEG) + [{1, 2}] * len(EEG))
        self.assertTrue(all(len(call.args[1]) == 20 for call in fit.call_args_list))
        self.assertEqual({r["subject"] for r in scores}, {"sub-01", "sub-02", "sub-03"})


if __name__ == "__main__":
    unittest.main()
