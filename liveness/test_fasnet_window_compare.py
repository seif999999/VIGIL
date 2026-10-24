"""Tests for mark-interval window compare (offline research)."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from analyze_fasnet_temporal_aggregation import (
    assign_windows_within_intervals,
    decide_median_real,
    marked_intervals_from_session,
    min_valid_for_window,
)
from analyze_fasnet_window_compare import apply_interval_rules


def _ts(seconds: float) -> str:
    base = datetime(2026, 10, 21, 15, 49, 0, tzinfo=timezone.utc)
    return (base + timedelta(seconds=seconds)).isoformat()


def _eval(t: float, *, real_avg: float = 0.2, native: str = "SPOOF") -> dict:
    return {
        "type": "eval",
        "marked": True,
        "timestamp_utc": _ts(t),
        "face_decision": "FACE_OK",
        "fasnet_native_class": native,
        "fasnet_winning_score": max(real_avg, 1.0 - real_avg),
        "fasnet_real_score_avg": real_avg,
        "fasnet_spoof_score_combined": 1.0 - real_avg,
        "fasnet_error": None,
        "identity_recognized": True,
    }


class WindowCompareTests(unittest.TestCase):
    def test_min_valid_proportional(self):
        self.assertEqual(min_valid_for_window(3.0), 3)
        self.assertEqual(min_valid_for_window(5.0), 5)

    def test_insufficient_valid_is_invalid_never_real(self):
        # Two usable frames < min_valid=3 for a 3s window.
        frames = [_eval(0.0), _eval(0.5)]
        d = decide_median_real(
            frames, 0.90, 0.85, window_s=3.0, min_valid=3, exclude_identity_failures=True
        )
        self.assertEqual(d.state, "INVALID")
        self.assertNotEqual(d.state, "REAL")

    def test_identity_failure_still_counts_for_pad(self):
        """FaceNet unknown_identity must not drop usable FasNet PAD evidence."""
        frames = [
            _eval(0.0, real_avg=0.99, native="REAL"),
            {
                **_eval(0.5, real_avg=0.99, native="REAL"),
                "identity_recognized": False,
            },
            _eval(1.0, real_avg=0.99, native="REAL"),
        ]
        d = decide_median_real(
            frames, 0.90, 0.85, window_s=3.0, min_valid=3, exclude_identity_failures=False
        )
        self.assertEqual(d.n_valid, 3)
        self.assertEqual(d.state, "REAL")

    def test_identity_exclusion_flag_still_available_for_e2e_experiments(self):
        frames = [
            _eval(0.0, real_avg=0.99, native="REAL"),
            {
                **_eval(0.5, real_avg=0.99, native="REAL"),
                "identity_recognized": False,
            },
            _eval(1.0, real_avg=0.99, native="REAL"),
        ]
        d = decide_median_real(
            frames, 0.90, 0.85, window_s=3.0, min_valid=3, exclude_identity_failures=True
        )
        self.assertEqual(d.n_valid, 2)
        self.assertEqual(d.state, "INVALID")

    def test_windows_do_not_cross_mark_intervals(self):
        session = {
            "meta": {"session_id": "X", "label": "PRINT"},
            "mark_events": [
                {"type": "mark_event", "timestamp_utc": _ts(0), "marked": True},
                {"type": "mark_event", "timestamp_utc": _ts(4), "marked": False},
                {"type": "mark_event", "timestamp_utc": _ts(10), "marked": True},
                {"type": "mark_event", "timestamp_utc": _ts(13), "marked": False},
            ],
            "evals": [
                _eval(1.0),
                _eval(2.0),
                _eval(3.0),
                _eval(11.0),
                _eval(12.0),
            ],
        }
        intervals = marked_intervals_from_session(session)
        self.assertEqual(len(intervals), 2)
        self.assertEqual(len(intervals[0].evals), 3)
        self.assertEqual(len(intervals[1].evals), 2)
        # 5s windows within each interval separately — no merged span across gap.
        windows = assign_windows_within_intervals(intervals, 5.0)
        self.assertEqual(len(windows), 2)
        self.assertEqual(len(windows[0]), 3)
        self.assertEqual(len(windows[1]), 2)

    def test_print3_rule_drops_contaminated_intervals(self):
        # Simulate PRINT3: 3 intervals; rule keeps only index 0.
        session = {
            "meta": {
                "session_id": "PRINT_20261021T154852Z",
                "label": "PRINT",
            },
            "mark_events": [
                {"type": "mark_event", "timestamp_utc": _ts(0), "marked": True},
                {"type": "mark_event", "timestamp_utc": _ts(100), "marked": False},
                {"type": "mark_event", "timestamp_utc": _ts(110), "marked": True},
                {"type": "mark_event", "timestamp_utc": _ts(113), "marked": False},
                {"type": "mark_event", "timestamp_utc": _ts(120), "marked": True},
                {"type": "mark_event", "timestamp_utc": _ts(121), "marked": False},
            ],
            "evals": (
                [_eval(float(i)) for i in range(1, 50)]
                + [_eval(111.0 + 0.1 * i, real_avg=0.95, native="REAL") for i in range(10)]
                + [_eval(120.5, real_avg=0.90, native="REAL")]
            ),
        }
        intervals, meta = apply_interval_rules(session)
        self.assertTrue(intervals[0].included)
        self.assertFalse(intervals[1].included)
        self.assertFalse(intervals[2].included)
        self.assertEqual(meta["n_evals_excluded"], 11)
        self.assertEqual(meta["label_status"], "PROVISIONAL_PRINT")


if __name__ == "__main__":
    unittest.main()
