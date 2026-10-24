"""Unit tests for offline FasNet temporal aggregation (no live pipeline)."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from analyze_fasnet_temporal_aggregation import (
    assign_nonoverlapping_windows,
    decide_conservative_sustained,
    decide_majority_vote,
    decide_mean_real,
    decide_median_real,
    evaluate_single_frame_session,
    frame_exclusion_reason,
    is_score_usable,
)


def _ts(seconds: float) -> str:
    base = datetime(2026, 10, 20, 12, 0, 0, tzinfo=timezone.utc)
    return (base + timedelta(seconds=seconds)).isoformat()


def _face(
    t: float,
    *,
    native: str = "REAL",
    win: float = 0.96,
    real_avg: float = 0.96,
    spoof_combined: float | None = None,
    face_decision: str = "FACE_OK",
    error: str | None = None,
) -> dict:
    if spoof_combined is None:
        spoof_combined = max(0.0, 1.0 - real_avg)
    return {
        "type": "eval",
        "marked": True,
        "timestamp_utc": _ts(t),
        "face_decision": face_decision,
        "fasnet_native_class": native if face_decision == "FACE_OK" else None,
        "fasnet_winning_score": win if face_decision == "FACE_OK" else None,
        "fasnet_real_score_avg": real_avg if face_decision == "FACE_OK" else None,
        "fasnet_spoof_score_combined": spoof_combined if face_decision == "FACE_OK" else None,
        "fasnet_error": error,
    }


class TemporalAggregationTests(unittest.TestCase):
    def test_no_face_never_usable_or_real_via_mean(self):
        frames = [_face(0.0, face_decision="NO_FACE"), _face(1.0, face_decision="NO_FACE")]
        self.assertTrue(all(not is_score_usable(f) for f in frames))
        d = decide_mean_real(frames, 0.95, 0.85, window_s=3.0)
        self.assertNotEqual(d.state, "REAL")
        self.assertEqual(d.n_valid, 0)

    def test_multiple_faces_excluded(self):
        ev = _face(0.0, face_decision="MULTIPLE_FACES")
        self.assertEqual(frame_exclusion_reason(ev), "MULTIPLE_FACES")
        d = decide_median_real([ev], 0.85, 0.85, window_s=3.0)
        self.assertNotEqual(d.state, "REAL")

    def test_mean_uses_real_class_mass_not_winning_spoof(self):
        frames = [
            _face(0.0, native="SPOOF", win=0.99, real_avg=0.01),
            _face(0.5, native="SPOOF", win=0.98, real_avg=0.02),
            _face(1.0, native="SPOOF", win=0.97, real_avg=0.03),
        ]
        d = decide_mean_real(frames, 0.85, 0.85, window_s=3.0)
        self.assertIsNotNone(d.mean_real)
        self.assertLess(d.mean_real, 0.05)
        self.assertEqual(d.state, "SPOOF")

    def test_mean_real_accepts_when_real_mass_high(self):
        frames = [
            _face(0.0, native="REAL", win=0.97, real_avg=0.97),
            _face(0.5, native="REAL", win=0.96, real_avg=0.96),
            _face(1.0, native="REAL", win=0.95, real_avg=0.95),
        ]
        d = decide_mean_real(frames, 0.95, 0.85, window_s=3.0)
        self.assertEqual(d.state, "REAL")

    def test_majority_vote_tie_with_real_is_uncertain(self):
        frames = [
            _face(0.0, native="REAL", win=0.96, real_avg=0.96),
            _face(0.5, native="SPOOF", win=0.96, real_avg=0.04),
        ]
        d = decide_majority_vote(frames, 0.85, 0.85, window_s=3.0)
        self.assertEqual(d.state, "UNCERTAIN")

    def test_conservative_requires_all_approach_a_real(self):
        frames = [
            _face(0.0, native="REAL", win=0.96, real_avg=0.96),
            _face(0.5, native="REAL", win=0.70, real_avg=0.70),
            _face(1.0, native="REAL", win=0.96, real_avg=0.96),
        ]
        d = decide_conservative_sustained(frames, 0.85, 0.85, window_s=3.0)
        self.assertNotEqual(d.state, "REAL")

    def test_conservative_all_real_passes(self):
        frames = [
            _face(0.0, native="REAL", win=0.96, real_avg=0.96),
            _face(0.5, native="REAL", win=0.97, real_avg=0.97),
        ]
        d = decide_conservative_sustained(frames, 0.85, 0.85, window_s=3.0)
        self.assertEqual(d.state, "REAL")

    def test_empty_window_never_real(self):
        d = decide_mean_real([], 0.85, 0.85, window_s=3.0)
        self.assertEqual(d.state, "ERROR")

    def test_nonoverlapping_windows_by_wall_clock(self):
        evals = [_face(float(i)) for i in range(0, 12)]
        windows = assign_nonoverlapping_windows(evals, 5.0)
        self.assertEqual(len(windows), 3)
        self.assertEqual(len(windows[0]), 5)
        self.assertEqual(len(windows[1]), 5)
        self.assertEqual(len(windows[2]), 2)

    def test_single_frame_baseline_counts(self):
        evals = [
            _face(0.0, native="REAL", win=0.96, real_avg=0.96),
            _face(0.1, face_decision="NO_FACE"),
            _face(0.2, native="SPOOF", win=0.90, real_avg=0.10),
        ]
        decs = evaluate_single_frame_session(evals, 0.85, 0.85)
        self.assertEqual([d.state for d in decs], ["REAL", "ERROR", "SPOOF"])


if __name__ == "__main__":
    unittest.main()
