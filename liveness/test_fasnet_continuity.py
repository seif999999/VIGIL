"""Unit tests for offline FasNet continuity helpers."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from analyze_fasnet_continuity import (
    apply_policy,
    face_ok_coverage,
    max_consecutive_face_loss_s,
    _policy_min_coverage,
    _policy_invalidate_multi,
)
from analyze_fasnet_temporal_aggregation import WindowDecision


def _ts(seconds: float) -> str:
    base = datetime(2026, 10, 22, 18, 58, 0, tzinfo=timezone.utc)
    return (base + timedelta(seconds=seconds)).isoformat()


def _ev(t: float, face: str) -> dict:
    return {
        "timestamp_utc": _ts(t),
        "face_decision": face,
        "fasnet_real_score_avg": 0.99 if face == "FACE_OK" else None,
        "fasnet_native_class": "REAL" if face == "FACE_OK" else None,
    }


class ContinuityHelperTests(unittest.TestCase):
    def test_coverage(self):
        frames = [_ev(0, "FACE_OK"), _ev(1, "NO_FACE"), _ev(2, "FACE_OK"), _ev(3, "MULTIPLE_FACES")]
        self.assertAlmostEqual(face_ok_coverage(frames), 0.5)

    def test_max_face_loss_span(self):
        frames = [
            _ev(0, "FACE_OK"),
            _ev(1, "NO_FACE"),
            _ev(2, "NO_FACE"),
            _ev(3, "MULTIPLE_FACES"),
            _ev(4, "FACE_OK"),
        ]
        # loss run from t=1 to t=3 → 2.0s
        self.assertAlmostEqual(max_consecutive_face_loss_s(frames), 2.0)

    def test_policy_forces_invalid_never_real(self):
        base = WindowDecision(
            state="REAL",
            method="median_real",
            window_s=5.0,
            n_frames=4,
            n_valid=2,
            exclusions={},
            mean_real=0.99,
            median_real=0.99,
            time_to_decision_s=4.0,
            t0=_ts(0),
            t1=_ts(4),
        )
        frames = [_ev(0, "NO_FACE"), _ev(1, "NO_FACE"), _ev(2, "FACE_OK"), _ev(3, "FACE_OK")]
        pol = _policy_min_coverage(0.60)
        out = apply_policy(base, frames, pol)
        self.assertEqual(out.state, "INVALID")
        self.assertNotEqual(out.state, "REAL")

    def test_multi_policy(self):
        base = WindowDecision(
            state="SPOOF",
            method="median_real",
            window_s=5.0,
            n_frames=2,
            n_valid=1,
            exclusions={},
            mean_real=0.1,
            median_real=0.1,
            time_to_decision_s=1.0,
            t0=_ts(0),
            t1=_ts(1),
        )
        frames = [_ev(0, "FACE_OK"), _ev(1, "MULTIPLE_FACES")]
        out = apply_policy(base, frames, _policy_invalidate_multi())
        self.assertEqual(out.state, "INVALID")


if __name__ == "__main__":
    unittest.main()
