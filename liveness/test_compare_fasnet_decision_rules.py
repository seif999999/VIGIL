"""Unit tests for offline FasNet decision-rule comparison (no camera / no JSONL mutation)."""

from __future__ import annotations

import unittest

from compare_fasnet_decision_rules import (
    approach_a_argmax_gated,
    approach_b_combined_mass,
    compare_approaches,
)


def _ev(
    *,
    native: str | None,
    win: float | None,
    real: float | None,
    spoof: float | None,
    auth: str = "DENY",
) -> dict:
    avg = None
    if real is not None and spoof is not None:
        # Split spoof mass evenly for class0/2 when reconstructing avg
        avg = [spoof * 0.5, real, spoof * 0.5]
    return {
        "type": "eval",
        "marked": True,
        "fasnet_native_class": native,
        "fasnet_winning_score": win,
        "fasnet_real_score_avg": real,
        "fasnet_spoof_score_combined": spoof,
        "fasnet_class_scores_avg": avg,
        "auth_decision": auth,
    }


class ApproachATests(unittest.TestCase):
    def test_native_real_above_gate(self) -> None:
        self.assertEqual(
            approach_a_argmax_gated(
                _ev(native="REAL", win=0.92, real=0.92, spoof=0.08), 0.85, 0.85
            ),
            "REAL",
        )

    def test_native_real_below_gate_uncertain(self) -> None:
        self.assertEqual(
            approach_a_argmax_gated(
                _ev(native="REAL", win=0.70, real=0.70, spoof=0.30), 0.85, 0.85
            ),
            "UNCERTAIN",
        )

    def test_missing_native_error(self) -> None:
        self.assertEqual(
            approach_a_argmax_gated(
                _ev(native=None, win=None, real=None, spoof=None), 0.85, 0.85
            ),
            "ERROR",
        )


class ApproachBTests(unittest.TestCase):
    def test_real_mass_gate(self) -> None:
        self.assertEqual(
            approach_b_combined_mass(
                _ev(native="REAL", win=0.92, real=0.92, spoof=0.08), 0.85, 0.85
            ),
            "REAL",
        )

    def test_spoof_combined_gate_without_argmax(self) -> None:
        # Individual class0 might be 0.40, class2 0.45 → sum 0.85; argmax could be either
        # Approach B uses the sum.
        ev = _ev(native="SPOOF", win=0.45, real=0.15, spoof=0.85)
        self.assertEqual(approach_b_combined_mass(ev, 0.85, 0.85), "SPOOF")
        # Approach A needs winning_score >= 0.85; win=0.45 → UNCERTAIN
        self.assertEqual(approach_a_argmax_gated(ev, 0.85, 0.85), "UNCERTAIN")

    def test_both_gates_pass_is_uncertain_no_tiebreak(self) -> None:
        # thr=0.40 allows both masses ≥0.40 when r=0.55, s=0.45
        self.assertEqual(
            approach_b_combined_mass(
                _ev(native="REAL", win=0.55, real=0.55, spoof=0.45), 0.40, 0.40
            ),
            "UNCERTAIN",
        )

    def test_inconsistent_mass_sum_uncertain(self) -> None:
        ev = _ev(native="REAL", win=0.9, real=0.9, spoof=0.9)  # sum=1.8
        self.assertEqual(approach_b_combined_mass(ev, 0.85, 0.85), "UNCERTAIN")

    def test_missing_scores_error(self) -> None:
        self.assertEqual(
            approach_b_combined_mass(
                _ev(native=None, win=None, real=None, spoof=None), 0.85, 0.85
            ),
            "ERROR",
        )


class CompareIntegrationTests(unittest.TestCase):
    def test_empty_sessions_excluded_and_na_denominators(self) -> None:
        empty = {
            "path": "e.jsonl",
            "meta": {"session_id": "PRINT_empty", "label": "PRINT"},
            "evals": [],
            "mark_events": [],
            "end": {},
        }
        data = {
            "path": "d.jsonl",
            "meta": {"session_id": "PRINT_data", "label": "PRINT"},
            "evals": [
                _ev(native="REAL", win=0.91, real=0.91, spoof=0.09, auth="ACCEPT")
            ],
            "mark_events": [],
            "end": {},
        }
        report = compare_approaches([empty, data], candidates=(0.85,))
        self.assertEqual(report["n_empty_excluded"], 1)
        self.assertEqual(report["n_data_bearing"], 1)
        # Validation PRINT empty → N/A false REAL
        b_val = next(
            g
            for g in report["symmetric_grids"]
            if g["approach"].startswith("B") and g["thresholds"]["real_min"] == 0.85
        )
        fr = b_val["validation"]["PRINT"]["false_REAL_all"]
        self.assertIsNone(fr["pct"])
        self.assertEqual(fr["display"], "N/A")


if __name__ == "__main__":
    unittest.main()
