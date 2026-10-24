"""FasNet score semantics + calibration recorder/analyzer tests (no camera)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from analyze_fasnet_calibration import analyze_sessions, load_session
from auth_decision import AuthDecision, AuthOutcome
from auth_pipeline import AuthPipelineResult
from fasnet_calibration_record import CalibrationSessionRecorder
from fasnet_decision_policy import RESEARCH_CANDIDATE_REAL_MIN
from fasnet_liveness import FasNetClass, FasNetResult, scores_from_ensemble_prediction


class ScoreSemanticsTests(unittest.TestCase):
    def test_deepface_decision_math(self) -> None:
        # Summed softmax row (two models); class1 wins
        pred = np.array([0.2, 1.6, 0.2], dtype=np.float64)
        is_real, win, scores, real_avg, spoof_c = scores_from_ensemble_prediction(pred)
        self.assertTrue(is_real)
        self.assertAlmostEqual(win, 0.8)
        self.assertAlmostEqual(scores[1], 0.8)
        self.assertAlmostEqual(real_avg, 0.8)
        self.assertAlmostEqual(spoof_c, 0.2)
        self.assertAlmostEqual(sum(scores), 1.0)

    def test_spoof_when_class0_wins(self) -> None:
        pred = np.array([1.5, 0.3, 0.2], dtype=np.float64)
        is_real, win, scores, real_avg, spoof_c = scores_from_ensemble_prediction(pred)
        self.assertFalse(is_real)
        self.assertAlmostEqual(win, 0.75)
        self.assertAlmostEqual(real_avg, 0.15)
        self.assertAlmostEqual(spoof_c, 0.85)

    def test_class2_also_maps_to_spoof(self) -> None:
        pred = np.array([0.1, 0.4, 1.5], dtype=np.float64)
        is_real, win, _, _, _ = scores_from_ensemble_prediction(pred)
        self.assertFalse(is_real)
        self.assertAlmostEqual(win, 0.75)


def _fake_result(
    *,
    native_real: bool,
    win: float,
    real_avg: float,
    spoof_c: float,
    decision: AuthDecision = AuthDecision.ACCEPT,
    reason: str = "real_and_recognized",
) -> AuthPipelineResult:
    fas = FasNetResult(
        native_real,
        win,
        FasNetClass.REAL if native_real else FasNetClass.SPOOF,
        12.0,
        (10, 10, 40, 40),
        None,
        class_scores_avg=(spoof_c * 0.5, real_avg, spoof_c * 0.5),
        real_score_avg=real_avg,
        spoof_score_combined=spoof_c,
    )
    det = SimpleNamespace(
        face_count=1,
        decision=SimpleNamespace(value="FACE_OK"),
        faces=(),
    )
    outcome = AuthOutcome(
        decision,
        reason,
        1,
        fas.predicted_class.value,
        decision == AuthDecision.ACCEPT,
        "Seif" if decision == AuthDecision.ACCEPT else None,
        fasnet_shadow_state="UNCERTAIN" if win < RESEARCH_CANDIDATE_REAL_MIN else "REAL",
    )
    return AuthPipelineResult(
        detection=det,
        fasnet=fas,
        identity=None,
        outcome=outcome,
        scrfd_ms=5.0,
        fasnet_ms=12.0,
        facenet_ms=0.0,
        total_ms=20.0,
    )


class RecorderTests(unittest.TestCase):
    def test_unmarked_not_saved_marked_saved(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            rec = CalibrationSessionRecorder.create(
                label="REAL",
                camera_index=1,
                out_dir=Path(td),
            )
            r = _fake_result(native_real=True, win=0.9, real_avg=0.9, spoof_c=0.1)
            self.assertFalse(
                rec.record_eval(r, camera_state="frames_ok", preview_fps=20.0, auth_hz=2.0)
            )
            self.assertEqual(rec.marked_records, 0)
            self.assertTrue(rec.toggle_marked())
            self.assertTrue(
                rec.record_eval(r, camera_state="frames_ok", preview_fps=20.0, auth_hz=2.0)
            )
            self.assertEqual(rec.marked_records, 1)
            path = rec.close()
            lines = path.read_text(encoding="utf-8").strip().splitlines()
            types = [json.loads(x)["type"] for x in lines]
            self.assertIn("session_meta", types)
            self.assertIn("mark_event", types)
            self.assertIn("eval", types)
            self.assertIn("session_end", types)
            ev = next(json.loads(x) for x in lines if json.loads(x)["type"] == "eval")
            self.assertEqual(ev["label"], "REAL")
            self.assertEqual(ev["fasnet_winning_score"], 0.9)
            self.assertIsNotNone(ev["fasnet_class_scores_avg"])
            self.assertNotIn("image", ev)
            self.assertNotIn("frame", ev)


class AnalyzerTests(unittest.TestCase):
    def test_insufficient_sessions_reported(self) -> None:
        report = analyze_sessions([])
        self.assertFalse(report["calibration_complete"])
        self.assertEqual(report["n_marked_frames_total"], 0)
        self.assertTrue(report.get("insufficient_independent_validation"))

    def test_empty_sessions_excluded_from_split(self) -> None:
        from analyze_fasnet_calibration import _rate

        na = _rate(0, 0, what="test")
        self.assertIsNone(na["pct"])
        self.assertEqual(na["display"], "N/A")

        empty = {
            "path": "empty.jsonl",
            "meta": {"session_id": "PRINT_empty", "label": "PRINT"},
            "evals": [],
            "mark_events": [],
            "end": {},
        }
        data = {
            "path": "data.jsonl",
            "meta": {"session_id": "PRINT_data", "label": "PRINT"},
            "evals": [
                {
                    "type": "eval",
                    "marked": True,
                    "fasnet_native_class": "REAL",
                    "fasnet_winning_score": 0.92,
                    "auth_decision": "ACCEPT",
                }
            ],
            "mark_events": [{"marked": True}],
            "end": {},
        }
        report = analyze_sessions([empty, data], candidates=(0.85,))
        self.assertEqual(report["n_empty_session_files"], 1)
        self.assertEqual(report["n_data_bearing_sessions"], 1)
        self.assertEqual(
            report["split"]["by_label"]["PRINT"]["selection"], ["PRINT_data"]
        )
        self.assertNotIn("PRINT_empty", report["split"]["by_label"]["PRINT"]["selection"])
        thr = report["candidate_threshold_grids"][0]
        # Validation empty for PRINT → N/A falseREAL, not fake 0%
        pr_val = thr["validation"]["by_label"]["PRINT"]
        self.assertEqual(pr_val["n_marked_evals"], 0)
        self.assertIsNone(pr_val["false_REAL_shadow_rate"]["pct"])
        self.assertEqual(pr_val["false_REAL_shadow_rate"]["display"], "N/A")
        pr_sel = thr["selection"]["by_label"]["PRINT"]
        self.assertEqual(pr_sel["false_REAL_shadow_rate"]["n"], 1)
        self.assertEqual(pr_sel["false_ACCEPT_live_policy_rate"]["n"], 1)

    def test_unmarked_eval_ignored_even_if_present(self) -> None:
        sess = {
            "path": "x.jsonl",
            "meta": {"session_id": "REAL_x", "label": "REAL"},
            "evals": [
                {
                    "type": "eval",
                    "marked": False,
                    "fasnet_native_class": "REAL",
                    "fasnet_winning_score": 0.99,
                    "auth_decision": "ACCEPT",
                },
                {
                    "type": "eval",
                    "marked": True,
                    "fasnet_native_class": "REAL",
                    "fasnet_winning_score": 0.99,
                    "auth_decision": "ACCEPT",
                },
            ],
            "mark_events": [],
            "end": {},
        }
        report = analyze_sessions([sess], candidates=(0.85,))
        self.assertEqual(report["n_marked_frames_total"], 1)

    def test_analysis_with_synthetic_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            # One REAL session with low and high scores
            rec = CalibrationSessionRecorder.create(
                label="REAL", camera_index=0, out_dir=root
            )
            rec.toggle_marked()
            rec.record_eval(
                _fake_result(native_real=True, win=0.60, real_avg=0.60, spoof_c=0.40),
                camera_state="frames_ok",
                preview_fps=10.0,
                auth_hz=1.0,
            )
            rec.record_eval(
                _fake_result(native_real=True, win=0.92, real_avg=0.92, spoof_c=0.08),
                camera_state="frames_ok",
                preview_fps=10.0,
                auth_hz=1.0,
            )
            rec.close()
            # One PRINT session
            rec2 = CalibrationSessionRecorder.create(
                label="PRINT", camera_index=0, out_dir=root
            )
            rec2.toggle_marked()
            rec2.record_eval(
                _fake_result(
                    native_real=False,
                    win=0.88,
                    real_avg=0.12,
                    spoof_c=0.88,
                    decision=AuthDecision.DENY,
                    reason="spoof",
                ),
                camera_state="frames_ok",
                preview_fps=10.0,
                auth_hz=1.0,
            )
            # Attack false-REAL frame
            rec2.record_eval(
                _fake_result(
                    native_real=True,
                    win=0.91,
                    real_avg=0.91,
                    spoof_c=0.09,
                    decision=AuthDecision.ACCEPT,
                    reason="real_and_recognized",
                ),
                camera_state="frames_ok",
                preview_fps=10.0,
                auth_hz=1.0,
            )
            rec2.close()

            sessions = [load_session(p) for p in sorted(root.glob("session_*.jsonl"))]
            report = analyze_sessions(sessions, candidates=(0.85, 0.70))
            self.assertFalse(report["calibration_complete"])
            self.assertTrue(
                any(
                    "SCREEN" in x or ">=2" in x or "Only" in x
                    for x in report["split"]["limitations"]
                )
                or not report["split"]["sufficient_for_held_out_validation"]
            )
            self.assertEqual(report["n_data_bearing_sessions"], 2)
            self.assertEqual(report["n_marked_frames_total"], 4)
            grid = report["candidate_threshold_grids"]
            thr85 = next(c for c in grid if c["thresholds"]["real_min"] == 0.85)
            real_sel = thr85["selection"]["by_label"]["REAL"]
            self.assertEqual(real_sel["n_marked_evals"], 2)
            self.assertIn("selection", thr85)
            self.assertIn("validation", thr85)

    def test_stratified_split_keeps_sessions_intact(self) -> None:
        def _sess(sid: str, label: str, n: int) -> dict:
            return {
                "path": f"{sid}.jsonl",
                "meta": {"session_id": sid, "label": label},
                "evals": [
                    {
                        "type": "eval",
                        "marked": True,
                        "fasnet_native_class": "SPOOF",
                        "fasnet_winning_score": 0.9,
                        "auth_decision": "DENY",
                    }
                    for _ in range(n)
                ],
                "mark_events": [],
                "end": {},
            }

        sessions = [
            _sess("PRINT_a", "PRINT", 10),
            _sess("PRINT_b", "PRINT", 100),
            _sess("REAL_a", "REAL", 5),
            _sess("REAL_b", "REAL", 7),
        ]
        report = analyze_sessions(sessions, candidates=(0.85,), selection_fraction=0.5)
        pr = report["split"]["by_label"]["PRINT"]
        self.assertEqual(pr["selection"], ["PRINT_a"])
        self.assertEqual(pr["validation"], ["PRINT_b"])
        # Frames from PRINT_b must not appear in selection metrics
        thr = report["candidate_threshold_grids"][0]
        self.assertEqual(thr["selection"]["by_label"]["PRINT"]["n_marked_evals"], 10)
        self.assertEqual(thr["validation"]["by_label"]["PRINT"]["n_marked_evals"], 100)


if __name__ == "__main__":
    unittest.main()
