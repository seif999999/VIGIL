"""FaceNet identity calibration recorder tests (no camera)."""

from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from face_bridge import FACE_DETECTOR_PATH, load_face_detector_module
from facenet_identity import (
    MATCH_THRESHOLD,
    MATCH_THRESHOLD_STATUS,
    empty_enrollment_db,
    save_enrollments,
)
from identity_calibration_record import (
    VALID_LABELS,
    IdentityCalibrationRecorder,
    hud_status_line,
)
from live_identity_calibration import create_scrfd_detector, run_session
from live_runtime import InterruptFlag, QUIT_KEY_CODES


def _eval_kwargs(**overrides):
    base = {
        "face_count": 1,
        "face_decision": "FACE_OK",
        "best_distance": 7.5,
        "recognized": True,
        "best_identity_candidate": "Seif",
        "threshold": MATCH_THRESHOLD,
        "threshold_status": MATCH_THRESHOLD_STATUS,
        "facenet_error": None,
        "scrfd_ms": 4.0,
        "facenet_ms": 12.0,
        "camera_state": "frames_ok",
        "preview_fps": 30.0,
        "eval_hz": 8.0,
    }
    base.update(overrides)
    return base


class IdentityStartupSmokeTests(unittest.TestCase):
    """Catch SCRFD / FaceNet API mismatches before a camera session is started."""

    def test_scrfd_module_api_matches_live_friend_auth(self) -> None:
        self.assertTrue(FACE_DETECTOR_PATH.is_file(), f"missing {FACE_DETECTOR_PATH}")
        mod = load_face_detector_module()
        self.assertTrue(hasattr(mod, "FaceDetector"), "expected FaceDetector class")
        self.assertTrue(callable(mod.FaceDetector))
        self.assertFalse(
            hasattr(mod, "create_detector"),
            "create_detector is not part of the SCRFD module API",
        )

    def test_create_scrfd_detector_uses_FaceDetector(self) -> None:
        src = inspect.getsource(create_scrfd_detector)
        self.assertIn("FaceDetector()", src)
        self.assertNotIn(".create_detector(", src)
        # Full session path must not call the non-existent factory either.
        import live_identity_calibration as lic

        session_src = inspect.getsource(lic.run_session)
        self.assertIn("create_scrfd_detector()", session_src)
        self.assertNotIn(".create_detector(", session_src)

    def test_create_scrfd_detector_initializes(self) -> None:
        detector = create_scrfd_detector()
        self.assertTrue(hasattr(detector, "detect"))
        self.assertTrue(callable(detector.detect))

    def test_interrupt_is_set_is_bool_property_not_method(self) -> None:
        flag = InterruptFlag()
        self.assertIsInstance(InterruptFlag.is_set, property)
        self.assertIsInstance(flag.is_set, bool)
        with self.assertRaises(TypeError):
            flag.is_set()  # type: ignore[operator]
        import live_identity_calibration as lic

        src = inspect.getsource(lic.run_session)
        self.assertIn("while not interrupt.is_set:", src)
        self.assertNotIn("interrupt.is_set()", src)


class IdentitySessionLoopSmokeTests(unittest.TestCase):
    """Exercise the preview loop with mocked camera/models (no physical camera)."""

    def test_run_session_enters_preview_loop_and_quits(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            enroll_path = out / "enrollments.json"
            db = empty_enrollment_db()
            db["identities"] = {
                "TestId": {
                    "display_name": "TestId",
                    "embeddings": [[0.1] * 128],
                }
            }
            save_enrollments(db, enroll_path)

            frame = np.random.default_rng(0).integers(
                40, 200, (240, 320, 3), dtype=np.uint8
            )
            face = SimpleNamespace(
                x1=40.0,
                y1=40.0,
                x2=200.0,
                y2=200.0,
                confidence=0.99,
            )
            det = SimpleNamespace(
                face_count=1,
                decision=SimpleNamespace(value="FACE_OK"),
                faces=(face,),
            )

            fake_cap = MagicMock()
            fake_cap.read.return_value = (True, frame)

            fake_detector = MagicMock()
            fake_detector.detect.return_value = det

            match = SimpleNamespace(
                inference_ms=3.0,
                error=None,
                best_distance=7.25,
                recognized=True,
                best_identity_candidate="TestId",
            )
            fake_facenet = MagicMock()
            fake_facenet.backend = "onnx"
            fake_facenet.warm_up.return_value = None
            fake_facenet.match_crop.return_value = match

            keys = iter([-1, -1, ord("q")])

            args = SimpleNamespace(
                enrollment_db=enroll_path,
                facenet_backend=None,
                label="GENUINE",
                camera=0,
                subject_id="smoke_loop",
                out_dir=out / "idcal",
                min_interval_ms=0.0,
            )
            interrupt = InterruptFlag()

            with (
                patch(
                    "live_identity_calibration.create_scrfd_detector",
                    return_value=fake_detector,
                ),
                patch(
                    "live_identity_calibration.FaceNetIdentity",
                    return_value=fake_facenet,
                ),
                patch(
                    "live_identity_calibration._open_camera",
                    return_value=fake_cap,
                ),
                patch("live_identity_calibration.cv2.namedWindow"),
                patch("live_identity_calibration.cv2.imshow"),
                patch("live_identity_calibration.cv2.destroyAllWindows"),
                patch(
                    "live_identity_calibration.poll_key_code",
                    side_effect=lambda _ms=1: next(keys, ord("q")),
                ),
                patch(
                    "live_identity_calibration.preview_window_closed",
                    return_value=False,
                ),
            ):
                code = run_session(args, interrupt)

            self.assertEqual(code, 0)
            self.assertGreaterEqual(fake_cap.read.call_count, 1)
            self.assertGreaterEqual(fake_detector.detect.call_count, 1)
            self.assertGreaterEqual(fake_facenet.match_crop.call_count, 1)
            sessions = list((out / "idcal").glob("session_GENUINE_*.jsonl"))
            self.assertEqual(len(sessions), 1)
            # Loop exited via quit key without needing interrupt.is_set() call.
            self.assertFalse(interrupt.is_set)
            self.assertIn(ord("q"), QUIT_KEY_CODES)

    def test_run_session_stops_when_interrupt_property_set(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            enroll_path = out / "enrollments.json"
            db = empty_enrollment_db()
            db["identities"] = {
                "TestId": {"display_name": "TestId", "embeddings": [[0.2] * 128]}
            }
            save_enrollments(db, enroll_path)

            frame = np.random.default_rng(1).integers(
                40, 200, (240, 320, 3), dtype=np.uint8
            )
            fake_cap = MagicMock()
            fake_cap.read.return_value = (True, frame)
            fake_detector = MagicMock()
            fake_detector.detect.return_value = SimpleNamespace(
                face_count=0,
                decision=SimpleNamespace(value="NO_FACE"),
                faces=(),
            )
            fake_facenet = MagicMock()
            fake_facenet.backend = "onnx"
            fake_facenet.warm_up.return_value = None

            args = SimpleNamespace(
                enrollment_db=enroll_path,
                facenet_backend=None,
                label="IMPOSTOR",
                camera=0,
                subject_id="smoke_intr",
                out_dir=out / "idcal",
                min_interval_ms=0.0,
            )
            interrupt = InterruptFlag()
            reads = {"n": 0}

            def _read():
                reads["n"] += 1
                if reads["n"] >= 2:
                    interrupt.set()
                return True, frame

            fake_cap.read.side_effect = _read

            with (
                patch(
                    "live_identity_calibration.create_scrfd_detector",
                    return_value=fake_detector,
                ),
                patch(
                    "live_identity_calibration.FaceNetIdentity",
                    return_value=fake_facenet,
                ),
                patch(
                    "live_identity_calibration._open_camera",
                    return_value=fake_cap,
                ),
                patch("live_identity_calibration.cv2.namedWindow"),
                patch("live_identity_calibration.cv2.imshow"),
                patch("live_identity_calibration.cv2.destroyAllWindows"),
                patch("live_identity_calibration.poll_key_code", return_value=-1),
                patch(
                    "live_identity_calibration.preview_window_closed",
                    return_value=False,
                ),
            ):
                code = run_session(args, interrupt)

            self.assertEqual(code, 0)
            self.assertTrue(interrupt.is_set)
            self.assertGreaterEqual(reads["n"], 2)


class IdentityRecorderTests(unittest.TestCase):
    def test_labels_are_genuine_and_impostor_only(self) -> None:
        self.assertEqual(VALID_LABELS, ("GENUINE", "IMPOSTOR"))

    def test_unmarked_not_saved_marked_saved(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            rec = IdentityCalibrationRecorder.create(
                label="GENUINE",
                camera_index=0,
                subject_id="genuine_A",
                out_dir=Path(td),
            )
            try:
                self.assertFalse(rec.record_eval(**_eval_kwargs()))
                self.assertEqual(rec.marked_records, 0)
                self.assertEqual(rec.unmarked_seen, 1)

                self.assertTrue(rec.toggle_marked())
                self.assertTrue(rec.record_eval(**_eval_kwargs(best_distance=6.2)))
                self.assertEqual(rec.marked_records, 1)

                self.assertFalse(rec.toggle_marked())
                self.assertFalse(rec.record_eval(**_eval_kwargs(best_distance=8.1)))
                self.assertEqual(rec.marked_records, 1)
                self.assertEqual(rec.unmarked_seen, 2)
            finally:
                rec.close()

            lines = [
                json.loads(x)
                for x in rec.out_path.read_text(encoding="utf-8").splitlines()
                if x.strip()
            ]
            types = [r["type"] for r in lines]
            self.assertEqual(types[0], "session_meta")
            self.assertIn("mark_event", types)
            self.assertEqual(types[-1], "session_end")
            evals = [r for r in lines if r["type"] == "eval"]
            self.assertEqual(len(evals), 1)
            self.assertAlmostEqual(evals[0]["best_distance"], 6.2)
            self.assertTrue(evals[0]["recognized"])
            self.assertEqual(evals[0]["label"], "GENUINE")
            self.assertEqual(evals[0]["subject_id"], "genuine_A")
            self.assertEqual(evals[0]["match_threshold"], MATCH_THRESHOLD)
            self.assertNotIn("embedding", evals[0])
            self.assertNotIn("image", evals[0])
            self.assertNotIn("frame", evals[0])

    def test_impostor_session_keeps_separate_label_and_path(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            g = IdentityCalibrationRecorder.create(
                label="GENUINE",
                camera_index=0,
                subject_id="g1",
                out_dir=out,
            )
            i = IdentityCalibrationRecorder.create(
                label="IMPOSTOR",
                camera_index=0,
                subject_id="i1",
                out_dir=out,
            )
            try:
                g.toggle_marked()
                i.toggle_marked()
                g.record_eval(**_eval_kwargs(best_distance=5.0, recognized=True))
                i.record_eval(
                    **_eval_kwargs(
                        best_distance=14.0,
                        recognized=False,
                        best_identity_candidate="Seif",
                    )
                )
            finally:
                g.close()
                i.close()

            self.assertIn("GENUINE", g.session_id)
            self.assertIn("IMPOSTOR", i.session_id)
            self.assertNotEqual(g.out_path, i.out_path)
            self.assertTrue(g.out_path.name.startswith("session_GENUINE_"))
            self.assertTrue(i.out_path.name.startswith("session_IMPOSTOR_"))

            g_meta = json.loads(g.out_path.read_text(encoding="utf-8").splitlines()[0])
            i_meta = json.loads(i.out_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(g_meta["label"], "GENUINE")
            self.assertEqual(i_meta["label"], "IMPOSTOR")
            self.assertFalse(g_meta["stores_images"])
            self.assertFalse(g_meta["stores_embeddings"])
            self.assertFalse(g_meta["thresholds_activated"])
            self.assertEqual(g_meta["match_threshold"], MATCH_THRESHOLD)

    def test_no_face_and_error_scores_are_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            rec = IdentityCalibrationRecorder.create(
                label="IMPOSTOR",
                camera_index=1,
                subject_id="imp_noface",
                out_dir=Path(td),
            )
            try:
                rec.toggle_marked()
                self.assertTrue(
                    rec.record_eval(
                        **_eval_kwargs(
                            face_count=0,
                            face_decision="NO_FACE",
                            best_distance=None,
                            recognized=None,
                            best_identity_candidate=None,
                            facenet_ms=None,
                        )
                    )
                )
                self.assertTrue(
                    rec.record_eval(
                        **_eval_kwargs(
                            best_distance=None,
                            recognized=False,
                            facenet_error="empty_face_crop",
                            facenet_ms=None,
                        )
                    )
                )
            finally:
                rec.close()

            evals = [
                json.loads(x)
                for x in rec.out_path.read_text(encoding="utf-8").splitlines()
                if x.strip()
            ]
            evals = [r for r in evals if r["type"] == "eval"]
            self.assertEqual(len(evals), 2)
            self.assertIsNone(evals[0]["best_distance"])
            self.assertEqual(evals[0]["face_decision"], "NO_FACE")
            self.assertEqual(evals[1]["facenet_error"], "empty_face_crop")
            self.assertFalse(evals[1]["recognized"])

    def test_rejects_bad_label_and_empty_subject(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):
                IdentityCalibrationRecorder.create(
                    label="REAL",
                    camera_index=0,
                    subject_id="x",
                    out_dir=Path(td),
                )
            with self.assertRaises(ValueError):
                IdentityCalibrationRecorder.create(
                    label="GENUINE",
                    camera_index=0,
                    subject_id="  ",
                    out_dir=Path(td),
                )
            with self.assertRaises(ValueError):
                IdentityCalibrationRecorder.create(
                    label="GENUINE",
                    camera_index=0,
                    subject_id="bad/id",
                    out_dir=Path(td),
                )

    def test_threshold_unchanged_and_hud(self) -> None:
        self.assertEqual(MATCH_THRESHOLD, 9.0)
        self.assertEqual(MATCH_THRESHOLD_STATUS, "UNVALIDATED_FRIEND_THRESHOLD")
        with tempfile.TemporaryDirectory() as td:
            rec = IdentityCalibrationRecorder.create(
                label="GENUINE",
                camera_index=0,
                subject_id="hud",
                out_dir=Path(td),
            )
            try:
                line = hud_status_line(rec)
                self.assertIn("GENUINE", line)
                self.assertIn("PAUSED", line)
                rec.toggle_marked()
                self.assertIn("MARKED", hud_status_line(rec))
            finally:
                rec.close()

    def test_recognized_decision_matches_existing_threshold_rule(self) -> None:
        """Document expected score handling vs MATCH_THRESHOLD (tool does not alter it)."""
        under = 8.999
        over = 9.0
        self.assertTrue(under < MATCH_THRESHOLD)
        self.assertFalse(over < MATCH_THRESHOLD)
        with tempfile.TemporaryDirectory() as td:
            rec = IdentityCalibrationRecorder.create(
                label="GENUINE",
                camera_index=0,
                subject_id="thr",
                out_dir=Path(td),
            )
            try:
                rec.toggle_marked()
                rec.record_eval(
                    **_eval_kwargs(
                        best_distance=under,
                        recognized=under < MATCH_THRESHOLD,
                    )
                )
                rec.record_eval(
                    **_eval_kwargs(
                        best_distance=over,
                        recognized=over < MATCH_THRESHOLD,
                    )
                )
            finally:
                rec.close()
            evals = [
                json.loads(x)
                for x in rec.out_path.read_text(encoding="utf-8").splitlines()
                if x.strip()
            ]
            evals = [r for r in evals if r["type"] == "eval"]
            self.assertTrue(evals[0]["recognized"])
            self.assertFalse(evals[1]["recognized"])


if __name__ == "__main__":
    unittest.main()
