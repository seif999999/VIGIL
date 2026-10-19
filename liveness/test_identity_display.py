"""Identity rename + display-name matching + overlay helper tests (no camera)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from auth_decision import AuthDecision, decide_auth
from facenet_identity import (
    MATCH_THRESHOLD,
    IdentityMatch,
    empty_enrollment_db,
    identity_embedding_count,
    match_identity,
    migrate_identity_label,
    rename_identity,
    save_enrollments,
)
from fasnet_liveness import FasNetClass, FasNetResult
from live_friend_auth import _fasnet_lines, _reason_label, draw_auth


def _emb(seed: float) -> list[float]:
    return [seed + i * 0.001 for i in range(128)]


class IdentityRenameTests(unittest.TestCase):
    def test_migrate_tester_to_named_preserves_count(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "enrollments.json"
            db = empty_enrollment_db()
            db["identities"]["tester"] = {
                "embeddings": [_emb(0.0), _emb(0.2), _emb(0.4)],
                "source_images": ["a.jpg", "b.jpg", "c.jpg"],
                "image_fingerprints": ["aa", "bb", "cc"],
            }
            save_enrollments(db, path)
            summary = migrate_identity_label(
                old_id="tester",
                new_id="Seif",
                display_name="Seif",
                path=path,
            )
            self.assertEqual(summary["embeddings"], 3)
            from facenet_identity import load_enrollments

            loaded = load_enrollments(path)
            self.assertNotIn("tester", loaded["identities"])
            self.assertIn("Seif", loaded["identities"])
            self.assertEqual(identity_embedding_count(loaded, "Seif"), 3)
            self.assertEqual(loaded["identities"]["Seif"]["display_name"], "Seif")
            self.assertEqual(loaded["identities"]["Seif"]["image_fingerprints"], ["aa", "bb", "cc"])

    def test_collision_refuses_merge(self) -> None:
        db = empty_enrollment_db()
        db["identities"]["tester"] = {
            "embeddings": [_emb(0.0)],
            "source_images": [],
            "image_fingerprints": [],
        }
        db["identities"]["Seif"] = {
            "embeddings": [_emb(9.0)],
            "source_images": [],
            "image_fingerprints": [],
        }
        with self.assertRaises(ValueError):
            rename_identity(db, old_id="tester", new_id="Seif", display_name="Seif")

    def test_match_returns_display_name(self) -> None:
        db = empty_enrollment_db()
        target = _emb(1.0)
        db["identities"]["Seif"] = {
            "display_name": "Seif",
            "embeddings": [target],
            "source_images": [],
            "image_fingerprints": [],
        }
        db["identities"]["Omar"] = {
            "display_name": "Omar",
            "embeddings": [_emb(50.0)],
            "source_images": [],
            "image_fingerprints": [],
        }
        m = match_identity(target, db, threshold=MATCH_THRESHOLD)
        self.assertTrue(m.recognized)
        self.assertEqual(m.identity, "Seif")
        self.assertEqual(m.display_name, "Seif")

        far = _emb(80.0)
        m2 = match_identity(far, db, threshold=MATCH_THRESHOLD)
        self.assertFalse(m2.recognized)
        self.assertIsNone(m2.identity)
        self.assertIsNone(m2.display_name)

    def test_accept_uses_display_name(self) -> None:
        fas = FasNetResult(True, 0.9, FasNetClass.REAL, 1.0, (0, 0, 10, 10), None)
        ident = IdentityMatch(
            True,
            "Seif",
            "Seif",
            2.0,
            {"Seif": 2.0},
            MATCH_THRESHOLD,
            "UNVALIDATED_FRIEND_THRESHOLD",
            128,
            1.0,
            None,
            display_name="Seif",
            display_name_candidate="Seif",
        )
        o = decide_auth(
            face_decision_value="FACE_OK",
            face_count=1,
            fasnet=fas,
            identity=ident,
        )
        self.assertEqual(o.decision, AuthDecision.ACCEPT)
        self.assertEqual(o.identity, "Seif")

    def test_unknown_not_labeled_as_person(self) -> None:
        fas = FasNetResult(True, 0.9, FasNetClass.REAL, 1.0, (0, 0, 10, 10), None)
        ident = IdentityMatch(
            False,
            None,
            "Seif",
            20.0,
            {"Seif": 20.0},
            MATCH_THRESHOLD,
            "UNVALIDATED_FRIEND_THRESHOLD",
            128,
            1.0,
            None,
            display_name=None,
            display_name_candidate="Seif",
        )
        o = decide_auth(
            face_decision_value="FACE_OK",
            face_count=1,
            fasnet=fas,
            identity=ident,
        )
        self.assertEqual(o.decision, AuthDecision.DENY)
        self.assertEqual(o.reason, "unknown_identity")
        # outcome.identity may carry candidate for diagnostics; UI must show UNKNOWN
        self.assertFalse(o.recognized)

    def test_fasnet_score_labeled_as_winning_class(self) -> None:
        spoof = FasNetResult(False, 0.81, FasNetClass.SPOOF, 1.0, (0, 0, 10, 10), None)
        lines = _fasnet_lines(spoof)
        joined = " ".join(lines)
        self.assertIn("SPOOF", joined)
        self.assertIn("0.810", joined)
        self.assertIn("uncalibrated", joined)
        self.assertNotIn("REAL class_conf: 0.810", joined)

    def test_reason_labels(self) -> None:
        self.assertEqual(_reason_label("spoof"), "Spoof detected")
        self.assertEqual(_reason_label("no_face"), "No face detected")

    def test_draw_auth_smoke_no_crash(self) -> None:
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        face = SimpleNamespace(x1=100, y1=80, x2=300, y2=320, confidence=0.97)
        det = SimpleNamespace(
            faces=(face,),
            face_count=1,
            decision=SimpleNamespace(value="FACE_OK"),
        )
        fas = FasNetResult(True, 0.88, FasNetClass.REAL, 12.0, (100, 80, 200, 240), None)
        ident = IdentityMatch(
            True,
            "Seif",
            "Seif",
            5.6,
            {"Seif": 5.6, "Omar": 12.0},
            9.0,
            "UNVALIDATED_FRIEND_THRESHOLD",
            128,
            40.0,
            None,
            display_name="Seif",
            display_name_candidate="Seif",
        )
        from auth_decision import AuthOutcome

        outcome = AuthOutcome(
            AuthDecision.ACCEPT, "real_and_recognized", 1, "REAL", True, "Seif"
        )
        result = SimpleNamespace(
            detection=det,
            fasnet=fas,
            identity=ident,
            outcome=outcome,
            scrfd_ms=8.0,
            fasnet_ms=50.0,
            facenet_ms=40.0,
            total_ms=100.0,
        )
        draw_auth(
            img,
            result,
            stale=False,
            mode="continuous",
            busy=False,
            age_ms=120.0,
            camera_state="frames_ok",
            preview_fps=25.0,
            auth_hz=2.0,
        )
        # Unknown person panel
        img2 = np.zeros((480, 640, 3), dtype=np.uint8)
        outcome_u = AuthOutcome(
            AuthDecision.DENY, "unknown_identity", 1, "REAL", False, "Seif"
        )
        ident_u = IdentityMatch(
            False,
            None,
            "Seif",
            15.0,
            {"Seif": 15.0},
            9.0,
            "UNVALIDATED_FRIEND_THRESHOLD",
            128,
            40.0,
            None,
            None,
            "Seif",
        )
        result_u = SimpleNamespace(
            detection=det,
            fasnet=fas,
            identity=ident_u,
            outcome=outcome_u,
            scrfd_ms=8.0,
            fasnet_ms=50.0,
            facenet_ms=40.0,
            total_ms=100.0,
        )
        draw_auth(
            img2,
            result_u,
            stale=False,
            mode="continuous",
            busy=False,
            age_ms=50.0,
            camera_state="frames_ok",
            preview_fps=20.0,
            auth_hz=1.5,
        )
        # Stale / no face / multi covered via reason labels + draw without crash
        draw_auth(
            img2,
            None,
            stale=True,
            mode="continuous",
            busy=True,
            age_ms=None,
            camera_state="unavailable",
            preview_fps=None,
            auth_hz=None,
        )


if __name__ == "__main__":
    unittest.main()
