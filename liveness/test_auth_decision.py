"""Unit tests for secure auth decision (no DeepFace / camera required)."""

from __future__ import annotations

import unittest

from auth_decision import AuthDecision, decide_auth
from facenet_identity import IdentityMatch, MATCH_THRESHOLD, MATCH_THRESHOLD_STATUS, match_identity
from fasnet_liveness import FasNetClass, FasNetResult, bbox_xyxy_to_xywh


def _fas(real: bool | None, *, err: str | None = None) -> FasNetResult:
    if err:
        return FasNetResult(
            None, None, FasNetClass.ERROR, 1.0, (0, 0, 10, 10), err
        )
    if real is None:
        return FasNetResult(None, None, FasNetClass.ERROR, 1.0, (0, 0, 10, 10), "x")
    return FasNetResult(
        real,
        0.9,
        FasNetClass.REAL if real else FasNetClass.SPOOF,
        1.0,
        (0, 0, 10, 10),
        None,
    )


def _id(recognized: bool, name: str = "alice") -> IdentityMatch:
    return IdentityMatch(
        recognized=recognized,
        identity=name if recognized else None,
        best_identity_candidate=name,
        best_distance=1.0 if recognized else 20.0,
        distances={name: 1.0 if recognized else 20.0},
        threshold=MATCH_THRESHOLD,
        threshold_status=MATCH_THRESHOLD_STATUS,
        embedding_dim=128,
        inference_ms=1.0,
        error=None,
    )


class AuthDecisionTests(unittest.TestCase):
    def test_bbox_conversion(self) -> None:
        self.assertEqual(bbox_xyxy_to_xywh((10.2, 20.8, 50.2, 80.8)), (10, 21, 40, 60))

    def test_accept_real_recognized(self) -> None:
        o = decide_auth(
            face_decision_value="FACE_OK",
            face_count=1,
            fasnet=_fas(True),
            identity=_id(True, "alice"),
        )
        self.assertEqual(o.decision, AuthDecision.ACCEPT)
        self.assertEqual(o.reason, "real_and_recognized")
        self.assertEqual(o.identity, "alice")

    def test_deny_spoof(self) -> None:
        o = decide_auth(
            face_decision_value="FACE_OK",
            face_count=1,
            fasnet=_fas(False),
            identity=_id(True),
        )
        self.assertEqual(o.decision, AuthDecision.DENY)
        self.assertEqual(o.reason, "spoof")

    def test_deny_unknown(self) -> None:
        o = decide_auth(
            face_decision_value="FACE_OK",
            face_count=1,
            fasnet=_fas(True),
            identity=_id(False),
        )
        self.assertEqual(o.decision, AuthDecision.DENY)
        self.assertEqual(o.reason, "unknown_identity")

    def test_deny_no_face(self) -> None:
        o = decide_auth(
            face_decision_value="NO_FACE",
            face_count=0,
            fasnet=None,
            identity=None,
        )
        self.assertEqual(o.decision, AuthDecision.DENY)
        self.assertEqual(o.reason, "no_face")

    def test_deny_multiple(self) -> None:
        o = decide_auth(
            face_decision_value="MULTIPLE_FACES",
            face_count=2,
            fasnet=_fas(True),
            identity=_id(True),
        )
        self.assertEqual(o.decision, AuthDecision.DENY)
        self.assertEqual(o.reason, "multiple_faces")

    def test_uncertain_liveness_error(self) -> None:
        o = decide_auth(
            face_decision_value="FACE_OK",
            face_count=1,
            fasnet=_fas(None, err="boom"),
            identity=None,
        )
        self.assertEqual(o.decision, AuthDecision.UNCERTAIN)
        self.assertIn("liveness_error", o.reason)

    def test_match_threshold_not_raised(self) -> None:
        emb = [0.0] * 128
        other = [0.1] * 128
        db = {
            "identities": {"bob": {"embeddings": [other]}},
        }
        # distance = sqrt(128*0.1^2) ≈ 1.13 < 9.0 → recognized
        m = match_identity(emb, db, threshold=MATCH_THRESHOLD)
        self.assertTrue(m.recognized)
        # Far embedding
        far = [10.0] * 128
        m2 = match_identity(far, db, threshold=MATCH_THRESHOLD)
        self.assertFalse(m2.recognized)
        self.assertEqual(m2.threshold, 9.0)


if __name__ == "__main__":
    unittest.main()
