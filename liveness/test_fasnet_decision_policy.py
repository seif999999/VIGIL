"""FasNet three-state confidence policy + auth routing tests (no camera)."""

from __future__ import annotations

import unittest

from auth_decision import AuthDecision, decide_auth
from facenet_identity import IdentityMatch, MATCH_THRESHOLD, MATCH_THRESHOLD_STATUS
from fasnet_decision_policy import (
    RESEARCH_CANDIDATE_REAL_MIN,
    RESEARCH_CANDIDATE_SPOOF_MIN,
    FasNetLivenessState,
    FasNetPolicyMode,
    active_research_policy,
    classify_fasnet_winning_score,
    default_shadow_policy,
    evaluate_fasnet_policy,
)
from fasnet_liveness import FasNetClass, FasNetResult


def _fas(
    real: bool | None,
    score: float | None = 0.9,
    *,
    err: str | None = None,
) -> FasNetResult:
    if err or real is None:
        return FasNetResult(
            None, None, FasNetClass.ERROR, 1.0, (0, 0, 10, 10), err or "x"
        )
    return FasNetResult(
        real,
        score,
        FasNetClass.REAL if real else FasNetClass.SPOOF,
        1.0,
        (0, 0, 10, 10),
        None,
    )


def _id(recognized: bool, name: str = "Seif") -> IdentityMatch:
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
        display_name=name if recognized else None,
        display_name_candidate=name,
    )


class ClassifyFasNetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = default_shadow_policy()

    def test_confident_real_at_boundary(self) -> None:
        st = classify_fasnet_winning_score(
            FasNetClass.REAL, RESEARCH_CANDIDATE_REAL_MIN, self.policy
        )
        self.assertEqual(st, FasNetLivenessState.REAL)

    def test_real_just_below_boundary_uncertain(self) -> None:
        st = classify_fasnet_winning_score(
            FasNetClass.REAL, RESEARCH_CANDIDATE_REAL_MIN - 1e-9, self.policy
        )
        self.assertEqual(st, FasNetLivenessState.UNCERTAIN)

    def test_confident_spoof_at_boundary(self) -> None:
        st = classify_fasnet_winning_score(
            FasNetClass.SPOOF, RESEARCH_CANDIDATE_SPOOF_MIN, self.policy
        )
        self.assertEqual(st, FasNetLivenessState.SPOOF)

    def test_spoof_below_boundary_uncertain(self) -> None:
        st = classify_fasnet_winning_score(
            FasNetClass.SPOOF, 0.50, self.policy
        )
        self.assertEqual(st, FasNetLivenessState.UNCERTAIN)

    def test_missing_score_uncertain(self) -> None:
        st = classify_fasnet_winning_score(FasNetClass.REAL, None, self.policy)
        self.assertEqual(st, FasNetLivenessState.UNCERTAIN)

    def test_error_class(self) -> None:
        st = classify_fasnet_winning_score(FasNetClass.ERROR, 0.99, self.policy)
        self.assertEqual(st, FasNetLivenessState.ERROR)

    def test_shadow_does_not_change_auth_state_vs_native(self) -> None:
        # Low-confidence REAL: native auth stays REAL; shadow UNCERTAIN
        fas = _fas(True, 0.60)
        ev = evaluate_fasnet_policy(fas, default_shadow_policy())
        self.assertEqual(ev.auth_liveness_state, FasNetLivenessState.REAL)
        self.assertEqual(ev.shadow_liveness_state, FasNetLivenessState.UNCERTAIN)
        self.assertEqual(ev.policy_mode, FasNetPolicyMode.SHADOW)
        self.assertTrue(ev.challenge_would_be_required)


class ShadowAuthTests(unittest.TestCase):
    """Default shadow mode: acceptance still follows native argmax."""

    def test_low_conf_real_still_accept_when_recognized(self) -> None:
        o = decide_auth(
            face_decision_value="FACE_OK",
            face_count=1,
            fasnet=_fas(True, 0.55),
            identity=_id(True),
            fasnet_policy=default_shadow_policy(),
        )
        self.assertEqual(o.decision, AuthDecision.ACCEPT)
        self.assertEqual(o.fasnet_auth_state, "REAL")
        self.assertEqual(o.fasnet_shadow_state, "UNCERTAIN")
        self.assertEqual(o.challenge_1_status, "SHADOW_WOULD_REQUIRE_CHALLENGE")
        self.assertFalse(o.challenge_1_required)

    def test_high_conf_real_accept(self) -> None:
        o = decide_auth(
            face_decision_value="FACE_OK",
            face_count=1,
            fasnet=_fas(True, 0.95),
            identity=_id(True),
            fasnet_policy=default_shadow_policy(),
        )
        self.assertEqual(o.decision, AuthDecision.ACCEPT)
        self.assertEqual(o.fasnet_shadow_state, "REAL")
        self.assertEqual(o.challenge_1_status, "NOT_REQUIRED")

    def test_native_spoof_deny_even_low_score(self) -> None:
        o = decide_auth(
            face_decision_value="FACE_OK",
            face_count=1,
            fasnet=_fas(False, 0.40),
            identity=_id(True),
            fasnet_policy=default_shadow_policy(),
        )
        self.assertEqual(o.decision, AuthDecision.DENY)
        self.assertEqual(o.reason, "spoof")
        self.assertEqual(o.fasnet_auth_state, "SPOOF")
        self.assertEqual(o.fasnet_shadow_state, "UNCERTAIN")


class ActiveResearchAuthTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = active_research_policy()

    def test_uncertain_recognized_challenge_required(self) -> None:
        o = decide_auth(
            face_decision_value="FACE_OK",
            face_count=1,
            fasnet=_fas(True, 0.70),
            identity=_id(True),
            fasnet_policy=self.policy,
        )
        self.assertEqual(o.decision, AuthDecision.CHALLENGE_REQUIRED)
        self.assertEqual(o.reason, "challenge_required_not_implemented")
        self.assertTrue(o.challenge_1_required)
        self.assertEqual(o.challenge_1_status, "REQUIRED_NOT_IMPLEMENTED")
        self.assertNeverAccepts(o)

    def test_uncertain_unknown_identity_denies_not_challenge(self) -> None:
        o = decide_auth(
            face_decision_value="FACE_OK",
            face_count=1,
            fasnet=_fas(True, 0.70),
            identity=_id(False),
            fasnet_policy=self.policy,
        )
        self.assertEqual(o.decision, AuthDecision.DENY)
        self.assertEqual(o.reason, "unknown_identity")
        self.assertFalse(o.challenge_1_required)

    def test_confident_spoof_deny(self) -> None:
        o = decide_auth(
            face_decision_value="FACE_OK",
            face_count=1,
            fasnet=_fas(False, 0.90),
            identity=_id(True),
            fasnet_policy=self.policy,
        )
        self.assertEqual(o.decision, AuthDecision.DENY)
        self.assertEqual(o.reason, "spoof")

    def test_confident_real_recognized_accept(self) -> None:
        o = decide_auth(
            face_decision_value="FACE_OK",
            face_count=1,
            fasnet=_fas(True, 0.90),
            identity=_id(True),
            fasnet_policy=self.policy,
        )
        self.assertEqual(o.decision, AuthDecision.ACCEPT)

    def test_no_face(self) -> None:
        o = decide_auth(
            face_decision_value="NO_FACE",
            face_count=0,
            fasnet=None,
            identity=None,
            fasnet_policy=self.policy,
        )
        self.assertEqual(o.decision, AuthDecision.DENY)
        self.assertEqual(o.reason, "no_face")

    def test_multiple_faces(self) -> None:
        o = decide_auth(
            face_decision_value="MULTIPLE_FACES",
            face_count=2,
            fasnet=None,
            identity=None,
            fasnet_policy=self.policy,
        )
        self.assertEqual(o.decision, AuthDecision.DENY)
        self.assertEqual(o.reason, "multiple_faces")

    def test_model_error_uncertain(self) -> None:
        o = decide_auth(
            face_decision_value="FACE_OK",
            face_count=1,
            fasnet=_fas(None, err="boom"),
            identity=_id(True),
            fasnet_policy=self.policy,
        )
        self.assertEqual(o.decision, AuthDecision.UNCERTAIN)
        self.assertTrue(o.reason.startswith("liveness_error"))

    def assertNeverAccepts(self, o) -> None:
        self.assertNotEqual(o.decision, AuthDecision.ACCEPT)


class Mn3PolicyUntouchedTests(unittest.TestCase):
    def test_mn3_provisional_defaults_unchanged(self) -> None:
        from decision_policy import ProvisionalDecisionPolicy

        p = ProvisionalDecisionPolicy()
        self.assertEqual(p.confident_real_min, 0.85)
        self.assertEqual(p.confident_spoof_min, 0.85)


if __name__ == "__main__":
    unittest.main()
