"""Unit tests for Challenge 1 head-movement session (no camera / no physical claims)."""

from __future__ import annotations

import unittest

import numpy as np

from auth_decision import AuthDecision, AuthOutcome, decide_auth
from challenge1_pose import (
    HeadDirection,
    PoseProxies,
    PoseThresholds,
    direction_progress,
    pose_from_landmarks,
)
from challenge1_session import (
    Challenge1Config,
    Challenge1FailReason,
    Challenge1Phase,
    Challenge1Session,
    decide_challenge1_gate,
    sample_instruction_sequence,
)
from facenet_identity import IdentityMatch
from fasnet_decision_policy import FasNetLivenessState, active_research_policy
from fasnet_liveness import FasNetClass, FasNetResult


def _lm(*, yaw: float = 0.0, pitch: float = 0.4) -> np.ndarray:
    """Synthetic SCRFD-like landmarks for approximate yaw/pitch proxies."""
    iod = 60.0
    eye_y = 80.0
    left_eye = np.array([100.0, eye_y])
    right_eye = np.array([100.0 + iod, eye_y])
    eye_mid = (left_eye + right_eye) * 0.5
    face_h = 70.0
    nose = np.array(
        [
            eye_mid[0] + yaw * iod,
            eye_mid[1] + pitch * face_h,
        ]
    )
    mouth_y = eye_y + face_h
    left_mouth = np.array([110.0, mouth_y])
    right_mouth = np.array([150.0, mouth_y])
    return np.stack([left_eye, right_eye, nose, left_mouth, right_mouth])


class PoseProxyTests(unittest.TestCase):
    def test_pose_from_landmarks_shape(self) -> None:
        p = pose_from_landmarks(_lm(yaw=0.2, pitch=0.5))
        self.assertIsNotNone(p)
        assert p is not None
        self.assertGreater(p.yaw, 0.1)

    def test_four_directions_progress(self) -> None:
        thr = PoseThresholds(yaw_delta=0.12, pitch_delta=0.10)
        cases = [
            (HeadDirection.LOOK_LEFT, 0.20, 0.0),
            (HeadDirection.LOOK_RIGHT, -0.20, 0.0),
            (HeadDirection.LOOK_DOWN, 0.0, 0.20),
            (HeadDirection.LOOK_UP, 0.0, -0.20),
        ]
        for direction, dy, dp in cases:
            prog, reached, wrong = direction_progress(direction, dy, dp, thr)
            self.assertGreaterEqual(prog, 1.0, msg=direction.value)
            self.assertTrue(reached, msg=direction.value)
            self.assertFalse(wrong, msg=direction.value)

    def test_wrong_direction_flagged(self) -> None:
        thr = PoseThresholds(yaw_delta=0.12, pitch_delta=0.10, wrong_ratio=0.85)
        prog, reached, wrong = direction_progress(
            HeadDirection.LOOK_LEFT, -0.20, 0.0, thr
        )
        self.assertTrue(wrong)
        self.assertFalse(reached)


class SequenceTests(unittest.TestCase):
    def test_sample_one_or_two(self) -> None:
        for _ in range(20):
            seq = sample_instruction_sequence()
            self.assertIn(len(seq), (1, 2))
            self.assertTrue(all(isinstance(d, HeadDirection) for d in seq))
            if len(seq) == 2:
                self.assertNotEqual(seq[0], seq[1])


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class SessionFlowTests(unittest.TestCase):
    def _run_direction(
        self,
        direction: HeadDirection,
        *,
        move_yaw: float,
        move_pitch: float,
        clock: FakeClock,
    ) -> Challenge1Session:
        cfg = Challenge1Config(
            thresholds=PoseThresholds(
                yaw_delta=0.12, pitch_delta=0.10, hold_ms=100.0, step_timeout_ms=5000.0
            ),
            baseline_ms=100.0,
        )
        sess = Challenge1Session((direction,), config=cfg, clock=clock)
        # Baseline
        sess.observe_detection(face_count=1, landmarks=_lm(yaw=0.0, pitch=0.4))
        clock.advance(0.15)
        sess.observe_detection(face_count=1, landmarks=_lm(yaw=0.0, pitch=0.4))
        self.assertEqual(sess.phase, Challenge1Phase.INSTRUCT)
        sess.on_instruction_spoken()
        # Move
        for _ in range(5):
            clock.advance(0.05)
            sess.observe_detection(
                face_count=1,
                landmarks=_lm(yaw=move_yaw, pitch=0.4 + move_pitch),
            )
            if sess.phase == Challenge1Phase.HOLD:
                break
        self.assertIn(sess.phase, (Challenge1Phase.HOLD, Challenge1Phase.PASSED_STEP))
        clock.advance(0.15)
        sess.observe_detection(
            face_count=1,
            landmarks=_lm(yaw=move_yaw, pitch=0.4 + move_pitch),
        )
        self.assertEqual(sess.phase, Challenge1Phase.PASSED_STEP)
        return sess

    def test_each_direction(self) -> None:
        moves = {
            HeadDirection.LOOK_LEFT: (0.25, 0.0),
            HeadDirection.LOOK_RIGHT: (-0.25, 0.0),
            HeadDirection.LOOK_DOWN: (0.0, 0.25),
            HeadDirection.LOOK_UP: (0.0, -0.25),
        }
        for direction, (dy, dp) in moves.items():
            clock = FakeClock()
            sess = self._run_direction(direction, move_yaw=dy, move_pitch=dp, clock=clock)
            self.assertTrue(sess.movement_verified, msg=direction.value)

    def test_two_step_ordered(self) -> None:
        clock = FakeClock()
        cfg = Challenge1Config(
            thresholds=PoseThresholds(
                yaw_delta=0.12, pitch_delta=0.10, hold_ms=80.0, step_timeout_ms=5000.0
            ),
            baseline_ms=80.0,
        )
        sess = Challenge1Session(
            (HeadDirection.LOOK_LEFT, HeadDirection.LOOK_UP),
            config=cfg,
            clock=clock,
        )
        # Step 1 LEFT
        sess.observe_detection(face_count=1, landmarks=_lm())
        clock.advance(0.1)
        sess.observe_detection(face_count=1, landmarks=_lm())
        sess.on_instruction_spoken()
        for _ in range(8):
            clock.advance(0.05)
            sess.observe_detection(face_count=1, landmarks=_lm(yaw=0.25))
            if sess.phase == Challenge1Phase.PASSED_STEP:
                break
        self.assertEqual(sess.phase, Challenge1Phase.PASSED_STEP)
        sess.advance_after_step_pass()
        self.assertEqual(sess.phase, Challenge1Phase.BASELINE)
        # Step 2 UP
        clock.advance(0.1)
        sess.observe_detection(face_count=1, landmarks=_lm(yaw=0.25))
        sess.on_instruction_spoken()
        for _ in range(10):
            clock.advance(0.05)
            sess.observe_detection(
                face_count=1, landmarks=_lm(yaw=0.25, pitch=0.4 - 0.25)
            )
            if sess.phase == Challenge1Phase.PASSED_STEP:
                break
        self.assertEqual(sess.phase, Challenge1Phase.PASSED_STEP)
        sess.advance_after_step_pass()
        self.assertEqual(sess.phase, Challenge1Phase.REEVAL)

    def test_wrong_direction_fails(self) -> None:
        clock = FakeClock()
        cfg = Challenge1Config(
            thresholds=PoseThresholds(hold_ms=80.0, step_timeout_ms=5000.0),
            baseline_ms=80.0,
        )
        sess = Challenge1Session((HeadDirection.LOOK_LEFT,), config=cfg, clock=clock)
        sess.observe_detection(face_count=1, landmarks=_lm())
        clock.advance(0.1)
        sess.observe_detection(face_count=1, landmarks=_lm())
        sess.on_instruction_spoken()
        clock.advance(0.05)
        sess.observe_detection(face_count=1, landmarks=_lm(yaw=-0.25))
        self.assertEqual(sess.phase, Challenge1Phase.FAILED)
        self.assertEqual(sess.fail_reason, Challenge1FailReason.WRONG_DIRECTION)

    def test_static_timeout(self) -> None:
        clock = FakeClock()
        cfg = Challenge1Config(
            thresholds=PoseThresholds(hold_ms=80.0, step_timeout_ms=200.0),
            baseline_ms=50.0,
        )
        sess = Challenge1Session((HeadDirection.LOOK_LEFT,), config=cfg, clock=clock)
        sess.observe_detection(face_count=1, landmarks=_lm())
        clock.advance(0.06)
        sess.observe_detection(face_count=1, landmarks=_lm())
        sess.on_instruction_spoken()
        clock.advance(0.25)
        sess.observe_detection(face_count=1, landmarks=_lm())
        self.assertEqual(sess.phase, Challenge1Phase.FAILED)
        self.assertIn(
            sess.fail_reason,
            (Challenge1FailReason.TIMEOUT, Challenge1FailReason.STATIC_NO_MOVE),
        )

    def test_face_lost(self) -> None:
        clock = FakeClock()
        cfg = Challenge1Config(
            thresholds=PoseThresholds(step_timeout_ms=5000.0),
            baseline_ms=50.0,
        )
        sess = Challenge1Session((HeadDirection.LOOK_LEFT,), config=cfg, clock=clock)
        sess.observe_detection(face_count=1, landmarks=_lm())
        clock.advance(0.06)
        sess.observe_detection(face_count=1, landmarks=_lm())
        sess.on_instruction_spoken()
        sess.observe_detection(face_count=0, landmarks=None)
        self.assertEqual(sess.fail_reason, Challenge1FailReason.FACE_LOST)

    def test_multiple_faces(self) -> None:
        clock = FakeClock()
        sess = Challenge1Session((HeadDirection.LOOK_LEFT,), clock=clock)
        sess.observe_detection(face_count=2, landmarks=_lm())
        self.assertEqual(sess.fail_reason, Challenge1FailReason.MULTIPLE_FACES)

    def test_camera_disconnect(self) -> None:
        clock = FakeClock()
        sess = Challenge1Session((HeadDirection.LOOK_LEFT,), clock=clock)
        sess.observe_detection(face_count=1, landmarks=_lm(), camera_ok=False)
        self.assertEqual(sess.fail_reason, Challenge1FailReason.CAMERA_DISCONNECTED)

    def test_stale_frame(self) -> None:
        clock = FakeClock()
        cfg = Challenge1Config(max_frame_age_ms=100.0)
        sess = Challenge1Session((HeadDirection.LOOK_LEFT,), config=cfg, clock=clock)
        sess.observe_detection(face_count=1, landmarks=_lm(), frame_age_ms=500.0)
        self.assertEqual(sess.fail_reason, Challenge1FailReason.STALE_FRAME)

    def test_replay_observation_token(self) -> None:
        clock = FakeClock()
        cfg = Challenge1Config(
            thresholds=PoseThresholds(hold_ms=50.0, step_timeout_ms=5000.0),
            baseline_ms=50.0,
        )
        sess = Challenge1Session((HeadDirection.LOOK_LEFT,), config=cfg, clock=clock)
        tok = "obs-1"
        sess.observe_detection(face_count=1, landmarks=_lm())
        clock.advance(0.06)
        sess.observe_detection(face_count=1, landmarks=_lm())
        sess.on_instruction_spoken()
        for _ in range(6):
            clock.advance(0.05)
            sess.observe_detection(
                face_count=1, landmarks=_lm(yaw=0.25), observation_token=tok
            )
            if sess.phase == Challenge1Phase.PASSED_STEP:
                break
        self.assertEqual(sess.phase, Challenge1Phase.PASSED_STEP)
        # Reusing token after step pass is recorded → fail on next observe with same token
        sess.advance_after_step_pass()
        # New baseline step — reuse old token should fail when observe with used token
        # (token was added on PASSED_STEP). Re-observe with same token:
        sess2 = Challenge1Session((HeadDirection.LOOK_LEFT,), config=cfg, clock=clock)
        sess2._used_tokens.add(tok)
        sess2.observe_detection(
            face_count=1, landmarks=_lm(), observation_token=tok
        )
        self.assertEqual(sess2.fail_reason, Challenge1FailReason.REPLAY_OBSERVATION)

    def test_reeval_spoof_denies(self) -> None:
        clock = FakeClock()
        sess = Challenge1Session((HeadDirection.LOOK_LEFT,), clock=clock)
        sess.phase = Challenge1Phase.REEVAL
        fas = FasNetResult(
            False, 0.9, FasNetClass.SPOOF, 1.0, (0, 0, 10, 10), None
        )
        base = AuthOutcome(AuthDecision.DENY, "spoof", 1, "SPOOF", False, None)
        out = sess.submit_reeval(
            auth_liveness=FasNetLivenessState.SPOOF,
            fasnet=fas,
            identity=IdentityMatch(
                False, None, None, 12.0, {}, 9.0, "UNVALIDATED", 128, 1.0
            ),
            face_count=1,
            base_outcome=base,
        )
        self.assertEqual(out.outcome.decision, AuthDecision.DENY)
        self.assertEqual(sess.fail_reason, Challenge1FailReason.SPOOF)

    def test_reeval_unknown_identity_denies(self) -> None:
        clock = FakeClock()
        sess = Challenge1Session((HeadDirection.LOOK_LEFT,), clock=clock)
        sess.phase = Challenge1Phase.REEVAL
        fas = FasNetResult(True, 0.9, FasNetClass.REAL, 1.0, (0, 0, 10, 10), None)
        ident = IdentityMatch(
            False, None, "Other", 10.0, {"Other": 10.0}, 9.0, "UNVALIDATED", 128, 1.0
        )
        out = sess.submit_reeval(
            auth_liveness=FasNetLivenessState.REAL,
            fasnet=fas,
            identity=ident,
            face_count=1,
            base_outcome=AuthOutcome(
                AuthDecision.DENY, "unknown_identity", 1, "REAL", False, "Other"
            ),
        )
        self.assertEqual(out.outcome.decision, AuthDecision.DENY)

    def test_reeval_uncertain_requests_challenge2(self) -> None:
        clock = FakeClock()
        sess = Challenge1Session((HeadDirection.LOOK_LEFT,), clock=clock)
        sess.phase = Challenge1Phase.REEVAL
        fas = FasNetResult(True, 0.5, FasNetClass.REAL, 1.0, (0, 0, 10, 10), None)
        ident = IdentityMatch(
            True, "Seif", "Seif", 2.0, {"Seif": 2.0}, 9.0, "UNVALIDATED", 128, 1.0,
            display_name="Seif",
        )
        out = sess.submit_reeval(
            auth_liveness=FasNetLivenessState.UNCERTAIN,
            fasnet=fas,
            identity=ident,
            face_count=1,
            base_outcome=AuthOutcome(
                AuthDecision.CHALLENGE_REQUIRED,
                "challenge_1_required",
                1,
                "UNCERTAIN",
                True,
                "Seif",
                challenge_1_required=True,
                challenge_1_status="REQUIRED",
            ),
        )
        self.assertEqual(out.outcome.decision, AuthDecision.CHALLENGE_REQUIRED)
        self.assertEqual(out.outcome.reason, "challenge_2_required")

    def test_reeval_real_recognized_accepts(self) -> None:
        clock = FakeClock()
        sess = Challenge1Session((HeadDirection.LOOK_LEFT,), clock=clock)
        sess.phase = Challenge1Phase.REEVAL
        fas = FasNetResult(True, 0.95, FasNetClass.REAL, 1.0, (0, 0, 10, 10), None)
        ident = IdentityMatch(
            True, "Seif", "Seif", 2.0, {"Seif": 2.0}, 9.0, "UNVALIDATED", 128, 1.0,
            display_name="Seif",
        )
        out = sess.submit_reeval(
            auth_liveness=FasNetLivenessState.REAL,
            fasnet=fas,
            identity=ident,
            face_count=1,
            base_outcome=AuthOutcome(
                AuthDecision.ACCEPT, "real_and_recognized", 1, "REAL", True, "Seif"
            ),
        )
        self.assertEqual(out.outcome.decision, AuthDecision.ACCEPT)
        self.assertEqual(out.outcome.challenge_1_status, "PASSED")

    def test_gate_rewrites_not_implemented(self) -> None:
        o = AuthOutcome(
            AuthDecision.CHALLENGE_REQUIRED,
            "challenge_required_not_implemented",
            1,
            "UNCERTAIN",
            True,
            "Seif",
            challenge_1_required=True,
            challenge_1_status="REQUIRED_NOT_IMPLEMENTED",
        )
        g = decide_challenge1_gate(challenge_enabled=True, outcome=o)
        self.assertEqual(g.reason, "challenge_1_required")
        self.assertEqual(g.challenge_1_status, "REQUIRED")

    def test_auth_decision_uncertain_still_challenge_without_impl_status(self) -> None:
        fas = FasNetResult(True, 0.5, FasNetClass.REAL, 1.0, (0, 0, 1, 1), None)
        ident = IdentityMatch(
            True, "Seif", "Seif", 1.0, {"Seif": 1.0}, 9.0, "UNVALIDATED", 128, 1.0,
            display_name="Seif",
        )
        o = decide_auth(
            face_decision_value="FACE_OK",
            face_count=1,
            fasnet=fas,
            identity=ident,
            fasnet_policy=active_research_policy(),
        )
        self.assertEqual(o.decision, AuthDecision.CHALLENGE_REQUIRED)
        self.assertEqual(o.challenge_1_status, "REQUIRED_NOT_IMPLEMENTED")


if __name__ == "__main__":
    unittest.main()
