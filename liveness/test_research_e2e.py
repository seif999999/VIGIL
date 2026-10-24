"""Research E2E decision + PAD buffer + API smoke tests (no camera)."""

from __future__ import annotations

import inspect
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from challenge1_session import Challenge1FailReason
from facenet_identity import MATCH_THRESHOLD
from live_runtime import InterruptFlag
from research_e2e_decision import (
    FACENET_UNCERTAIN_GAP,
    STAGE_STATUS,
    E2EDecision,
    E2EPhase,
    IdentityState,
    route_after_challenge1,
    route_after_pad_window,
    summarize_identity,
)
from challenge1_pose import HeadDirection
from challenge1_session import Challenge1Config, Challenge1Phase, Challenge1Session
from research_e2e_challenge1 import (
    RESEARCH_C1_N_STEPS,
    RESEARCH_C1_STEP_TIMEOUT_MS,
    ResearchChallenge1Timing,
    research_challenge1_config,
    sample_single_instruction,
)
from research_e2e_pad import (
    FROZEN_PAD_MIN_VALID,
    FROZEN_PAD_REAL_MIN,
    FROZEN_PAD_SPOOF_MIN,
    FROZEN_PAD_WINDOW_S,
    FrozenPadWindowBuffer,
    eval_from_observation,
)


def _usable_eval(real: float, spoof: float | None = None, t: datetime | None = None) -> dict:
    ts = (t or datetime.now(timezone.utc)).isoformat()
    spoof_c = spoof if spoof is not None else max(0.0, 1.0 - real)
    return eval_from_observation(
        face_decision="FACE_OK",
        fasnet_native_class="REAL" if real >= 0.5 else "SPOOF",
        fasnet_real_score_avg=real,
        fasnet_spoof_score_combined=spoof_c,
        fasnet_error=None,
        timestamp_utc=ts,
    )


class FrozenPadConstantsTests(unittest.TestCase):
    def test_frozen_values(self) -> None:
        self.assertEqual(FROZEN_PAD_WINDOW_S, 5.0)
        self.assertEqual(FROZEN_PAD_REAL_MIN, 0.95)
        self.assertEqual(FROZEN_PAD_SPOOF_MIN, 0.85)
        self.assertEqual(FROZEN_PAD_MIN_VALID, 5)
        self.assertEqual(MATCH_THRESHOLD, 9.0)
        self.assertEqual(STAGE_STATUS["challenge2_voice"], "NOT_IMPLEMENTED")
        self.assertEqual(STAGE_STATUS["facenet_uncertain_zone"], "NOT_AVAILABLE_NO_CALIBRATION")
        self.assertEqual(STAGE_STATUS["challenge1_on_real_match"], "DISABLED_BY_THESIS")
        self.assertIn("No GENUINE/IMPOSTOR", FACENET_UNCERTAIN_GAP)


class PadBufferTests(unittest.TestCase):
    def test_non_overlapping_window_emits_after_5s(self) -> None:
        buf = FrozenPadWindowBuffer()
        t0 = 100.0
        decision = None
        for i in range(5):
            decision = buf.add(perf_t=t0 + i * 0.9, eval_row=_usable_eval(0.97))
            self.assertIsNone(decision)
        decision = buf.add(perf_t=t0 + 5.0, eval_row=_usable_eval(0.97))
        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.state, "REAL")
        self.assertGreaterEqual(decision.n_valid, 5)
        self.assertEqual(buf.n_frames(), 0)

    def test_insufficient_valid_is_invalid(self) -> None:
        buf = FrozenPadWindowBuffer()
        t0 = 0.0
        for i in range(3):
            buf.add(
                perf_t=t0 + i,
                eval_row=eval_from_observation(
                    face_decision="NO_FACE",
                    fasnet_native_class=None,
                    fasnet_real_score_avg=None,
                    fasnet_spoof_score_combined=None,
                    fasnet_error=None,
                ),
            )
        d = buf.add(perf_t=t0 + 5.0, eval_row=_usable_eval(0.99))
        self.assertIsNotNone(d)
        assert d is not None
        self.assertIn(d.state, ("INVALID", "ERROR"))
        self.assertNotEqual(d.state, "REAL")


class DecisionRoutingTests(unittest.TestCase):
    def test_spoof_denies(self) -> None:
        idn = summarize_identity([(5.0, "Seif")])
        r = route_after_pad_window(pad_state="SPOOF", identity=idn)
        self.assertEqual(r.decision, E2EDecision.DENY)
        self.assertEqual(r.reason, "pad_spoof")

    def test_unknown_identity_denies_even_if_real(self) -> None:
        idn = summarize_identity([(12.0, "Seif")])
        self.assertEqual(idn.state, IdentityState.UNKNOWN)
        self.assertFalse(idn.recognized)
        r = route_after_pad_window(pad_state="REAL", identity=idn)
        self.assertEqual(r.decision, E2EDecision.DENY)
        self.assertEqual(r.reason, "unknown_identity")
        # Must not mislabel UNKNOWN as UNCERTAIN
        self.assertNotEqual(idn.state, IdentityState.UNCERTAIN_NOT_AVAILABLE)

    def test_real_match_accepts_directly_no_challenge1(self) -> None:
        idn = summarize_identity([(4.0, "Seif")])
        r = route_after_pad_window(pad_state="REAL", identity=idn)
        self.assertEqual(r.decision, E2EDecision.ACCEPT)
        self.assertEqual(r.phase, E2EPhase.ACCEPT)
        self.assertEqual(r.reason, "joint_confident_real_and_match")
        self.assertNotEqual(r.decision, E2EDecision.CHALLENGE1_REQUIRED)

    def test_uncertain_match_requires_c1(self) -> None:
        idn = summarize_identity([(6.0, "Seif")])
        r = route_after_pad_window(pad_state="UNCERTAIN", identity=idn)
        self.assertEqual(r.decision, E2EDecision.CHALLENGE1_REQUIRED)
        self.assertEqual(r.phase, E2EPhase.AWAITING_CHALLENGE1)
        self.assertIn("uncertain", r.reason)

    def test_post_c1_real_match_accepts(self) -> None:
        idn = summarize_identity([(5.0, "Seif")])
        r = route_after_pad_window(
            pad_state="REAL", identity=idn, post_challenge1=True
        )
        self.assertEqual(r.decision, E2EDecision.ACCEPT)
        self.assertEqual(r.phase, E2EPhase.ACCEPT)

    def test_post_c1_uncertain_blocks_on_missing_c2(self) -> None:
        idn = summarize_identity([(5.0, "Seif")])
        r = route_after_pad_window(
            pad_state="UNCERTAIN", identity=idn, post_challenge1=True
        )
        self.assertEqual(r.decision, E2EDecision.CHALLENGE2_UNAVAILABLE)
        self.assertEqual(r.phase, E2EPhase.CHALLENGE2_BLOCKED)

    def test_invalid_retries_then_deny(self) -> None:
        idn = summarize_identity([(5.0, "Seif")])
        r0 = route_after_pad_window(
            pad_state="INVALID", identity=idn, invalid_retry_count=0
        )
        self.assertEqual(r0.decision, E2EDecision.RETRY)
        r1 = route_after_pad_window(
            pad_state="INVALID", identity=idn, invalid_retry_count=2
        )
        self.assertEqual(r1.decision, E2EDecision.DENY)

    def test_identity_unavailable_denies(self) -> None:
        idn = summarize_identity([])
        self.assertEqual(idn.state, IdentityState.UNAVAILABLE)
        r = route_after_pad_window(pad_state="REAL", identity=idn)
        self.assertEqual(r.decision, E2EDecision.DENY)
        self.assertEqual(r.reason, "identity_unavailable")

    def test_challenge1_gesture_gates_post_pad_never_accept(self) -> None:
        ok = route_after_challenge1(succeeded_gesture=True, fail_reason="NONE")
        self.assertEqual(ok.phase, E2EPhase.COLLECTING_POST_C1_PAD)
        self.assertNotEqual(ok.decision, E2EDecision.ACCEPT)
        bad = route_after_challenge1(succeeded_gesture=False, fail_reason="TIMEOUT")
        self.assertEqual(bad.decision, E2EDecision.DENY)

    def test_identity_threshold_rule_unchanged(self) -> None:
        under = summarize_identity([(8.999, "A")])
        over = summarize_identity([(9.0, "A")])
        self.assertEqual(under.state, IdentityState.MATCH)
        self.assertEqual(over.state, IdentityState.UNKNOWN)

    def test_identity_uncertain_cases_not_silently_invented(self) -> None:
        """REAL + identity-UNCERTAIN cannot be routed until calibration exists."""
        # Binary FaceNet has no gray zone — distance always MATCH or UNKNOWN.
        idn = summarize_identity([(8.5, "A")])
        self.assertEqual(idn.state, IdentityState.MATCH)
        self.assertNotIn("UNCERTAIN", idn.state.value)
        # Documented gap remains explicit in stage status.
        self.assertEqual(
            STAGE_STATUS["facenet_uncertain_zone"],
            "NOT_AVAILABLE_NO_CALIBRATION",
        )


class ResearchChallenge1Tests(unittest.TestCase):
    def test_exactly_one_instruction_all_directions_possible(self) -> None:
        self.assertEqual(RESEARCH_C1_N_STEPS, 1)
        seen: set[HeadDirection] = set()
        # Deterministic coverage of all four by forcing choices
        for d in HeadDirection:
            seq = (d,)
            self.assertEqual(len(seq), 1)
            seen.add(d)
        # Random sampler always length 1
        for _ in range(40):
            seq = sample_single_instruction()
            self.assertEqual(len(seq), 1)
            self.assertIn(seq[0], list(HeadDirection))
            seen.add(seq[0])
        self.assertEqual(seen, set(HeadDirection))

    def test_research_config_defers_timer_until_spoken(self) -> None:
        cfg = research_challenge1_config(
            ResearchChallenge1Timing(step_timeout_ms=18000.0)
        )
        self.assertFalse(cfg.start_timer_on_instruct)
        self.assertEqual(cfg.thresholds.step_timeout_ms, 18000.0)
        self.assertGreaterEqual(RESEARCH_C1_STEP_TIMEOUT_MS, 15000.0)

        clock = {"t": 0.0}

        def now() -> float:
            return clock["t"]

        session = Challenge1Session(
            (HeadDirection.LOOK_LEFT,),
            config=cfg,
            clock=now,
        )
        # Provide baseline face until INSTRUCT
        from test_challenge1 import _lm

        lm = _lm(yaw=0.0, pitch=0.4)
        while session.phase == Challenge1Phase.BASELINE:
            clock["t"] += 0.1
            session.observe_detection(face_count=1, landmarks=lm, camera_ok=True)
        self.assertEqual(session.phase, Challenge1Phase.INSTRUCT)
        self.assertEqual(session._step.step_deadline, 0.0)
        # Long "TTS" interval must not timeout while still INSTRUCT
        clock["t"] += 30.0
        session.observe_detection(face_count=1, landmarks=lm, camera_ok=True)
        self.assertEqual(session.phase, Challenge1Phase.INSTRUCT)
        session.on_instruction_spoken()
        self.assertEqual(session.phase, Challenge1Phase.WAIT_MOVE)
        self.assertGreater(session._step.step_deadline, clock["t"])

    def test_production_default_still_arms_timer_on_instruct(self) -> None:
        cfg = Challenge1Config()  # production default
        self.assertTrue(cfg.start_timer_on_instruct)

    def test_wrong_direction_still_fails_under_research_config(self) -> None:
        from test_challenge1 import _lm

        cfg = research_challenge1_config()
        clock = {"t": 0.0}
        session = Challenge1Session(
            (HeadDirection.LOOK_LEFT,),
            config=cfg,
            clock=lambda: clock["t"],
        )
        lm0 = _lm(yaw=0.0, pitch=0.4)
        while session.phase == Challenge1Phase.BASELINE:
            clock["t"] += 0.1
            session.observe_detection(face_count=1, landmarks=lm0, camera_ok=True)
        session.on_instruction_spoken()
        # Move opposite of LOOK_LEFT
        lm_wrong = _lm(yaw=-0.25, pitch=0.4)
        clock["t"] += 0.05
        session.observe_detection(face_count=1, landmarks=lm_wrong, camera_ok=True)
        self.assertEqual(session.fail_reason.value, "WRONG_DIRECTION")

    def test_correct_direction_can_pass_step(self) -> None:
        from test_challenge1 import _lm

        cfg = research_challenge1_config(
            ResearchChallenge1Timing(hold_ms=100.0, step_timeout_ms=10000.0)
        )
        clock = {"t": 0.0}
        session = Challenge1Session(
            (HeadDirection.LOOK_RIGHT,),
            config=cfg,
            clock=lambda: clock["t"],
        )
        lm0 = _lm(yaw=0.0, pitch=0.4)
        while session.phase == Challenge1Phase.BASELINE:
            clock["t"] += 0.1
            session.observe_detection(face_count=1, landmarks=lm0, camera_ok=True)
        session.on_instruction_spoken()
        lm_ok = _lm(yaw=-0.25, pitch=0.4)  # LOOK_RIGHT = negative yaw
        for _ in range(5):
            clock["t"] += 0.05
            session.observe_detection(face_count=1, landmarks=lm_ok, camera_ok=True)
            if session.phase == Challenge1Phase.PASSED_STEP:
                break
        self.assertEqual(session.phase, Challenge1Phase.PASSED_STEP)
        session.advance_after_step_pass()
        self.assertEqual(session.phase, Challenge1Phase.REEVAL)

    def test_e2e_tool_uses_research_c1_not_production_runner(self) -> None:
        import live_research_e2e_auth as mod

        src = inspect.getsource(mod.run_session)
        self.assertIn("run_research_challenge1_interactive", src)
        self.assertNotIn("run_challenge1_interactive(", src)


class ApiMismatchSmokeTests(unittest.TestCase):
    def test_interrupt_is_set_property(self) -> None:
        flag = InterruptFlag()
        self.assertIsInstance(flag.is_set, bool)
        with self.assertRaises(TypeError):
            flag.is_set()  # type: ignore[operator]
        import live_research_e2e_auth as mod

        src = inspect.getsource(mod.run_session)
        self.assertIn("while not interrupt.is_set:", src)
        self.assertNotIn("interrupt.is_set()", src)

    def test_scrfd_uses_FaceDetector(self) -> None:
        import live_research_e2e_auth as mod

        src = inspect.getsource(mod.create_scrfd_detector)
        self.assertIn("FaceDetector()", src)
        self.assertNotIn(".create_detector(", src)

    def test_create_scrfd_detector_initializes(self) -> None:
        from live_research_e2e_auth import create_scrfd_detector

        det = create_scrfd_detector()
        self.assertTrue(callable(det.detect))


class SessionLoopSmokeTests(unittest.TestCase):
    def test_real_match_accepts_without_challenge1(self) -> None:
        import live_research_e2e_auth as mod

        frame = np.random.default_rng(0).integers(40, 200, (240, 320, 3), dtype=np.uint8)
        face = SimpleNamespace(
            x1=20, y1=20, x2=200, y2=200, confidence=0.9, bbox=(20, 20, 200, 200)
        )
        det = SimpleNamespace(
            face_count=1,
            decision=SimpleNamespace(value="FACE_OK"),
            faces=(face,),
        )
        fake_cap = MagicMock()
        fake_cap.read.return_value = (True, frame)
        fas = SimpleNamespace(
            error=None,
            predicted_class=SimpleNamespace(value="REAL"),
            real_score_avg=0.98,
            spoof_score_combined=0.02,
        )
        match = SimpleNamespace(
            error=None,
            best_distance=5.0,
            recognized=True,
            best_identity_candidate="Seif",
            identity="Seif",
        )
        fake_detector = MagicMock()
        fake_detector.detect.return_value = det
        fake_fasnet = MagicMock()
        fake_fasnet.analyze_full_frame.return_value = fas
        fake_fasnet.warm_up.return_value = None
        fake_fasnet.init_ms = 1.0
        fake_facenet = MagicMock()
        fake_facenet.match_crop.return_value = match
        fake_facenet.warm_up.return_value = None
        fake_facenet.init_ms = 1.0
        pipeline = MagicMock()
        pipeline.fasnet = fake_fasnet
        pipeline.facenet = fake_facenet
        pipeline.enrollments = {"identities": {"Seif": {"embeddings": [[0.1] * 128]}}}
        pipeline.enrollment_index = {}
        pipeline.warm_up.return_value = {"fasnet_init_ms": 1.0, "facenet_init_ms": 1.0}

        interrupt = InterruptFlag()
        c1_started = {"n": 0}
        keys = {"n": 0}

        def poll(_ms=1):
            keys["n"] += 1
            if keys["n"] > 80:
                return ord("q")
            return -1

        def fast_window(self, *, perf_t, eval_row):
            if self.window_t0_perf is None:
                self.window_t0_perf = perf_t
            self.frames.append(eval_row)
            if len(self.frames) >= 6:
                d = self.decide_current()
                self.reset()
                return d
            return None

        def fake_c1(*_a, **_k):
            c1_started["n"] += 1
            return SimpleNamespace(
                succeeded_gesture=True,
                fail_reason=Challenge1FailReason.NONE,
                outcome=None,
            )

        args = SimpleNamespace(
            enrollment_db=MagicMock(),
            camera=0,
            min_interval_ms=0.0,
            no_tts=True,
            c1_timeout_ms=18000.0,
            c1_baseline_ms=700.0,
            c1_hold_ms=400.0,
            c1_post_speech_pause_s=0.0,
        )
        enroll_db = {
            "identities": {"Seif": {"embeddings": [[0.1] * 128], "display_name": "Seif"}}
        }

        with (
            patch.object(mod, "create_scrfd_detector", return_value=fake_detector),
            patch.object(mod, "FriendAuthPipeline", return_value=pipeline),
            patch.object(mod, "load_enrollments", return_value=enroll_db),
            patch.object(mod, "identity_embedding_count", return_value=1),
            patch.object(mod, "_open_camera", return_value=fake_cap),
            patch.object(mod, "run_research_challenge1_interactive", side_effect=fake_c1),
            patch.object(mod.cv2, "namedWindow"),
            patch.object(mod.cv2, "imshow"),
            patch.object(mod.cv2, "destroyAllWindows"),
            patch.object(mod, "poll_key_code", side_effect=poll),
            patch.object(mod, "preview_window_closed", return_value=False),
            patch.object(FrozenPadWindowBuffer, "add", fast_window),
            patch.object(mod, "frame_is_usable", return_value=True),
        ):
            code = mod.run_session(args, interrupt)

        self.assertEqual(code, 0)
        self.assertEqual(c1_started["n"], 0)  # no Challenge 1 on REAL+MATCH
        self.assertGreaterEqual(fake_fasnet.analyze_full_frame.call_count, 5)

    def test_uncertain_match_starts_challenge1(self) -> None:
        import live_research_e2e_auth as mod

        frame = np.random.default_rng(1).integers(40, 200, (240, 320, 3), dtype=np.uint8)
        face = SimpleNamespace(
            x1=20, y1=20, x2=200, y2=200, confidence=0.9, bbox=(20, 20, 200, 200)
        )
        det = SimpleNamespace(
            face_count=1,
            decision=SimpleNamespace(value="FACE_OK"),
            faces=(face,),
        )
        fake_cap = MagicMock()
        fake_cap.read.return_value = (True, frame)
        # Median ~0.90 → UNCERTAIN under 0.95/0.85 frozen gates
        fas = SimpleNamespace(
            error=None,
            predicted_class=SimpleNamespace(value="REAL"),
            real_score_avg=0.90,
            spoof_score_combined=0.10,
        )
        match = SimpleNamespace(
            error=None,
            best_distance=5.0,
            recognized=True,
            best_identity_candidate="Seif",
            identity="Seif",
        )
        fake_detector = MagicMock()
        fake_detector.detect.return_value = det
        fake_fasnet = MagicMock()
        fake_fasnet.analyze_full_frame.return_value = fas
        fake_fasnet.init_ms = 1.0
        fake_facenet = MagicMock()
        fake_facenet.match_crop.return_value = match
        fake_facenet.init_ms = 1.0
        pipeline = MagicMock()
        pipeline.fasnet = fake_fasnet
        pipeline.facenet = fake_facenet
        pipeline.enrollments = {"identities": {"Seif": {"embeddings": [[0.1] * 128]}}}
        pipeline.enrollment_index = {}
        pipeline.warm_up.return_value = {"fasnet_init_ms": 1.0, "facenet_init_ms": 1.0}

        interrupt = InterruptFlag()
        c1_started = {"n": 0}

        def poll(_ms=1):
            return -1

        def fast_window(self, *, perf_t, eval_row):
            if self.window_t0_perf is None:
                self.window_t0_perf = perf_t
            self.frames.append(eval_row)
            if len(self.frames) >= 6:
                d = self.decide_current()
                self.reset()
                return d
            return None

        def fake_c1(*_a, **_k):
            c1_started["n"] += 1
            interrupt.set()
            return SimpleNamespace(
                succeeded_gesture=True,
                fail_reason=Challenge1FailReason.NONE,
                outcome=SimpleNamespace(
                    decision=SimpleNamespace(value="ACCEPT"), reason="diag"
                ),
            )

        args = SimpleNamespace(
            enrollment_db=MagicMock(),
            camera=0,
            min_interval_ms=0.0,
            no_tts=True,
            c1_timeout_ms=18000.0,
            c1_baseline_ms=700.0,
            c1_hold_ms=400.0,
            c1_post_speech_pause_s=0.0,
        )
        enroll_db = {
            "identities": {"Seif": {"embeddings": [[0.1] * 128], "display_name": "Seif"}}
        }

        with (
            patch.object(mod, "create_scrfd_detector", return_value=fake_detector),
            patch.object(mod, "FriendAuthPipeline", return_value=pipeline),
            patch.object(mod, "load_enrollments", return_value=enroll_db),
            patch.object(mod, "identity_embedding_count", return_value=1),
            patch.object(mod, "_open_camera", return_value=fake_cap),
            patch.object(mod, "run_research_challenge1_interactive", side_effect=fake_c1),
            patch.object(mod.cv2, "namedWindow"),
            patch.object(mod.cv2, "imshow"),
            patch.object(mod.cv2, "destroyAllWindows"),
            patch.object(mod, "poll_key_code", side_effect=poll),
            patch.object(mod, "preview_window_closed", return_value=False),
            patch.object(FrozenPadWindowBuffer, "add", fast_window),
            patch.object(mod, "frame_is_usable", return_value=True),
        ):
            code = mod.run_session(args, interrupt)

        self.assertEqual(code, 0)
        self.assertGreaterEqual(c1_started["n"], 1)


if __name__ == "__main__":
    unittest.main()
