"""Challenge 1 interactive head-movement session (fail-closed).

Opt-in only. Does not change FaceNet threshold 9.0 or FasNet score semantics.
Gesture completion alone never ACCEPT — fresh SCRFD/FasNet/FaceNet re-eval required.
"""

from __future__ import annotations

import enum
import secrets
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from auth_decision import AuthDecision, AuthOutcome
from challenge1_pose import (
    HeadDirection,
    PoseProxies,
    PoseThresholds,
    direction_deltas,
    direction_progress,
    pose_from_landmarks,
)
from facenet_identity import IdentityMatch
from fasnet_decision_policy import FasNetLivenessState
from fasnet_liveness import FasNetClass, FasNetResult


class Challenge1Phase(str, enum.Enum):
    IDLE = "IDLE"
    BASELINE = "BASELINE"
    INSTRUCT = "INSTRUCT"
    WAIT_MOVE = "WAIT_MOVE"
    HOLD = "HOLD"
    REEVAL = "REEVAL"
    PASSED_STEP = "PASSED_STEP"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    DONE = "DONE"


class Challenge1FailReason(str, enum.Enum):
    NONE = "NONE"
    TIMEOUT = "TIMEOUT"
    NO_FACE = "NO_FACE"
    MULTIPLE_FACES = "MULTIPLE_FACES"
    TRACKING_UNCERTAIN = "TRACKING_UNCERTAIN"
    WRONG_DIRECTION = "WRONG_DIRECTION"
    STATIC_NO_MOVE = "STATIC_NO_MOVE"
    FACE_LOST = "FACE_LOST"
    CAMERA_DISCONNECTED = "CAMERA_DISCONNECTED"
    STALE_FRAME = "STALE_FRAME"
    REPLAY_OBSERVATION = "REPLAY_OBSERVATION"
    SPOOF = "SPOOF"
    UNKNOWN_IDENTITY = "UNKNOWN_IDENTITY"
    IDENTITY_ERROR = "IDENTITY_ERROR"
    LIVENESS_ERROR = "LIVENESS_ERROR"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    CANCELLED = "CANCELLED"


ALL_DIRECTIONS: tuple[HeadDirection, ...] = (
    HeadDirection.LOOK_LEFT,
    HeadDirection.LOOK_RIGHT,
    HeadDirection.LOOK_UP,
    HeadDirection.LOOK_DOWN,
)


def sample_instruction_sequence(
    *,
    n_steps: int | None = None,
    rng: secrets.SystemRandom | None = None,
) -> tuple[HeadDirection, ...]:
    """Unpredictable 1–2 step sequence (secure RNG by default)."""
    r = rng or secrets.SystemRandom()
    if n_steps is None:
        n_steps = r.choice((1, 2))
    n_steps = max(1, min(2, int(n_steps)))
    first = r.choice(ALL_DIRECTIONS)
    if n_steps == 1:
        return (first,)
    rest = [d for d in ALL_DIRECTIONS if d != first]
    second = r.choice(rest)
    return (first, second)


@dataclass
class Challenge1Config:
    thresholds: PoseThresholds = field(default_factory=PoseThresholds)
    baseline_ms: float = 350.0
    min_motion_before_hold: bool = True
    require_fresh_reeval: bool = True
    max_frame_age_ms: float = 500.0
    # After gesture: need this many consecutive successful re-eval frames
    reeval_passes_needed: int = 1
    # When True (production default), step timeout starts at INSTRUCT.
    # Research E2E sets False so TTS can finish before the move window opens.
    start_timer_on_instruct: bool = True


@dataclass
class Challenge1HudState:
    session_id: str
    phase: str
    instructions: tuple[str, ...]
    current_index: int
    current_instruction: str | None
    progress: float
    remaining_ms: float
    movement_verified: bool
    step_verified: bool
    fail_reason: str
    reeval_liveness: str | None
    reeval_identity: str | None
    final_decision: str | None
    final_reason: str | None


@dataclass
class Challenge1Result:
    session_id: str
    instructions: tuple[HeadDirection, ...]
    succeeded_gesture: bool
    fail_reason: Challenge1FailReason
    outcome: AuthOutcome | None
    reeval_fasnet: FasNetResult | None
    reeval_identity: IdentityMatch | None
    observation_token: str


@dataclass
class _StepRuntime:
    direction: HeadDirection
    baseline: PoseProxies | None = None
    baseline_deadline: float = 0.0
    step_deadline: float = 0.0
    hold_until: float | None = None
    saw_motion: bool = False
    progress: float = 0.0


class Challenge1Session:
    """Stateful Challenge 1 verifier. One session = one authentication attempt."""

    def __init__(
        self,
        instructions: Sequence[HeadDirection],
        *,
        config: Challenge1Config | None = None,
        expected_identity: str | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if not instructions:
            raise ValueError("instructions required")
        self.instructions = tuple(instructions)
        self.config = config or Challenge1Config()
        self.expected_identity = expected_identity
        self._clock = clock
        self.session_id = uuid.uuid4().hex
        self.observation_token = uuid.uuid4().hex
        self._used_tokens: set[str] = set()
        self.phase = Challenge1Phase.BASELINE
        self.fail_reason = Challenge1FailReason.NONE
        self._step_i = 0
        self._step = _StepRuntime(direction=self.instructions[0])
        self._step.baseline_deadline = self._clock() + self.config.baseline_ms / 1000.0
        self._final_outcome: AuthOutcome | None = None
        self._reeval_fasnet: FasNetResult | None = None
        self._reeval_identity: IdentityMatch | None = None
        self._reeval_passes = 0
        self._last_progress = 0.0
        self.movement_verified = False

    @property
    def done(self) -> bool:
        return self.phase in (
            Challenge1Phase.SUCCEEDED,
            Challenge1Phase.FAILED,
            Challenge1Phase.DONE,
        )

    def hud(self) -> Challenge1HudState:
        now = self._clock()
        rem = 0.0
        if self.phase in (
            Challenge1Phase.INSTRUCT,
            Challenge1Phase.WAIT_MOVE,
            Challenge1Phase.HOLD,
            Challenge1Phase.BASELINE,
        ):
            rem = max(0.0, (self._step.step_deadline - now) * 1000.0) if self._step.step_deadline else 0.0
            if self.phase == Challenge1Phase.BASELINE:
                rem = max(0.0, (self._step.baseline_deadline - now) * 1000.0)
        cur = (
            self.instructions[self._step_i].value
            if self._step_i < len(self.instructions)
            else None
        )
        return Challenge1HudState(
            session_id=self.session_id,
            phase=self.phase.value,
            instructions=tuple(d.value for d in self.instructions),
            current_index=self._step_i,
            current_instruction=cur,
            progress=float(self._last_progress),
            remaining_ms=rem,
            movement_verified=self.movement_verified,
            step_verified=self.phase
            in (Challenge1Phase.PASSED_STEP, Challenge1Phase.REEVAL, Challenge1Phase.SUCCEEDED),
            fail_reason=self.fail_reason.value,
            reeval_liveness=(
                self._reeval_fasnet.predicted_class.value
                if self._reeval_fasnet is not None
                else None
            ),
            reeval_identity=(
                self._reeval_identity.display_name
                or self._reeval_identity.identity
                if self._reeval_identity is not None
                else None
            ),
            final_decision=(
                self._final_outcome.decision.value if self._final_outcome else None
            ),
            final_reason=self._final_outcome.reason if self._final_outcome else None,
        )

    def _fail(self, reason: Challenge1FailReason) -> None:
        self.phase = Challenge1Phase.FAILED
        self.fail_reason = reason
        self.movement_verified = False

    def begin_step_timer(self) -> None:
        thr = self.config.thresholds
        self._step.step_deadline = self._clock() + thr.step_timeout_ms / 1000.0

    def on_instruction_spoken(self) -> None:
        """Call after TTS/HUD shows the current instruction — motion must start after."""
        if self.phase == Challenge1Phase.BASELINE:
            return
        self.phase = Challenge1Phase.WAIT_MOVE
        self.begin_step_timer()
        self._step.hold_until = None
        self._step.saw_motion = False
        self._last_progress = 0.0

    def observe_detection(
        self,
        *,
        face_count: int,
        landmarks: np.ndarray | None,
        frame_age_ms: float | None = None,
        camera_ok: bool = True,
        observation_token: str | None = None,
    ) -> Challenge1Phase:
        """Feed SCRFD-only observations (no FasNet/FaceNet required)."""
        if self.done:
            return self.phase

        if observation_token is not None:
            if observation_token in self._used_tokens:
                self._fail(Challenge1FailReason.REPLAY_OBSERVATION)
                return self.phase
            # Token recorded when step completes — early reuse of same token blocked below

        if not camera_ok:
            self._fail(Challenge1FailReason.CAMERA_DISCONNECTED)
            return self.phase

        if frame_age_ms is not None and frame_age_ms > self.config.max_frame_age_ms:
            self._fail(Challenge1FailReason.STALE_FRAME)
            return self.phase

        if face_count <= 0:
            if self.phase in (Challenge1Phase.WAIT_MOVE, Challenge1Phase.HOLD, Challenge1Phase.INSTRUCT):
                self._fail(Challenge1FailReason.FACE_LOST)
            else:
                self._fail(Challenge1FailReason.NO_FACE)
            return self.phase
        if face_count > 1:
            self._fail(Challenge1FailReason.MULTIPLE_FACES)
            return self.phase

        pose = pose_from_landmarks(landmarks) if landmarks is not None else None
        if pose is None:
            self._fail(Challenge1FailReason.TRACKING_UNCERTAIN)
            return self.phase

        now = self._clock()
        thr = self.config.thresholds

        if self.phase == Challenge1Phase.BASELINE:
            if now < self._step.baseline_deadline:
                # Keep updating baseline until deadline (face present before instruction)
                self._step.baseline = pose
                return self.phase
            # Baseline locked — ready to instruct
            if self._step.baseline is None:
                self._step.baseline = pose
            self.phase = Challenge1Phase.INSTRUCT
            if self.config.start_timer_on_instruct:
                self.begin_step_timer()
            return self.phase

        if self.phase == Challenge1Phase.INSTRUCT:
            # Waiting for caller to announce; still track presence.
            # Timeout only if a deadline was armed (production default).
            if self._step.step_deadline > 0 and now > self._step.step_deadline:
                self._fail(Challenge1FailReason.TIMEOUT)
            return self.phase

        if self.phase in (Challenge1Phase.WAIT_MOVE, Challenge1Phase.HOLD):
            if now > self._step.step_deadline:
                if not self._step.saw_motion:
                    self._fail(Challenge1FailReason.STATIC_NO_MOVE)
                else:
                    self._fail(Challenge1FailReason.TIMEOUT)
                return self.phase

            assert self._step.baseline is not None
            d_yaw, d_pitch = direction_deltas(self._step.baseline, pose)
            progress, reached, wrong = direction_progress(
                self._step.direction, d_yaw, d_pitch, thr
            )
            self._last_progress = max(0.0, progress)
            if wrong:
                self._fail(Challenge1FailReason.WRONG_DIRECTION)
                return self.phase
            if progress >= 0.35:
                self._step.saw_motion = True

            if self.phase == Challenge1Phase.WAIT_MOVE:
                if reached:
                    self.phase = Challenge1Phase.HOLD
                    self._step.hold_until = now + thr.hold_ms / 1000.0
                return self.phase

            # HOLD
            if not reached:
                # Left the pose too early
                self.phase = Challenge1Phase.WAIT_MOVE
                self._step.hold_until = None
                return self.phase
            if self._step.hold_until is not None and now >= self._step.hold_until:
                self.movement_verified = True
                self.phase = Challenge1Phase.PASSED_STEP
                if observation_token:
                    self._used_tokens.add(observation_token)
            return self.phase

        return self.phase

    def advance_after_step_pass(self) -> Challenge1Phase:
        """After a step is held, either next instruction or enter REEVAL."""
        if self.phase != Challenge1Phase.PASSED_STEP:
            return self.phase
        self._step_i += 1
        if self._step_i >= len(self.instructions):
            self.phase = Challenge1Phase.REEVAL
            self._reeval_passes = 0
            return self.phase
        # Next sequential instruction — new baseline from current pose required
        self._step = _StepRuntime(direction=self.instructions[self._step_i])
        self._step.baseline_deadline = self._clock() + self.config.baseline_ms / 1000.0
        self.phase = Challenge1Phase.BASELINE
        self.movement_verified = False
        self._last_progress = 0.0
        return self.phase

    def submit_reeval(
        self,
        *,
        auth_liveness: FasNetLivenessState,
        fasnet: FasNetResult | None,
        identity: IdentityMatch | None,
        face_count: int,
        base_outcome: AuthOutcome,
    ) -> Challenge1Result:
        """Apply fresh FasNet/FaceNet after gesture. Gesture cannot override spoof/unknown."""
        if self.phase != Challenge1Phase.REEVAL:
            if self.done:
                return self.result()
            self._fail(Challenge1FailReason.INSUFFICIENT_EVIDENCE)
            return self.result()

        self._reeval_fasnet = fasnet
        self._reeval_identity = identity

        if face_count <= 0:
            self._fail(Challenge1FailReason.NO_FACE)
            return self._finalize_failed(base_outcome)
        if face_count > 1:
            self._fail(Challenge1FailReason.MULTIPLE_FACES)
            return self._finalize_failed(base_outcome)

        if fasnet is None or fasnet.predicted_class == FasNetClass.ERROR:
            self._fail(Challenge1FailReason.LIVENESS_ERROR)
            return self._finalize_failed(base_outcome)

        if auth_liveness == FasNetLivenessState.SPOOF:
            self._fail(Challenge1FailReason.SPOOF)
            return self._finalize_failed(
                AuthOutcome(
                    AuthDecision.DENY,
                    "challenge1_spoof",
                    face_count,
                    FasNetClass.SPOOF.value,
                    False,
                    None,
                    challenge_1_required=False,
                    challenge_1_status="FAILED_SPOOF",
                )
            )

        if identity is None or identity.error:
            self._fail(Challenge1FailReason.IDENTITY_ERROR)
            return self._finalize_failed(base_outcome)
        if not identity.recognized:
            self._fail(Challenge1FailReason.UNKNOWN_IDENTITY)
            return self._finalize_failed(
                AuthOutcome(
                    AuthDecision.DENY,
                    "challenge1_unknown_identity",
                    face_count,
                    auth_liveness.value,
                    False,
                    identity.best_identity_candidate,
                    challenge_1_required=False,
                    challenge_1_status="FAILED_UNKNOWN_IDENTITY",
                )
            )

        if (
            self.expected_identity
            and identity.identity
            and identity.identity != self.expected_identity
        ):
            self._fail(Challenge1FailReason.UNKNOWN_IDENTITY)
            return self._finalize_failed(
                AuthOutcome(
                    AuthDecision.DENY,
                    "challenge1_identity_mismatch",
                    face_count,
                    auth_liveness.value,
                    False,
                    identity.best_identity_candidate,
                    challenge_1_required=False,
                    challenge_1_status="FAILED_IDENTITY_MISMATCH",
                )
            )

        if auth_liveness == FasNetLivenessState.REAL and identity.recognized:
            self._reeval_passes += 1
            if self._reeval_passes >= self.config.reeval_passes_needed:
                self.phase = Challenge1Phase.SUCCEEDED
                self._final_outcome = AuthOutcome(
                    AuthDecision.ACCEPT,
                    "challenge1_passed_real_and_recognized",
                    face_count,
                    FasNetClass.REAL.value,
                    True,
                    identity.display_name or identity.identity,
                    challenge_1_required=False,
                    challenge_1_status="PASSED",
                )
                return self.result()
            return self.result()

        # Gesture OK but still uncertain / insufficient → Challenge 2 required (non-accepting)
        self.phase = Challenge1Phase.DONE
        self.fail_reason = Challenge1FailReason.INSUFFICIENT_EVIDENCE
        self._final_outcome = AuthOutcome(
            AuthDecision.CHALLENGE_REQUIRED,
            "challenge_2_required",
            face_count,
            auth_liveness.value,
            True,
            identity.display_name or identity.identity,
            challenge_1_required=False,
            challenge_1_status="PASSED_INSUFFICIENT_CHALLENGE_2_REQUIRED",
        )
        return self.result()

    def _finalize_failed(self, outcome: AuthOutcome) -> Challenge1Result:
        self.phase = Challenge1Phase.FAILED
        self._final_outcome = outcome
        return self.result()

    def cancel(self) -> None:
        self._fail(Challenge1FailReason.CANCELLED)

    def result(self) -> Challenge1Result:
        return Challenge1Result(
            session_id=self.session_id,
            instructions=self.instructions,
            succeeded_gesture=self.phase
            in (Challenge1Phase.SUCCEEDED, Challenge1Phase.DONE)
            and self.fail_reason
            in (Challenge1FailReason.NONE, Challenge1FailReason.INSUFFICIENT_EVIDENCE),
            fail_reason=self.fail_reason,
            outcome=self._final_outcome,
            reeval_fasnet=self._reeval_fasnet,
            reeval_identity=self._reeval_identity,
            observation_token=self.observation_token,
        )


def decide_challenge1_gate(
    *,
    challenge_enabled: bool,
    outcome: AuthOutcome,
) -> AuthOutcome:
    """Rewrite CHALLENGE_REQUIRED status when Challenge 1 is available (opt-in)."""
    if not challenge_enabled:
        return outcome
    if outcome.decision != AuthDecision.CHALLENGE_REQUIRED:
        return outcome
    if outcome.challenge_1_status == "REQUIRED_NOT_IMPLEMENTED":
        return AuthOutcome(
            decision=outcome.decision,
            reason="challenge_1_required",
            face_count=outcome.face_count,
            liveness_class=outcome.liveness_class,
            recognized=outcome.recognized,
            identity=outcome.identity,
            fasnet_native_class=outcome.fasnet_native_class,
            fasnet_winning_score=outcome.fasnet_winning_score,
            fasnet_auth_state=outcome.fasnet_auth_state,
            fasnet_shadow_state=outcome.fasnet_shadow_state,
            fasnet_policy_mode=outcome.fasnet_policy_mode,
            fasnet_policy_status=outcome.fasnet_policy_status,
            fasnet_real_min=outcome.fasnet_real_min,
            fasnet_spoof_min=outcome.fasnet_spoof_min,
            challenge_1_required=True,
            challenge_1_status="REQUIRED",
        )
    return outcome
