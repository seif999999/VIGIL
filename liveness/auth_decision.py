"""Secure authentication decision (post-HMAC perception).

ACCEPT only when:
  face_count == 1 (FACE_OK) AND confident/native REAL AND FaceNet recognized.

UNCERTAIN liveness (active research confidence policy) → CHALLENGE_REQUIRED
(non-accepting). Challenge 1 (head movement) is opt-in via live UI / verifier flags;
this module only routes to CHALLENGE_REQUIRED — it never ACCEPTS from gesture alone.

Identity failure always DENY — a challenge must never override unknown identity.
HMAC Accept is enforced upstream; this module never bypasses crypto gates.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

from facenet_identity import IdentityMatch
from fasnet_decision_policy import (
    FasNetConfidencePolicy,
    FasNetLivenessState,
    FasNetPolicyEvaluation,
    FasNetPolicyMode,
    default_shadow_policy,
    evaluate_fasnet_policy,
)
from fasnet_liveness import FasNetClass, FasNetResult


class AuthDecision(str, enum.Enum):
    ACCEPT = "ACCEPT"
    DENY = "DENY"
    UNCERTAIN = "UNCERTAIN"
    CHALLENGE_REQUIRED = "CHALLENGE_REQUIRED"


@dataclass(frozen=True)
class AuthOutcome:
    decision: AuthDecision
    reason: str
    face_count: int
    liveness_class: str | None
    recognized: bool
    identity: str | None
    # Confidence-policy diagnostics (always filled when decide_auth runs)
    fasnet_native_class: str | None = None
    fasnet_winning_score: float | None = None
    fasnet_auth_state: str | None = None
    fasnet_shadow_state: str | None = None
    fasnet_policy_mode: str | None = None
    fasnet_policy_status: str | None = None
    fasnet_real_min: float | None = None
    fasnet_spoof_min: float | None = None
    challenge_1_required: bool = False
    challenge_1_status: str = "NOT_APPLICABLE"


def _with_policy(
    outcome: AuthOutcome,
    ev: FasNetPolicyEvaluation,
    *,
    challenge_1_required: bool,
    challenge_1_status: str,
) -> AuthOutcome:
    return AuthOutcome(
        decision=outcome.decision,
        reason=outcome.reason,
        face_count=outcome.face_count,
        liveness_class=outcome.liveness_class,
        recognized=outcome.recognized,
        identity=outcome.identity,
        fasnet_native_class=ev.native_class,
        fasnet_winning_score=ev.native_winning_score,
        fasnet_auth_state=ev.auth_liveness_state.value,
        fasnet_shadow_state=ev.shadow_liveness_state.value,
        fasnet_policy_mode=ev.policy_mode.value,
        fasnet_policy_status=ev.policy_status,
        fasnet_real_min=ev.confident_real_min,
        fasnet_spoof_min=ev.confident_spoof_min,
        challenge_1_required=challenge_1_required,
        challenge_1_status=challenge_1_status,
    )


def decide_auth(
    *,
    face_decision_value: str,
    face_count: int,
    fasnet: FasNetResult | None,
    identity: IdentityMatch | None,
    fasnet_policy: FasNetConfidencePolicy | None = None,
) -> AuthOutcome:
    """Fail closed on missing inference, spoof, unknown identity, or face-count errors."""
    policy = fasnet_policy or default_shadow_policy()
    ev = evaluate_fasnet_policy(fasnet, policy)

    def finish(
        decision: AuthDecision,
        reason: str,
        liveness_class: str | None,
        recognized: bool,
        identity_name: str | None,
        *,
        challenge_1_required: bool = False,
        challenge_1_status: str = "NOT_APPLICABLE",
    ) -> AuthOutcome:
        base = AuthOutcome(
            decision,
            reason,
            face_count,
            liveness_class,
            recognized,
            identity_name,
        )
        return _with_policy(
            base,
            ev,
            challenge_1_required=challenge_1_required,
            challenge_1_status=challenge_1_status,
        )

    if face_decision_value == "NO_FACE" or face_count <= 0:
        return finish(AuthDecision.DENY, "no_face", None, False, None)
    if face_decision_value == "MULTIPLE_FACES" or face_count > 1:
        return finish(AuthDecision.DENY, "multiple_faces", None, False, None)
    if face_decision_value != "FACE_OK" or face_count != 1:
        return finish(AuthDecision.DENY, "face_gate_failed", None, False, None)

    if fasnet is None:
        return finish(
            AuthDecision.UNCERTAIN,
            "liveness_unavailable",
            None,
            False,
            None,
        )
    if fasnet.predicted_class == FasNetClass.ERROR or fasnet.is_real is None:
        return finish(
            AuthDecision.UNCERTAIN,
            f"liveness_error:{fasnet.error or 'unknown'}",
            FasNetClass.ERROR.value,
            False,
            None,
        )

    auth_live = ev.auth_liveness_state

    if auth_live == FasNetLivenessState.ERROR:
        return finish(
            AuthDecision.UNCERTAIN,
            f"liveness_error:{fasnet.error or 'unknown'}",
            FasNetClass.ERROR.value,
            False,
            None,
        )

    if auth_live == FasNetLivenessState.SPOOF:
        return finish(
            AuthDecision.DENY,
            "spoof",
            FasNetClass.SPOOF.value,
            False,
            None,
        )

    # UNCERTAIN under active research policy: still require identity first.
    # Challenge must never override an identity failure.
    if auth_live == FasNetLivenessState.UNCERTAIN:
        if policy.mode != FasNetPolicyMode.ACTIVE_RESEARCH:
            # Should not happen for binary/shadow auth_state mapping.
            return finish(
                AuthDecision.UNCERTAIN,
                "liveness_uncertain",
                FasNetLivenessState.UNCERTAIN.value,
                False,
                None,
            )
        # Identity gate before challenge routing
        if identity is None:
            return finish(
                AuthDecision.UNCERTAIN,
                "identity_unavailable",
                FasNetLivenessState.UNCERTAIN.value,
                False,
                None,
            )
        if identity.error:
            return finish(
                AuthDecision.UNCERTAIN,
                f"identity_error:{identity.error}",
                FasNetLivenessState.UNCERTAIN.value,
                False,
                None,
            )
        if not identity.recognized:
            return finish(
                AuthDecision.DENY,
                "unknown_identity",
                FasNetLivenessState.UNCERTAIN.value,
                False,
                identity.best_identity_candidate,
            )
        # Recognized but liveness uncertain → challenge path (not implemented)
        return finish(
            AuthDecision.CHALLENGE_REQUIRED,
            "challenge_required_not_implemented",
            FasNetLivenessState.UNCERTAIN.value,
            True,
            identity.display_name or identity.identity,
            challenge_1_required=True,
            challenge_1_status="REQUIRED_NOT_IMPLEMENTED",
        )

    # auth_live == REAL
    if identity is None:
        return finish(
            AuthDecision.UNCERTAIN,
            "identity_unavailable",
            FasNetClass.REAL.value,
            False,
            None,
        )
    if identity.error:
        return finish(
            AuthDecision.UNCERTAIN,
            f"identity_error:{identity.error}",
            FasNetClass.REAL.value,
            False,
            None,
        )
    if not identity.recognized:
        return finish(
            AuthDecision.DENY,
            "unknown_identity",
            FasNetClass.REAL.value,
            False,
            identity.best_identity_candidate,
        )

    # Shadow diagnostic: would research gates have required a challenge?
    shadow_challenge = (
        policy.mode in (FasNetPolicyMode.SHADOW, FasNetPolicyMode.BINARY_NATIVE)
        and ev.shadow_liveness_state == FasNetLivenessState.UNCERTAIN
    )
    return finish(
        AuthDecision.ACCEPT,
        "real_and_recognized",
        FasNetClass.REAL.value,
        True,
        identity.display_name or identity.identity,
        challenge_1_required=False,
        challenge_1_status=(
            "SHADOW_WOULD_REQUIRE_CHALLENGE"
            if shadow_challenge
            else "NOT_REQUIRED"
        ),
    )
