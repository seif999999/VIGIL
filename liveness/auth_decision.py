"""Secure authentication decision (post-HMAC perception).

ACCEPT only when:
  face_count == 1 (FACE_OK) AND FasNet REAL AND FaceNet recognized.

HMAC Accept is enforced upstream; this module never bypasses crypto gates.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

from facenet_identity import IdentityMatch
from fasnet_liveness import FasNetClass, FasNetResult


class AuthDecision(str, enum.Enum):
    ACCEPT = "ACCEPT"
    DENY = "DENY"
    UNCERTAIN = "UNCERTAIN"


@dataclass(frozen=True)
class AuthOutcome:
    decision: AuthDecision
    reason: str
    face_count: int
    liveness_class: str | None
    recognized: bool
    identity: str | None


def decide_auth(
    *,
    face_decision_value: str,
    face_count: int,
    fasnet: FasNetResult | None,
    identity: IdentityMatch | None,
) -> AuthOutcome:
    """Fail closed on missing inference, spoof, unknown identity, or face-count errors."""
    if face_decision_value == "NO_FACE" or face_count <= 0:
        return AuthOutcome(
            AuthDecision.DENY, "no_face", face_count, None, False, None
        )
    if face_decision_value == "MULTIPLE_FACES" or face_count > 1:
        return AuthOutcome(
            AuthDecision.DENY, "multiple_faces", face_count, None, False, None
        )
    if face_decision_value != "FACE_OK" or face_count != 1:
        return AuthOutcome(
            AuthDecision.DENY, "face_gate_failed", face_count, None, False, None
        )

    if fasnet is None:
        return AuthOutcome(
            AuthDecision.UNCERTAIN,
            "liveness_unavailable",
            face_count,
            None,
            False,
            None,
        )
    if fasnet.predicted_class == FasNetClass.ERROR or fasnet.is_real is None:
        return AuthOutcome(
            AuthDecision.UNCERTAIN,
            f"liveness_error:{fasnet.error or 'unknown'}",
            face_count,
            FasNetClass.ERROR.value,
            False,
            None,
        )
    if fasnet.is_real is False or fasnet.predicted_class == FasNetClass.SPOOF:
        return AuthOutcome(
            AuthDecision.DENY,
            "spoof",
            face_count,
            FasNetClass.SPOOF.value,
            False,
            None,
        )

    if identity is None:
        return AuthOutcome(
            AuthDecision.UNCERTAIN,
            "identity_unavailable",
            face_count,
            FasNetClass.REAL.value,
            False,
            None,
        )
    if identity.error:
        return AuthOutcome(
            AuthDecision.UNCERTAIN,
            f"identity_error:{identity.error}",
            face_count,
            FasNetClass.REAL.value,
            False,
            None,
        )
    if not identity.recognized:
        return AuthOutcome(
            AuthDecision.DENY,
            "unknown_identity",
            face_count,
            FasNetClass.REAL.value,
            False,
            identity.best_identity_candidate,
        )

    return AuthOutcome(
        AuthDecision.ACCEPT,
        "real_and_recognized",
        face_count,
        FasNetClass.REAL.value,
        True,
        identity.display_name or identity.identity,
    )
