"""Research E2E joint auth routing (LOCAL TEST only).

NON-NEGOTIABLE thesis rule:
  Challenge 1 triggers ONLY when the combined assessment is UNCERTAIN —
  never when FasNet is confidently REAL and FaceNet is a confident MATCH.

PAD (FasNet): frozen 5s median REAL-class mass @ 0.95 / SPOOF @ 0.85 / min_valid=5
  → REAL | SPOOF | UNCERTAIN | INVALID | ERROR

Identity (FaceNet): existing binary MATCH_THRESHOLD 9.0 only
  → MATCH (distance < 9.0) | UNKNOWN (distance >= 9.0) | UNAVAILABLE
  There is NO calibrated identity UNCERTAIN zone yet (no GENUINE/IMPOSTOR
  distance calibration evidence on disk). Do not invent gray-zone thresholds.
  Do not label UNKNOWN as UNCERTAIN.

Supported joint policy:
  REAL + MATCH           → ACCEPT (direct; no Challenge 1)
  UNCERTAIN + MATCH      → Challenge 1
  SPOOF                  → DENY
  UNKNOWN identity       → DENY (confident rejection under binary threshold)
  UNAVAILABLE identity   → DENY / fail closed
  INVALID/ERROR PAD      → RETRY then fail closed

Not yet implementable (no identity UNCERTAIN state):
  REAL + identity UNCERTAIN → Challenge 1
  both UNCERTAIN           → Challenge 1

Post-Challenge 1: gesture never ACCEPT; require fresh PAD REAL + MATCH.
If still UNCERTAIN after C1 → Challenge 2 unavailable (fail closed).
Face-stage ACCEPT here is research-only while Challenge 2 is unimplemented.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any

from facenet_identity import MATCH_THRESHOLD, MATCH_THRESHOLD_STATUS
from research_e2e_pad import (
    FROZEN_PAD_MIN_VALID,
    FROZEN_PAD_REAL_MIN,
    FROZEN_PAD_SPOOF_MIN,
    FROZEN_PAD_WINDOW_S,
)


class E2EPhase(str, enum.Enum):
    COLLECTING_PAD = "COLLECTING_PAD"
    AWAITING_CHALLENGE1 = "AWAITING_CHALLENGE1"
    CHALLENGE1_RUNNING = "CHALLENGE1_RUNNING"
    COLLECTING_POST_C1_PAD = "COLLECTING_POST_C1_PAD"
    CHALLENGE2_BLOCKED = "CHALLENGE2_BLOCKED"
    ACCEPT = "ACCEPT"
    DENY = "DENY"
    RETRY = "RETRY"


class E2EDecision(str, enum.Enum):
    PENDING = "PENDING"
    ACCEPT = "ACCEPT"
    DENY = "DENY"
    RETRY = "RETRY"
    CHALLENGE1_REQUIRED = "CHALLENGE1_REQUIRED"
    CHALLENGE2_UNAVAILABLE = "CHALLENGE2_UNAVAILABLE"


class IdentityState(str, enum.Enum):
    """Binary FaceNet outcome — no calibrated UNCERTAIN."""

    MATCH = "MATCH"
    UNKNOWN = "UNKNOWN"
    UNAVAILABLE = "UNAVAILABLE"
    # Placeholder for future evidence-based gray zone (not activated).
    UNCERTAIN_NOT_AVAILABLE = "UNCERTAIN_NOT_AVAILABLE"


STAGE_STATUS = {
    "scrfd": "IMPLEMENTED",
    "fasnet_frozen_pad": "IMPLEMENTED",
    "facenet_binary_match": "IMPLEMENTED",
    "facenet_uncertain_zone": "NOT_AVAILABLE_NO_CALIBRATION",
    "challenge1_on_joint_uncertain": "IMPLEMENTED",
    "challenge1_on_real_match": "DISABLED_BY_THESIS",
    "challenge2_voice": "NOT_IMPLEMENTED",
    "default_live_auth": "UNCHANGED",
}

KNOWN_LIMITATION = (
    "Moving printed photographs can sometimes fool FasNet and may also fool "
    "landmark-based head-motion checks. Integration does not eliminate this."
)

FACENET_UNCERTAIN_GAP = (
    "FaceNet exposes only MATCH_THRESHOLD=9.0 (UNVALIDATED_FRIEND_THRESHOLD). "
    "No GENUINE/IMPOSTOR distance calibration JSONL was found under "
    "data/identity_calibration/, so a defensible MATCH/UNCERTAIN/UNKNOWN "
    "split cannot be established. UNKNOWN (distance >= 9.0) is a confident "
    "rejection under the binary rule — not identity UNCERTAIN. Joint cases "
    "that require identity UNCERTAIN remain unimplemented."
)


@dataclass(frozen=True)
class IdentityWindowSummary:
    """Identity over a PAD window — separate from PAD state."""

    n_facenet_ok: int
    best_distance: float | None
    recognized: bool
    best_identity: str | None
    state: IdentityState
    threshold: float = MATCH_THRESHOLD
    threshold_status: str = MATCH_THRESHOLD_STATUS

    @property
    def unknown(self) -> bool:
        return self.state == IdentityState.UNKNOWN


@dataclass(frozen=True)
class E2ERoute:
    decision: E2EDecision
    phase: E2EPhase
    reason: str
    pad_state: str
    identity_state: str
    identity_recognized: bool
    next_action: str
    joint_summary: str


def summarize_identity(
    distances: list[tuple[float, str | None]],
    *,
    threshold: float = MATCH_THRESHOLD,
) -> IdentityWindowSummary:
    """Min L2 over successful FaceNet evals; MATCH iff min < threshold."""
    if not distances:
        return IdentityWindowSummary(
            n_facenet_ok=0,
            best_distance=None,
            recognized=False,
            best_identity=None,
            state=IdentityState.UNAVAILABLE,
            threshold=threshold,
        )
    best_d, best_id = min(distances, key=lambda t: t[0])
    matched = bool(best_d < threshold)
    return IdentityWindowSummary(
        n_facenet_ok=len(distances),
        best_distance=float(best_d),
        recognized=matched,
        best_identity=best_id,
        state=IdentityState.MATCH if matched else IdentityState.UNKNOWN,
        threshold=threshold,
    )


def joint_label(pad_state: str, identity: IdentityWindowSummary) -> str:
    return f"PAD={pad_state}/{identity.state.value}"


def route_after_pad_window(
    *,
    pad_state: str,
    identity: IdentityWindowSummary,
    post_challenge1: bool = False,
    max_invalid_retries: int = 3,
    invalid_retry_count: int = 0,
) -> E2ERoute:
    """Map frozen PAD + binary identity to the next integrated action."""
    st = (pad_state or "").upper()
    id_st = identity.state
    jl = joint_label(st, identity)

    def _route(
        decision: E2EDecision,
        phase: E2EPhase,
        reason: str,
        next_action: str,
    ) -> E2ERoute:
        return E2ERoute(
            decision=decision,
            phase=phase,
            reason=reason,
            pad_state=st,
            identity_state=id_st.value,
            identity_recognized=identity.recognized,
            next_action=next_action,
            joint_summary=jl,
        )

    if st == "SPOOF":
        return _route(
            E2EDecision.DENY,
            E2EPhase.DENY,
            "pad_spoof",
            "fail_closed_spoof",
        )

    if st in ("INVALID", "ERROR"):
        if invalid_retry_count + 1 >= max_invalid_retries:
            return _route(
                E2EDecision.DENY,
                E2EPhase.DENY,
                f"pad_{st.lower()}_retries_exhausted",
                "fail_closed_insufficient_evidence",
            )
        return _route(
            E2EDecision.RETRY,
            E2EPhase.RETRY,
            f"pad_{st.lower()}_retry",
            "collect_another_pad_window",
        )

    if id_st == IdentityState.UNAVAILABLE:
        return _route(
            E2EDecision.DENY,
            E2EPhase.DENY,
            "identity_unavailable",
            "fail_closed_no_facenet",
        )

    # Binary FaceNet: distance >= 9.0 is confident UNKNOWN — not UNCERTAIN.
    if id_st == IdentityState.UNKNOWN:
        return _route(
            E2EDecision.DENY,
            E2EPhase.DENY,
            "unknown_identity",
            "fail_closed_unknown_identity",
        )

    # From here: identity is MATCH
    if st == "UNCERTAIN":
        if post_challenge1:
            return _route(
                E2EDecision.CHALLENGE2_UNAVAILABLE,
                E2EPhase.CHALLENGE2_BLOCKED,
                "challenge_2_not_implemented_after_c1_still_uncertain",
                "fail_closed_challenge2_unavailable",
            )
        return _route(
            E2EDecision.CHALLENGE1_REQUIRED,
            E2EPhase.AWAITING_CHALLENGE1,
            "joint_uncertain_pad_uncertain_identity_match",
            "run_challenge1",
        )

    if st == "REAL":
        if post_challenge1:
            return _route(
                E2EDecision.ACCEPT,
                E2EPhase.ACCEPT,
                "post_c1_pad_real_and_match",
                "accept_face_stage_research_only",
            )
        # Thesis: confident REAL + MATCH → ACCEPT directly (no Challenge 1).
        return _route(
            E2EDecision.ACCEPT,
            E2EPhase.ACCEPT,
            "joint_confident_real_and_match",
            "accept_direct",
        )

    return _route(
        E2EDecision.DENY,
        E2EPhase.DENY,
        f"pad_unhandled_state:{st}",
        "fail_closed",
    )


def route_after_challenge1(
    *,
    succeeded_gesture: bool,
    fail_reason: str | None,
) -> E2ERoute:
    """Gesture alone never ACCEPT — only gate entry to post-C1 PAD collection."""
    if not succeeded_gesture:
        return E2ERoute(
            decision=E2EDecision.DENY,
            phase=E2EPhase.DENY,
            reason=f"challenge1_failed:{fail_reason or 'unknown'}",
            pad_state="N/A",
            identity_state="N/A",
            identity_recognized=False,
            next_action="fail_closed_challenge1",
            joint_summary="C1=FAIL",
        )
    return E2ERoute(
        decision=E2EDecision.PENDING,
        phase=E2EPhase.COLLECTING_POST_C1_PAD,
        reason="challenge1_gesture_ok_collect_fresh_pad",
        pad_state="N/A",
        identity_state="N/A",
        identity_recognized=True,
        next_action="collect_post_c1_pad_window",
        joint_summary="C1=GESTURE_OK",
    )


def frozen_pad_banner() -> dict[str, Any]:
    return {
        "window_s": FROZEN_PAD_WINDOW_S,
        "method": "median_real",
        "real_min": FROZEN_PAD_REAL_MIN,
        "spoof_min": FROZEN_PAD_SPOOF_MIN,
        "min_valid": FROZEN_PAD_MIN_VALID,
        "match_threshold": MATCH_THRESHOLD,
        "match_threshold_status": MATCH_THRESHOLD_STATUS,
        "stages": dict(STAGE_STATUS),
        "known_limitation": KNOWN_LIMITATION,
        "facenet_uncertain_gap": FACENET_UNCERTAIN_GAP,
        "production_activation": False,
        "challenge1_policy": "only_on_joint_uncertainty_pad_uncertain_plus_match",
    }
