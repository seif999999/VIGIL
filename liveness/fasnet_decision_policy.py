"""FasNet three-state liveness policy (separate from anti-spoof-mn3).

mn3 uses dual softmax probs (real_score, spoof_score) with provisional 0.85 gates
in decision_policy.py. DeepFace FasNet.analyze returns only:

  (is_real, score) where score = sum(softmax_m1 + softmax_m2)[argmax] / 2

i.e. an uncalibrated *winning-class* confidence in (0, 1], not dual REAL/SPOOF
probabilities and not interchangeable with mn3 scores.

No local FasNet REAL / PRINT / SCREEN score corpus has been established in this
repo. RESEARCH_CANDIDATE_* values mirror the *numeric structure* of the mn3
provisional demo gates for shadow comparison only — they are NOT calibrated,
NOT production-safe, and must not be treated as validated FasNet thresholds.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

from fasnet_liveness import FasNetClass, FasNetResult


class FasNetLivenessState(str, enum.Enum):
    REAL = "REAL"
    SPOOF = "SPOOF"
    UNCERTAIN = "UNCERTAIN"
    ERROR = "ERROR"


class FasNetPolicyMode(str, enum.Enum):
    """How the confidence policy interacts with authentication."""

    # Auth uses native argmax only (historical FasNet friend path).
    BINARY_NATIVE = "binary_native"
    # Auth stays binary_native; three-state is diagnostic/shadow only.
    SHADOW = "shadow"
    # Three-state affects auth. UNCERTAIN → CHALLENGE_REQUIRED (non-accepting).
    # Explicit opt-in; thresholds remain RESEARCH / unvalidated.
    ACTIVE_RESEARCH = "active_research"


POLICY_STATUS_NOT_ESTABLISHED = "NOT_ESTABLISHED"
POLICY_STATUS_RESEARCH_CANDIDATE = "RESEARCH_CANDIDATE_UNVALIDATED"

# Shadow / research-only candidate gates (NOT validated on FasNet).
# Chosen only as a structural parallel to mn3 ProvisionalDecisionPolicy(0.85, 0.85)
# so operators can compare behavior — not as claimed FasNet operating points.
RESEARCH_CANDIDATE_REAL_MIN = 0.85
RESEARCH_CANDIDATE_SPOOF_MIN = 0.85

RESEARCH_CANDIDATE_NOTE = (
    "FasNet winning-class score gates are RESEARCH_CANDIDATE_UNVALIDATED. "
    "No FasNet REAL/PRINT/SCREEN calibration set exists in-repo. "
    "Values 0.85/0.85 mirror mn3 provisional demo structure only; "
    "mn3≠FasNet score semantics. Not production-safe."
)


@dataclass(frozen=True)
class FasNetConfidencePolicy:
    """Confidence gates for FasNet winning-class scores."""

    confident_real_min: float
    confident_spoof_min: float
    status: str
    mode: FasNetPolicyMode
    note: str


def default_shadow_policy() -> FasNetConfidencePolicy:
    """Default: auth binary; research candidates evaluated in shadow only."""
    return FasNetConfidencePolicy(
        confident_real_min=RESEARCH_CANDIDATE_REAL_MIN,
        confident_spoof_min=RESEARCH_CANDIDATE_SPOOF_MIN,
        status=POLICY_STATUS_RESEARCH_CANDIDATE,
        mode=FasNetPolicyMode.SHADOW,
        note=RESEARCH_CANDIDATE_NOTE,
    )


def binary_native_policy() -> FasNetConfidencePolicy:
    """No score gates for auth; optional shadow still uses research candidates."""
    return FasNetConfidencePolicy(
        confident_real_min=RESEARCH_CANDIDATE_REAL_MIN,
        confident_spoof_min=RESEARCH_CANDIDATE_SPOOF_MIN,
        status=POLICY_STATUS_RESEARCH_CANDIDATE,
        mode=FasNetPolicyMode.BINARY_NATIVE,
        note=RESEARCH_CANDIDATE_NOTE,
    )


def active_research_policy(
    *,
    confident_real_min: float = RESEARCH_CANDIDATE_REAL_MIN,
    confident_spoof_min: float = RESEARCH_CANDIDATE_SPOOF_MIN,
) -> FasNetConfidencePolicy:
    """Explicit opt-in: three-state drives auth (UNCERTAIN never ACCEPT)."""
    return FasNetConfidencePolicy(
        confident_real_min=confident_real_min,
        confident_spoof_min=confident_spoof_min,
        status=POLICY_STATUS_RESEARCH_CANDIDATE,
        mode=FasNetPolicyMode.ACTIVE_RESEARCH,
        note=RESEARCH_CANDIDATE_NOTE + " ACTIVE_RESEARCH opted in.",
    )


def classify_fasnet_winning_score(
    predicted_class: FasNetClass | str | None,
    winning_score: float | None,
    policy: FasNetConfidencePolicy,
) -> FasNetLivenessState:
    """Map FasNet native (class, winning-score) → REAL / SPOOF / UNCERTAIN / ERROR."""
    if predicted_class is None:
        return FasNetLivenessState.ERROR
    if isinstance(predicted_class, str):
        try:
            predicted_class = FasNetClass(predicted_class)
        except ValueError:
            return FasNetLivenessState.ERROR
    if predicted_class == FasNetClass.ERROR:
        return FasNetLivenessState.ERROR
    if winning_score is None:
        return FasNetLivenessState.UNCERTAIN

    if predicted_class == FasNetClass.REAL:
        if winning_score >= policy.confident_real_min:
            return FasNetLivenessState.REAL
        return FasNetLivenessState.UNCERTAIN

    if predicted_class == FasNetClass.SPOOF:
        if winning_score >= policy.confident_spoof_min:
            return FasNetLivenessState.SPOOF
        return FasNetLivenessState.UNCERTAIN

    return FasNetLivenessState.UNCERTAIN


def classify_fasnet_result(
    fasnet: FasNetResult | None,
    policy: FasNetConfidencePolicy,
) -> FasNetLivenessState:
    if fasnet is None:
        return FasNetLivenessState.ERROR
    return classify_fasnet_winning_score(
        fasnet.predicted_class,
        fasnet.antispoof_score,
        policy,
    )


def native_binary_state(fasnet: FasNetResult | None) -> FasNetLivenessState:
    """Historical friend-path mapping: argmax only (no score gate)."""
    if fasnet is None:
        return FasNetLivenessState.ERROR
    if fasnet.predicted_class == FasNetClass.ERROR or fasnet.is_real is None:
        return FasNetLivenessState.ERROR
    if fasnet.is_real is True or fasnet.predicted_class == FasNetClass.REAL:
        return FasNetLivenessState.REAL
    if fasnet.is_real is False or fasnet.predicted_class == FasNetClass.SPOOF:
        return FasNetLivenessState.SPOOF
    return FasNetLivenessState.UNCERTAIN


@dataclass(frozen=True)
class FasNetPolicyEvaluation:
    """Native prediction + confidence policy view (auth and/or shadow)."""

    native_class: str | None
    native_winning_score: float | None
    auth_liveness_state: FasNetLivenessState
    shadow_liveness_state: FasNetLivenessState
    policy_mode: FasNetPolicyMode
    policy_status: str
    confident_real_min: float
    confident_spoof_min: float
    policy_note: str
    challenge_would_be_required: bool


def evaluate_fasnet_policy(
    fasnet: FasNetResult | None,
    policy: FasNetConfidencePolicy,
) -> FasNetPolicyEvaluation:
    native = native_binary_state(fasnet)
    scored = classify_fasnet_result(fasnet, policy)

    if policy.mode == FasNetPolicyMode.ACTIVE_RESEARCH:
        auth_state = scored
    else:
        auth_state = native

    native_cls = None
    score = None
    if fasnet is not None:
        native_cls = fasnet.predicted_class.value
        score = fasnet.antispoof_score

    # Diagnostic: under the research candidate gates, would a challenge be needed?
    challenge_diag = scored == FasNetLivenessState.UNCERTAIN

    return FasNetPolicyEvaluation(
        native_class=native_cls,
        native_winning_score=score,
        auth_liveness_state=auth_state,
        shadow_liveness_state=scored,
        policy_mode=policy.mode,
        policy_status=policy.status,
        confident_real_min=policy.confident_real_min,
        confident_spoof_min=policy.confident_spoof_min,
        policy_note=policy.note,
        challenge_would_be_required=challenge_diag,
    )
