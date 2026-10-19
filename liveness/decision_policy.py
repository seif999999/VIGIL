"""Provisional demo liveness outcomes (uncalibrated thresholds; not authorization)."""

from __future__ import annotations

import enum
from dataclasses import dataclass

from import_paths import prepare_liveness_imports

prepare_liveness_imports()

from detector import LivenessResult

PROVISIONAL_POLICY_NOTE = (
    "Provisional policy (uncalibrated thresholds) — not final identity authorization"
)


class ProvisionalDemoOutcome(str, enum.Enum):
    NO_FACE = "NO_FACE"
    MULTIPLE_FACES = "MULTIPLE_FACES"
    REAL = "REAL"
    SPOOF = "SPOOF"
    UNCERTAIN_REVIEW = "UNCERTAIN_REVIEW"


@dataclass(frozen=True)
class ProvisionalDecisionPolicy:
    """Score gates for confident REAL/SPOOF; otherwise UNCERTAIN_REVIEW."""

    confident_real_min: float = 0.85
    confident_spoof_min: float = 0.85


def classify_provisional_scores(
    real_score: float,
    spoof_score: float,
    policy: ProvisionalDecisionPolicy,
) -> ProvisionalDemoOutcome:
    if real_score >= policy.confident_real_min and real_score > spoof_score:
        return ProvisionalDemoOutcome.REAL
    if spoof_score >= policy.confident_spoof_min and spoof_score >= real_score:
        return ProvisionalDemoOutcome.SPOOF
    return ProvisionalDemoOutcome.UNCERTAIN_REVIEW


def resolve_demo_outcome(
    face_decision_value: str,
    liveness: LivenessResult | None,
    policy: ProvisionalDecisionPolicy,
) -> ProvisionalDemoOutcome:
    if face_decision_value == "NO_FACE":
        return ProvisionalDemoOutcome.NO_FACE
    if face_decision_value == "MULTIPLE_FACES":
        return ProvisionalDemoOutcome.MULTIPLE_FACES
    if liveness is None:
        return ProvisionalDemoOutcome.UNCERTAIN_REVIEW
    return classify_provisional_scores(
        liveness.real_score,
        liveness.spoof_score,
        policy,
    )
