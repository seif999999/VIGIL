"""Print provisional demo outcomes for fixed scores (no camera, no model)."""

from __future__ import annotations

import argparse

from decision_policy import ProvisionalDecisionPolicy, resolve_demo_outcome
from detector import LivenessResult, PredictedClass


def _live(real: float, spoof: float) -> LivenessResult:
    pred = PredictedClass.REAL if real > spoof else PredictedClass.SPOOF
    return LivenessResult(real, spoof, pred, 0.0)


CASES = {
    "confident-real": ("FACE_OK", _live(0.95, 0.05)),
    "confident-spoof": ("FACE_OK", _live(0.05, 0.95)),
    "uncertain": ("FACE_OK", _live(0.52, 0.48)),
    "no-face": ("NO_FACE", None),
    "multiple-faces": ("MULTIPLE_FACES", None),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show provisional demo outcome for a labelled scenario"
    )
    parser.add_argument(
        "--case",
        choices=tuple(CASES.keys()),
        required=True,
    )
    parser.add_argument("--provisional-real-min", type=float, default=0.85)
    parser.add_argument("--provisional-spoof-min", type=float, default=0.85)
    args = parser.parse_args()

    policy = ProvisionalDecisionPolicy(
        confident_real_min=args.provisional_real_min,
        confident_spoof_min=args.provisional_spoof_min,
    )
    face_decision, liveness = CASES[args.case]
    outcome = resolve_demo_outcome(face_decision, liveness, policy)
    print(f"case: {args.case}")
    print(f"face_gate: {face_decision}")
    if liveness is not None:
        print(f"real_score: {liveness.real_score:.4f}")
        print(f"spoof_score: {liveness.spoof_score:.4f}")
        print(f"model_class: {liveness.predicted_class.value}")
    print(f"provisional_demo_outcome: {outcome.value}")


if __name__ == "__main__":
    main()
