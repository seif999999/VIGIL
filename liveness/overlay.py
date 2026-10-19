"""Draw liveness readout on a BGR frame."""

from __future__ import annotations

import cv2
import numpy as np

from import_paths import prepare_liveness_imports

prepare_liveness_imports()

from decision_policy import (
    PROVISIONAL_POLICY_NOTE,
    ProvisionalDecisionPolicy,
    resolve_demo_outcome,
)
from detector import LivenessResult, MODEL_NAME


def draw_demo_liveness_readout(
    frame_bgr: np.ndarray,
    *,
    face_decision_value: str,
    liveness: LivenessResult | None,
    policy: ProvisionalDecisionPolicy,
    total_ms: float | None = None,
) -> np.ndarray:
    """Live demo: face gate, raw scores, model argmax, provisional demo outcome."""
    out = frame_bgr.copy()
    demo = resolve_demo_outcome(face_decision_value, liveness, policy)
    lines = [
        PROVISIONAL_POLICY_NOTE,
        f"Demo outcome: {demo.value}",
        f"Face gate: {face_decision_value}",
    ]
    if liveness is not None:
        lines.extend(
            [
                f"Liveness: {MODEL_NAME}",
                f"Real score: {liveness.real_score:.4f}",
                f"Spoof score: {liveness.spoof_score:.4f}",
                f"Model class (argmax): {liveness.predicted_class.value}",
                f"Liveness infer: {liveness.inference_ms:.1f} ms",
            ]
        )
    else:
        lines.append(f"Liveness: {MODEL_NAME} (not run — face gate)")
    if total_ms is not None:
        lines.append(f"SCRFD+Liveness: {total_ms:.1f} ms")

    y = out.shape[0] - len(lines) * 22 - 8
    y = max(120, y)
    for line in lines:
        cv2.putText(
            out,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 200, 0),
            1,
            cv2.LINE_AA,
        )
        y += 22
    return out


def draw_liveness_overlay(
    frame_bgr: np.ndarray,
    liveness: LivenessResult | None,
    *,
    total_ms: float | None = None,
) -> np.ndarray:
    out = frame_bgr.copy()
    if liveness is None:
        return out

    lines = [
        f"Liveness: {MODEL_NAME}",
        f"Real score: {liveness.real_score:.4f}",
        f"Spoof score: {liveness.spoof_score:.4f}",
        f"Prediction: {liveness.predicted_class.value}",
        f"Liveness: {liveness.inference_ms:.1f} ms",
    ]
    if total_ms is not None:
        lines.append(f"SCRFD+Liveness: {total_ms:.1f} ms")

    y = out.shape[0] - len(lines) * 22 - 8
    y = max(120, y)
    for line in lines:
        cv2.putText(
            out,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 200, 0),
            1,
            cv2.LINE_AA,
        )
        y += 22
    return out
