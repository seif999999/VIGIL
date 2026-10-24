"""SCRFD 5-point landmark head-pose proxies for Challenge 1.

Uses InsightFace SCRFD-2.5G_KPS landmarks already produced by face detection:
  0 left_eye, 1 right_eye, 2 nose, 3 left_mouth, 4 right_mouth

Yaw/pitch are *relative geometric proxies* (not absolute Euler degrees from a
PnP solve). Thresholds are EMPIRICAL_UNVALIDATED and must be tuned on physical
tests — they are not claimed universal.

Coordinate convention (raw camera frame, not display-mirrored):
  +yaw  = nose shifts toward image-right relative to eye midline
          → subject turning their head LEFT (facing camera).
  +pitch = nose drops toward mouth relative to eye line
          → subject looking DOWN.

Instruction mapping uses *subject* left/right (not screen-mirrored labels).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

# Landmark indices (InsightFace SCRFD)
LEFT_EYE = 0
RIGHT_EYE = 1
NOSE = 2
LEFT_MOUTH = 3
RIGHT_MOUTH = 4


class HeadDirection(str, Enum):
    LOOK_LEFT = "LOOK_LEFT"
    LOOK_RIGHT = "LOOK_RIGHT"
    LOOK_UP = "LOOK_UP"
    LOOK_DOWN = "LOOK_DOWN"

    @property
    def spoken(self) -> str:
        return {
            HeadDirection.LOOK_LEFT: "Look left",
            HeadDirection.LOOK_RIGHT: "Look right",
            HeadDirection.LOOK_UP: "Look up",
            HeadDirection.LOOK_DOWN: "Look down",
        }[self]


# EMPIRICAL_UNVALIDATED — tune after physical Challenge 1 tests.
DEFAULT_YAW_DELTA = 0.12
DEFAULT_PITCH_DELTA = 0.10
DEFAULT_HOLD_MS = 250.0
DEFAULT_STEP_TIMEOUT_MS = 8000.0
DEFAULT_WRONG_RATIO = 0.85  # opposite-axis motion this large → fail


@dataclass(frozen=True)
class PoseProxies:
    yaw: float
    pitch: float
    interocular: float
    face_height: float


@dataclass(frozen=True)
class PoseThresholds:
    yaw_delta: float = DEFAULT_YAW_DELTA
    pitch_delta: float = DEFAULT_PITCH_DELTA
    hold_ms: float = DEFAULT_HOLD_MS
    step_timeout_ms: float = DEFAULT_STEP_TIMEOUT_MS
    wrong_ratio: float = DEFAULT_WRONG_RATIO
    status: str = "EMPIRICAL_UNVALIDATED"


def pose_from_landmarks(landmarks: np.ndarray) -> PoseProxies | None:
    """Return yaw/pitch proxies or None if landmarks are unusable."""
    lm = np.asarray(landmarks, dtype=np.float64)
    if lm.shape != (5, 2):
        return None
    if not np.isfinite(lm).all():
        return None
    left_eye = lm[LEFT_EYE]
    right_eye = lm[RIGHT_EYE]
    nose = lm[NOSE]
    left_mouth = lm[LEFT_MOUTH]
    right_mouth = lm[RIGHT_MOUTH]
    iod = float(np.linalg.norm(right_eye - left_eye))
    if iod < 1e-3:
        return None
    eye_mid = (left_eye + right_eye) * 0.5
    mouth_mid = (left_mouth + right_mouth) * 0.5
    face_h = float(np.linalg.norm(mouth_mid - eye_mid))
    if face_h < 1e-3:
        return None
    yaw = float((nose[0] - eye_mid[0]) / iod)
    pitch = float((nose[1] - eye_mid[1]) / face_h)
    return PoseProxies(yaw=yaw, pitch=pitch, interocular=iod, face_height=face_h)


def direction_deltas(
    baseline: PoseProxies,
    current: PoseProxies,
) -> tuple[float, float]:
    """Return (d_yaw, d_pitch) relative to pre-instruction baseline."""
    return current.yaw - baseline.yaw, current.pitch - baseline.pitch


def direction_progress(
    direction: HeadDirection,
    d_yaw: float,
    d_pitch: float,
    thr: PoseThresholds,
) -> tuple[float, bool, bool]:
    """
    Returns (progress 0..1+, reached, wrong_direction).

    Subject-centric:
      LOOK_LEFT  → +yaw (nose to image-right)
      LOOK_RIGHT → -yaw
      LOOK_DOWN  → +pitch
      LOOK_UP    → -pitch
    """
    if direction == HeadDirection.LOOK_LEFT:
        primary = d_yaw / thr.yaw_delta
        opposite = (-d_yaw) / thr.yaw_delta
        cross = abs(d_pitch) / thr.pitch_delta
    elif direction == HeadDirection.LOOK_RIGHT:
        primary = (-d_yaw) / thr.yaw_delta
        opposite = d_yaw / thr.yaw_delta
        cross = abs(d_pitch) / thr.pitch_delta
    elif direction == HeadDirection.LOOK_DOWN:
        primary = d_pitch / thr.pitch_delta
        opposite = (-d_pitch) / thr.pitch_delta
        cross = abs(d_yaw) / thr.yaw_delta
    else:  # LOOK_UP
        primary = (-d_pitch) / thr.pitch_delta
        opposite = d_pitch / thr.pitch_delta
        cross = abs(d_yaw) / thr.yaw_delta

    wrong = opposite >= thr.wrong_ratio and opposite > primary
    # Large cross-axis only wrong if primary not progressing
    if cross >= thr.wrong_ratio * 1.2 and primary < 0.4:
        wrong = True
    reached = primary >= 1.0 and not wrong
    return float(primary), reached, wrong
