"""HMAC-verified frame perception: SCRFD → FasNet → FaceNet → auth decision.

Call only after HMAC ACCEPT. Legacy anti-spoof-mn3 is not used here
(see pipeline.run_liveness_if_face_ok / detector.LivenessDetector).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from auth_decision import AuthOutcome, decide_auth
from facenet_identity import (
    DEFAULT_ENROLLMENT_PATH,
    MATCH_THRESHOLD,
    FaceNetIdentity,
    IdentityMatch,
    build_enrollment_index,
    load_enrollments,
)
from fasnet_decision_policy import (
    FasNetConfidencePolicy,
    FasNetLivenessState,
    FasNetPolicyMode,
    default_shadow_policy,
    evaluate_fasnet_policy,
)
from fasnet_liveness import FasNetLiveness, FasNetResult


@dataclass(frozen=True)
class AuthPipelineResult:
    detection: Any
    fasnet: FasNetResult | None
    identity: IdentityMatch | None
    outcome: AuthOutcome
    scrfd_ms: float
    fasnet_ms: float
    facenet_ms: float
    total_ms: float
    # Optional stage breakdown (profiling; unused by auth decisions)
    fasnet_v2_ms: float | None = None
    fasnet_v1se_ms: float | None = None
    fasnet_preprocess_ms: float | None = None
    match_ms: float | None = None


class FriendAuthPipeline:
    """Reusable models for steady-state inference (warm once)."""

    def __init__(
        self,
        face_detector,
        *,
        enrollment_path: Path | None = None,
        match_threshold: float = MATCH_THRESHOLD,
        fasnet: FasNetLiveness | None = None,
        facenet: FaceNetIdentity | None = None,
        fasnet_policy: FasNetConfidencePolicy | None = None,
    ) -> None:
        self.face_detector = face_detector
        self.enrollment_path = enrollment_path or DEFAULT_ENROLLMENT_PATH
        self.match_threshold = match_threshold
        self.fasnet = fasnet or FasNetLiveness()
        self.facenet = facenet or FaceNetIdentity()
        self.fasnet_policy = fasnet_policy or default_shadow_policy()
        self._enrollments: dict[str, Any] | None = None
        self._enrollment_index: dict[str, Any] | None = None

    def warm_up(self) -> dict[str, float | None]:
        self.fasnet.warm_up()
        self.facenet.warm_up()
        self.reload_enrollments()
        return {
            "fasnet_init_ms": self.fasnet.init_ms,
            "facenet_init_ms": self.facenet.init_ms,
        }

    def reload_enrollments(self) -> dict[str, Any]:
        self._enrollments = load_enrollments(self.enrollment_path)
        self._enrollment_index = build_enrollment_index(self._enrollments)
        return self._enrollments

    @property
    def enrollments(self) -> dict[str, Any]:
        if self._enrollments is None:
            self.reload_enrollments()
        assert self._enrollments is not None
        return self._enrollments

    @property
    def enrollment_index(self) -> dict[str, Any]:
        if self._enrollment_index is None:
            self.reload_enrollments()
        assert self._enrollment_index is not None
        return self._enrollment_index

    def process_verified_frame(self, frame_bgr: np.ndarray) -> AuthPipelineResult:
        t0 = time.perf_counter()
        det = self.face_detector.detect(frame_bgr)
        scrfd_ms = float(det.inference_ms)

        fasnet_result: FasNetResult | None = None
        identity: IdentityMatch | None = None
        fasnet_ms = 0.0
        facenet_ms = 0.0
        match_ms: float | None = None
        fasnet_v2_ms: float | None = None
        fasnet_v1se_ms: float | None = None
        fasnet_preprocess_ms: float | None = None

        if det.decision.value == "FACE_OK" and det.face_count == 1:
            face = det.faces[0]
            fasnet_result = self.fasnet.analyze_full_frame(frame_bgr, face.bbox)
            fasnet_ms = fasnet_result.inference_ms
            fasnet_v2_ms = fasnet_result.mini_v2_ms
            fasnet_v1se_ms = fasnet_result.mini_v1se_ms
            fasnet_preprocess_ms = fasnet_result.preprocess_ms

            # FaceNet when native REAL (binary/shadow auth) or when active
            # research policy maps to REAL/UNCERTAIN (identity before challenge).
            policy_ev = evaluate_fasnet_policy(fasnet_result, self.fasnet_policy)
            run_identity = False
            if fasnet_result.error is None:
                if self.fasnet_policy.mode == FasNetPolicyMode.ACTIVE_RESEARCH:
                    run_identity = policy_ev.auth_liveness_state in (
                        FasNetLivenessState.REAL,
                        FasNetLivenessState.UNCERTAIN,
                    )
                else:
                    run_identity = fasnet_result.is_real is True

            if run_identity:
                # Friend path: crop SCRFD/OpenCV bbox then FaceNet.represent
                x1, y1, x2, y2 = (
                    int(face.x1),
                    int(face.y1),
                    int(face.x2),
                    int(face.y2),
                )
                h, w = frame_bgr.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                crop = frame_bgr[y1:y2, x1:x2]
                if crop.size == 0:
                    identity = IdentityMatch(
                        recognized=False,
                        identity=None,
                        best_identity_candidate=None,
                        best_distance=None,
                        distances={},
                        threshold=self.match_threshold,
                        threshold_status="UNVALIDATED_FRIEND_THRESHOLD",
                        embedding_dim=0,
                        inference_ms=0.0,
                        error="empty_face_crop",
                    )
                else:
                    identity = self.facenet.match_crop(
                        crop,
                        self.enrollments,
                        threshold=self.match_threshold,
                        enrollment_index=self.enrollment_index,
                    )
                    facenet_ms = identity.inference_ms
                    # Enrollment compare is <<1 ms with the vectorized index; folded into facenet_ms.
                    match_ms = None

        outcome = decide_auth(
            face_decision_value=det.decision.value,
            face_count=det.face_count,
            fasnet=fasnet_result,
            identity=identity,
            fasnet_policy=self.fasnet_policy,
        )
        total_ms = (time.perf_counter() - t0) * 1000.0
        return AuthPipelineResult(
            detection=det,
            fasnet=fasnet_result,
            identity=identity,
            outcome=outcome,
            scrfd_ms=scrfd_ms,
            fasnet_ms=fasnet_ms,
            facenet_ms=facenet_ms,
            total_ms=total_ms,
            fasnet_v2_ms=fasnet_v2_ms,
            fasnet_v1se_ms=fasnet_v1se_ms,
            fasnet_preprocess_ms=fasnet_preprocess_ms,
            match_ms=match_ms,
        )


def run_fasnet_if_face_ok(
    frame_bgr: np.ndarray,
    detection_result,
    fasnet: FasNetLiveness | None,
) -> FasNetResult | None:
    """FasNet only on FACE_OK — parallel to pipeline.run_liveness_if_face_ok (mn3)."""
    if fasnet is None:
        return None
    decision = getattr(detection_result, "decision", None)
    if decision is None or getattr(decision, "value", decision) != "FACE_OK":
        return None
    if getattr(detection_result, "face_count", 0) != 1:
        return None
    face = detection_result.faces[0]
    return fasnet.analyze_full_frame(frame_bgr, face.bbox)
