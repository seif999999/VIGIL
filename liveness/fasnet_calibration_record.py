"""Labeled FasNet confidence session recording (scores/metadata only — no images).

Physical-test workflow:
  1. Launch live_friend_auth with --record-label REAL|PRINT|SCREEN
  2. Wait until camera is stable
  3. Press M to start the MARKED interval (only marked evals are saved)
  4. Press M again to pause before transitions / attack media swaps
  5. Press M to resume; Q/ESC ends and flushes the session file

Default auth remains shadow mode. Recordings are research-only JSONL under
liveness/data/fasnet_calibration/ (gitignored with the rest of liveness/data/).
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from auth_pipeline import AuthPipelineResult
from fasnet_decision_policy import (
    RESEARCH_CANDIDATE_REAL_MIN,
    RESEARCH_CANDIDATE_SPOOF_MIN,
    default_shadow_policy,
    evaluate_fasnet_policy,
)

VALID_LABELS = ("REAL", "PRINT", "SCREEN")

DEFAULT_CALIBRATION_DIR = (
    Path(__file__).resolve().parent / "data" / "fasnet_calibration"
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class CalibrationSessionRecorder:
    """Thread-safe JSONL recorder; writes only while marked=True."""

    label: str
    out_path: Path
    camera_index: int
    session_id: str
    marked: bool = False
    marked_records: int = 0
    unmarked_seen: int = 0
    _lock: threading.Lock = threading.Lock()
    _fh: Any = None
    _closed: bool = False

    @classmethod
    def create(
        cls,
        *,
        label: str,
        camera_index: int,
        out_dir: Path | None = None,
        session_note: str | None = None,
        fasnet_backend: str | None = None,
        facenet_backend: str | None = None,
        auth_policy_mode: str | None = None,
    ) -> CalibrationSessionRecorder:
        label = label.upper().strip()
        if label not in VALID_LABELS:
            raise ValueError(f"label must be one of {VALID_LABELS}, got {label!r}")
        out_dir = out_dir or DEFAULT_CALIBRATION_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        session_id = f"{label}_{stamp}"
        path = out_dir / f"session_{session_id}.jsonl"
        # Never overwrite an existing session file (preserve prior evidence).
        if path.is_file():
            raise FileExistsError(f"Calibration session already exists: {path}")
        rec = cls(
            label=label,
            out_path=path,
            camera_index=camera_index,
            session_id=session_id,
        )
        rec._fh = path.open("w", encoding="utf-8")
        meta = {
            "type": "session_meta",
            "session_id": session_id,
            "label": label,
            "camera_index": camera_index,
            "started_at_utc": utc_now_iso(),
            "fasnet_backend": fasnet_backend,
            "facenet_backend": facenet_backend,
            "auth_policy_mode": auth_policy_mode or "shadow",
            "note": session_note
            or (
                "Marked-interval recording only. Press M to toggle MARKED. "
                "No facial images stored. Auth policy remains shadow by default "
                "(candidate thresholds compared offline only)."
            ),
            "research_candidate_real_min": RESEARCH_CANDIDATE_REAL_MIN,
            "research_candidate_spoof_min": RESEARCH_CANDIDATE_SPOOF_MIN,
            "thresholds_activated": False,
            "score_not_calibrated_probability": True,
            "fasnet_score_semantics": (
                "winning_score=sum_softmax[argmax]/2 (uncalibrated winning-class "
                "confidence, NOT a calibrated probability); "
                "class_scores_avg=ensemble_prediction/2; index1=REAL; "
                "real_score_avg=avg[1]; spoof_score_combined=avg[0]+avg[2]"
            ),
        }
        rec._write_line(meta)
        return rec

    def _write_line(self, obj: dict[str, Any]) -> None:
        if self._fh is None or self._closed:
            return
        self._fh.write(json.dumps(obj, ensure_ascii=True) + "\n")
        self._fh.flush()

    def toggle_marked(self) -> bool:
        with self._lock:
            self.marked = not self.marked
            self._write_line(
                {
                    "type": "mark_event",
                    "timestamp_utc": utc_now_iso(),
                    "marked": self.marked,
                    "session_id": self.session_id,
                }
            )
            return self.marked

    def record_eval(
        self,
        result: AuthPipelineResult,
        *,
        camera_state: str,
        preview_fps: float | None,
        auth_hz: float | None,
        wall_ms: float | None = None,
    ) -> bool:
        """Append one auth evaluation if currently marked. Returns True if saved."""
        with self._lock:
            if self._closed:
                return False
            if not self.marked:
                self.unmarked_seen += 1
                return False

            fas = result.fasnet
            ident = result.identity
            outcome = result.outcome
            # Always recompute shadow view for logging consistency
            policy = default_shadow_policy()
            shadow = evaluate_fasnet_policy(fas, policy)

            row: dict[str, Any] = {
                "type": "eval",
                "timestamp_utc": utc_now_iso(),
                "session_id": self.session_id,
                "label": self.label,
                "marked": True,
                "camera_state": camera_state,
                "preview_fps": preview_fps,
                "auth_hz": auth_hz,
                "face_count": int(result.detection.face_count),
                "face_decision": result.detection.decision.value,
                "fasnet_native_class": (
                    fas.predicted_class.value if fas is not None else None
                ),
                "fasnet_winning_score": (
                    fas.antispoof_score if fas is not None else None
                ),
                "fasnet_is_real": fas.is_real if fas is not None else None,
                "fasnet_class_scores_avg": (
                    list(fas.class_scores_avg)
                    if fas is not None and fas.class_scores_avg is not None
                    else None
                ),
                "fasnet_real_score_avg": (
                    fas.real_score_avg if fas is not None else None
                ),
                "fasnet_spoof_score_combined": (
                    fas.spoof_score_combined if fas is not None else None
                ),
                "fasnet_error": fas.error if fas is not None else None,
                "shadow_liveness_state": shadow.shadow_liveness_state.value,
                "shadow_auth_liveness_state": shadow.auth_liveness_state.value,
                "shadow_real_min": shadow.confident_real_min,
                "shadow_spoof_min": shadow.confident_spoof_min,
                "shadow_policy_status": shadow.policy_status,
                "identity_recognized": (
                    bool(ident.recognized) if ident is not None else None
                ),
                "identity_name": outcome.identity,
                "facenet_distance": (
                    ident.best_distance if ident is not None else None
                ),
                "facenet_threshold": (
                    ident.threshold if ident is not None else None
                ),
                "facenet_error": ident.error if ident is not None else None,
                "auth_decision": outcome.decision.value,
                "auth_reason": outcome.reason,
                "challenge_1_status": outcome.challenge_1_status,
                "scrfd_ms": result.scrfd_ms,
                "fasnet_ms": result.fasnet_ms,
                "facenet_ms": result.facenet_ms,
                "total_ms": result.total_ms,
                "wall_ms": wall_ms,
            }
            self._write_line(row)
            self.marked_records += 1
            return True

    def close(self) -> Path:
        with self._lock:
            if not self._closed:
                self._write_line(
                    {
                        "type": "session_end",
                        "timestamp_utc": utc_now_iso(),
                        "session_id": self.session_id,
                        "marked_records": self.marked_records,
                        "unmarked_evals_seen": self.unmarked_seen,
                        "final_marked": self.marked,
                    }
                )
                if self._fh is not None:
                    self._fh.close()
                    self._fh = None
                self._closed = True
            return self.out_path


def status_line(rec: CalibrationSessionRecorder | None) -> str:
    if rec is None:
        return ""
    state = "MARKED" if rec.marked else "PAUSED"
    return f"REC {rec.label} {state} n={rec.marked_records}"
