"""FaceNet identity calibration session recording (distances/metadata only).

Physical-test workflow (see IDENTITY_CALIBRATION.md):
  1. Launch live_identity_calibration.py with --label GENUINE|IMPOSTOR
  2. Wait until camera is stable
  3. Press M to start the MARKED interval (only marked evals are saved)
  4. Press M to pause between subjects / pose changes
  5. Press M to resume; Q/ESC ends and flushes the session file

Does not store camera images or biometric embeddings.
Does not change MATCH_THRESHOLD (9.0) or the matching algorithm.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from facenet_identity import MATCH_THRESHOLD, MATCH_THRESHOLD_STATUS

VALID_LABELS = ("GENUINE", "IMPOSTOR")

DEFAULT_IDENTITY_CALIBRATION_DIR = (
    Path(__file__).resolve().parent / "data" / "identity_calibration"
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class IdentityCalibrationRecorder:
    """Thread-safe JSONL recorder; writes only while marked=True."""

    label: str
    out_path: Path
    camera_index: int
    session_id: str
    subject_id: str
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
        subject_id: str,
        out_dir: Path | None = None,
        session_note: str | None = None,
        facenet_backend: str | None = None,
        enrollment_path: str | None = None,
        n_enrolled_identities: int | None = None,
        n_enrolled_embeddings: int | None = None,
    ) -> IdentityCalibrationRecorder:
        label = label.upper().strip()
        if label not in VALID_LABELS:
            raise ValueError(f"label must be one of {VALID_LABELS}, got {label!r}")
        sid = (subject_id or "").strip()
        if not sid:
            raise ValueError("subject_id must be a non-empty anonymous identifier")
        if any(ch in sid for ch in ("/", "\\", "\n", "\r")):
            raise ValueError("subject_id must not contain path separators or newlines")

        out_dir = out_dir or DEFAULT_IDENTITY_CALIBRATION_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        session_id = f"{label}_{sid}_{stamp}"
        path = out_dir / f"session_{session_id}.jsonl"
        if path.is_file():
            raise FileExistsError(f"Identity calibration session already exists: {path}")

        rec = cls(
            label=label,
            out_path=path,
            camera_index=camera_index,
            session_id=session_id,
            subject_id=sid,
        )
        rec._fh = path.open("w", encoding="utf-8")
        meta = {
            "type": "session_meta",
            "session_id": session_id,
            "label": label,
            "subject_id": sid,
            "camera_index": camera_index,
            "started_at_utc": utc_now_iso(),
            "facenet_backend": facenet_backend,
            "enrollment_path": enrollment_path,
            "n_enrolled_identities": n_enrolled_identities,
            "n_enrolled_embeddings": n_enrolled_embeddings,
            "match_threshold": MATCH_THRESHOLD,
            "match_threshold_status": MATCH_THRESHOLD_STATUS,
            "score_semantics": (
                "best_distance = min L2 Euclidean distance in FaceNet-128 embedding "
                "space to any enrolled embedding for the nearest identity; "
                "recognized = (best_distance < match_threshold). "
                "NOT a probability and NOT cosine similarity."
            ),
            "stores_images": False,
            "stores_embeddings": False,
            "thresholds_activated": False,
            "note": session_note
            or (
                "Marked-interval FaceNet identity calibration. Press M to toggle "
                "MARKED. No images or embeddings stored. Threshold 9.0 unchanged "
                "(UNVALIDATED_FRIEND_THRESHOLD)."
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
                    "subject_id": self.subject_id,
                    "label": self.label,
                }
            )
            return self.marked

    def record_eval(
        self,
        *,
        face_count: int,
        face_decision: str,
        best_distance: float | None,
        recognized: bool | None,
        best_identity_candidate: str | None,
        threshold: float,
        threshold_status: str,
        facenet_error: str | None,
        scrfd_ms: float | None,
        facenet_ms: float | None,
        camera_state: str,
        preview_fps: float | None,
        eval_hz: float | None,
    ) -> bool:
        """Append one evaluation if marked. Never writes embeddings or images."""
        with self._lock:
            if self._closed:
                return False
            if not self.marked:
                self.unmarked_seen += 1
                return False

            row: dict[str, Any] = {
                "type": "eval",
                "timestamp_utc": utc_now_iso(),
                "session_id": self.session_id,
                "label": self.label,
                "subject_id": self.subject_id,
                "marked": True,
                "camera_state": camera_state,
                "preview_fps": preview_fps,
                "eval_hz": eval_hz,
                "face_count": int(face_count),
                "face_decision": face_decision,
                "best_distance": best_distance,
                "recognized": recognized,
                "best_identity_candidate": best_identity_candidate,
                "match_threshold": float(threshold),
                "match_threshold_status": threshold_status,
                "facenet_error": facenet_error,
                "scrfd_ms": scrfd_ms,
                "facenet_ms": facenet_ms,
            }
            self._write_line(row)
            self.marked_records += 1
            return True

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._write_line(
                {
                    "type": "session_end",
                    "timestamp_utc": utc_now_iso(),
                    "session_id": self.session_id,
                    "subject_id": self.subject_id,
                    "label": self.label,
                    "marked_records": self.marked_records,
                    "unmarked_evals_seen": self.unmarked_seen,
                    "final_marked": self.marked,
                }
            )
            if self._fh is not None:
                self._fh.close()
                self._fh = None
            self._closed = True


def hud_status_line(rec: IdentityCalibrationRecorder) -> str:
    state = "MARKED" if rec.marked else "PAUSED"
    return (
        f"IDCAL {rec.label} subj={rec.subject_id} {state} n={rec.marked_records}"
    )
