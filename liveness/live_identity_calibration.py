"""Standalone FaceNet identity calibration (camera) — research only.

Runs SCRFD + FaceNet on every usable single-face frame, independent of FasNet.
Does not modify live authentication, FasNet, Challenge 1/2, enrollment, or models.
Does not change MATCH_THRESHOLD (9.0).

Controls:
  M     — toggle MARKED recording (only marked evals are written to JSONL)
  Q/ESC — stop, flush session, exit

See IDENTITY_CALIBRATION.md for the physical-test procedure.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from face_bridge import load_face_detector_module
from facenet_identity import (
    DEFAULT_ENROLLMENT_PATH,
    MATCH_THRESHOLD,
    MATCH_THRESHOLD_STATUS,
    FaceNetIdentity,
    build_enrollment_index,
    identity_embedding_count,
    load_enrollments,
)
from identity_calibration_record import (
    DEFAULT_IDENTITY_CALIBRATION_DIR,
    VALID_LABELS,
    IdentityCalibrationRecorder,
    hud_status_line,
)
from live_friend_auth import _open_camera
from live_runtime import (
    InterruptFlag,
    QUIT_KEY_CODES,
    frame_is_usable,
    poll_key_code,
    preview_window_closed,
    run_with_interrupt_guard,
)

WINDOW = "VIGIL Identity Calibration"


def _draw_hud(
    frame: np.ndarray,
    *,
    rec: IdentityCalibrationRecorder,
    face_decision: str,
    face_count: int,
    best_distance: float | None,
    recognized: bool | None,
    candidate: str | None,
    facenet_error: str | None,
    camera_state: str,
) -> None:
    h, w = frame.shape[:2]
    y = 28
    lines = [
        hud_status_line(rec),
        f"camera={camera_state}  faces={face_count}  {face_decision}",
        f"threshold={MATCH_THRESHOLD:.1f} ({MATCH_THRESHOLD_STATUS})",
    ]
    if facenet_error:
        lines.append(f"FaceNet ERROR: {facenet_error}")
    elif best_distance is not None:
        decision = "RECOGNIZED" if recognized else "UNKNOWN"
        cand = candidate or "—"
        lines.append(f"L2 distance={best_distance:.3f}  →  {decision}  cand={cand}")
    elif face_decision == "FACE_OK":
        lines.append("FaceNet: waiting…")
    else:
        lines.append("FaceNet: skipped (need single FACE_OK)")

    lines.append("M=mark/pause   Q/ESC=quit   (no images/embeddings saved)")

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 20 + 22 * len(lines)), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    for i, text in enumerate(lines):
        color = (220, 220, 220)
        if "RECOGNIZED" in text:
            color = (80, 220, 80)
        elif "UNKNOWN" in text:
            color = (80, 80, 240)
        elif "MARKED" in text:
            color = (80, 200, 255)
        cv2.putText(
            frame,
            text,
            (10, y + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            1,
            cv2.LINE_AA,
        )


def _crop_face(frame_bgr: np.ndarray, face: Any) -> np.ndarray | None:
    x1, y1, x2, y2 = int(face.x1), int(face.y1), int(face.x2), int(face.y2)
    h, w = frame_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop


def create_scrfd_detector() -> Any:
    """SCRFD entry point shared with live_friend_auth (`face_mod.FaceDetector()`)."""
    face_mod = load_face_detector_module()
    return face_mod.FaceDetector()


def run_session(
    args: argparse.Namespace,
    interrupt: InterruptFlag | None = None,
) -> int:
    enroll_path = Path(args.enrollment_db)
    db = load_enrollments(enroll_path)
    identities = db.get("identities") or {}
    n_ids = len(identities)
    n_emb = sum(identity_embedding_count(db, i) for i in identities)
    if n_emb <= 0:
        print(
            f"[FAIL] No enrolled embeddings in {enroll_path}. "
            "Run enroll_identity.py first.",
            file=sys.stderr,
        )
        return 2

    face_detector = create_scrfd_detector()
    facenet = FaceNetIdentity(backend=args.facenet_backend)
    print(f"[INFO] FaceNet backend={facenet.backend}; warming up…")
    facenet.warm_up()
    enrollment_index = build_enrollment_index(db)

    rec = IdentityCalibrationRecorder.create(
        label=args.label,
        camera_index=args.camera,
        subject_id=args.subject_id,
        out_dir=Path(args.out_dir) if args.out_dir else None,
        facenet_backend=facenet.backend,
        enrollment_path=str(enroll_path.resolve()),
        n_enrolled_identities=n_ids,
        n_enrolled_embeddings=n_emb,
    )
    print(f"[INFO] Session file: {rec.out_path}")
    print(
        f"[INFO] label={rec.label} subject_id={rec.subject_id} "
        f"gallery identities={n_ids} embeddings={n_emb} "
        f"threshold={MATCH_THRESHOLD} ({MATCH_THRESHOLD_STATUS})"
    )
    print("[INFO] Press M to MARK, Q/ESC to quit. FaceNet runs without FasNet.")

    # Same InterruptFlag API as live_friend_auth: ``is_set`` is a bool property.
    if interrupt is None:
        interrupt = InterruptFlag()
        interrupt.install()
    cap = _open_camera(args.camera)
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    last_eval_t = 0.0
    eval_interval_s = max(0.05, float(args.min_interval_ms) / 1000.0)
    fps_t0 = time.perf_counter()
    fps_n = 0
    preview_fps: float | None = None
    eval_times: list[float] = []
    camera_state = "frames_ok"

    face_decision = "NO_FACE"
    face_count = 0
    best_distance: float | None = None
    recognized: bool | None = None
    candidate: str | None = None
    facenet_error: str | None = None
    box_xyxy: tuple[int, int, int, int] | None = None

    try:
        while not interrupt.is_set:
            ok, frame = cap.read()
            if not ok or frame is None or not frame_is_usable(frame):
                camera_state = "waiting_for_frames"
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                _draw_hud(
                    blank,
                    rec=rec,
                    face_decision="NO_FRAME",
                    face_count=0,
                    best_distance=None,
                    recognized=None,
                    candidate=None,
                    facenet_error=None,
                    camera_state=camera_state,
                )
                cv2.imshow(WINDOW, blank)
            else:
                camera_state = "frames_ok"
                fps_n += 1
                now = time.perf_counter()
                if now - fps_t0 >= 1.0:
                    preview_fps = fps_n / (now - fps_t0)
                    fps_n = 0
                    fps_t0 = now

                if now - last_eval_t >= eval_interval_s:
                    last_eval_t = now
                    t_scrfd0 = time.perf_counter()
                    det = face_detector.detect(frame)
                    scrfd_ms = (time.perf_counter() - t_scrfd0) * 1000.0
                    face_count = int(det.face_count)
                    face_decision = det.decision.value

                    best_distance = None
                    recognized = None
                    candidate = None
                    facenet_error = None
                    facenet_ms: float | None = None
                    box_xyxy = None

                    if face_decision == "FACE_OK" and face_count == 1 and det.faces:
                        f0 = det.faces[0]
                        box_xyxy = (int(f0.x1), int(f0.y1), int(f0.x2), int(f0.y2))
                        crop = _crop_face(frame, f0)
                        if crop is None:
                            facenet_error = "empty_face_crop"
                            recognized = False
                        else:
                            match = facenet.match_crop(
                                crop,
                                db,
                                threshold=MATCH_THRESHOLD,
                                enrollment_index=enrollment_index,
                            )
                            facenet_ms = match.inference_ms
                            facenet_error = match.error
                            best_distance = match.best_distance
                            recognized = (
                                bool(match.recognized) if match.error is None else False
                            )
                            candidate = match.best_identity_candidate
                    # FaceNet intentionally skipped for NO_FACE / MULTIPLE_FACES

                    eval_hz = None
                    eval_times.append(now)
                    eval_times = [t for t in eval_times if now - t <= 2.0]
                    if len(eval_times) >= 2:
                        span = eval_times[-1] - eval_times[0]
                        if span > 0:
                            eval_hz = (len(eval_times) - 1) / span

                    rec.record_eval(
                        face_count=face_count,
                        face_decision=face_decision,
                        best_distance=best_distance,
                        recognized=recognized,
                        best_identity_candidate=candidate,
                        threshold=MATCH_THRESHOLD,
                        threshold_status=MATCH_THRESHOLD_STATUS,
                        facenet_error=facenet_error,
                        scrfd_ms=scrfd_ms,
                        facenet_ms=facenet_ms,
                        camera_state=camera_state,
                        preview_fps=preview_fps,
                        eval_hz=eval_hz,
                    )

                display = frame.copy()
                if box_xyxy is not None:
                    x1, y1, x2, y2 = box_xyxy
                    cv2.rectangle(
                        display,
                        (x1, y1),
                        (x2, y2),
                        (80, 220, 80) if recognized else (80, 80, 240),
                        2,
                    )
                _draw_hud(
                    display,
                    rec=rec,
                    face_decision=face_decision,
                    face_count=face_count,
                    best_distance=best_distance,
                    recognized=recognized,
                    candidate=candidate,
                    facenet_error=facenet_error,
                    camera_state=camera_state,
                )
                cv2.imshow(WINDOW, display)

            key = poll_key_code(1)
            if key in QUIT_KEY_CODES:
                break
            if key in (ord("m"), ord("M")):
                on = rec.toggle_marked()
                print(
                    f"[REC] marked={on} saved_so_far={rec.marked_records} "
                    f"file={rec.out_path.name}"
                )
            if preview_window_closed(WINDOW):
                break
    finally:
        try:
            cap.release()
        except Exception:  # noqa: BLE001
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:  # noqa: BLE001
            pass
        rec.close()
        print(
            f"[INFO] Closed session marked_records={rec.marked_records} "
            f"unmarked_seen={rec.unmarked_seen} -> {rec.out_path}"
        )
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Physical FaceNet identity calibration (GENUINE / IMPOSTOR). "
            "Independent of FasNet. Threshold 9.0 unchanged."
        )
    )
    p.add_argument(
        "--label",
        required=True,
        choices=VALID_LABELS,
        help="Session ground-truth cohort: GENUINE (enrolled user) or IMPOSTOR",
    )
    p.add_argument(
        "--subject-id",
        required=True,
        help=(
            "Anonymous subject/session tag (e.g. genuine_A, impostor_01). "
            "Do not use legal names if you want files to stay anonymous."
        ),
    )
    p.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    p.add_argument(
        "--enrollment-db",
        type=Path,
        default=DEFAULT_ENROLLMENT_PATH,
        help="Existing enrollments.json (read-only for this tool)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_IDENTITY_CALIBRATION_DIR,
        help="Directory for session_*.jsonl (default: data/identity_calibration)",
    )
    p.add_argument(
        "--facenet-backend",
        default="auto",
        choices=("onnx", "tf", "auto"),
        help="FaceNet backend (default: auto / ONNX when available)",
    )
    p.add_argument(
        "--min-interval-ms",
        type=float,
        default=100.0,
        help="Minimum ms between FaceNet evaluations (preview stays continuous)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.facenet_backend == "auto":
        args.facenet_backend = None

    code = 0
    interrupt = InterruptFlag()
    interrupt.install()

    def _run() -> None:
        nonlocal code
        code = run_session(args, interrupt)

    run_with_interrupt_guard(_run, interrupt)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
