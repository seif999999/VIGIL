"""Live experimental auth: SCRFD → FasNet → FaceNet (no HMAC).

Default: continuous authentication on the latest camera frame (no SPACE required).
Optional --mode manual restores SPACE-to-score.

Models are warmed once and reused. Only one inference runs at a time; while busy,
preview stays live and the newest frame is kept for the next pass (drop stale).
"""

from __future__ import annotations

import argparse
import statistics
import sys
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from auth_decision import AuthDecision
from auth_pipeline import AuthPipelineResult, FriendAuthPipeline
from face_bridge import load_face_detector_module
from facenet_identity import DEFAULT_ENROLLMENT_PATH, MATCH_THRESHOLD, MATCH_THRESHOLD_STATUS
from fasnet_liveness import FasNetClass

# BGR colors
_GREEN = (40, 200, 40)
_RED = (40, 40, 220)
_AMBER = (0, 180, 255)
_WHITE = (240, 240, 240)
_PANEL = (24, 24, 24)


@dataclass
class _LatestFrame:
    frame: np.ndarray | None = None
    seq: int = 0


def _open_camera(index: int) -> cv2.VideoCapture:
    if hasattr(cv2, "CAP_DSHOW"):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            return cap
        cap.release()
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {index}")
    return cap


def _reason_label(reason: str) -> str:
    mapping = {
        "real_and_recognized": "Real and recognized",
        "spoof": "Spoof detected",
        "unknown_identity": "Unknown identity",
        "no_face": "No face detected",
        "multiple_faces": "Multiple faces detected",
        "face_gate_failed": "Face gate failed",
        "liveness_unavailable": "Liveness unavailable",
        "identity_unavailable": "Identity unavailable",
    }
    if reason in mapping:
        return mapping[reason]
    if reason.startswith("liveness_error"):
        return f"Liveness inference error ({reason.split(':', 1)[-1]})"
    if reason.startswith("identity_error"):
        return f"Identity inference error ({reason.split(':', 1)[-1]})"
    return reason


def _fasnet_lines(fas) -> list[str]:
    """
    DeepFace FasNet returns (is_real, score) where
      score = sum(softmax_model1 + softmax_model2)[argmax_label] / 2
    i.e. uncalibrated confidence for the *winning* class (REAL or SPOOF),
    not a separate REAL probability when the class is SPOOF.
    """
    if fas is None:
        return ["Liveness: N/A", "  (not run)"]
    if fas.predicted_class == FasNetClass.ERROR or fas.is_real is None:
        return [
            "Liveness: ERROR",
            f"  {fas.error or 'unknown'}",
            "  Check: FAIL",
        ]
    cls = fas.predicted_class.value
    score = fas.antispoof_score
    check = "PASS" if fas.is_real else "FAIL"
    # Label score as belonging to the predicted class only.
    if score is None:
        score_line = f"  {cls} class_conf: n/a (uncalibrated)"
    else:
        score_line = f"  {cls} class_conf: {score:.3f} (uncalibrated)"
    return [
        f"Liveness: {cls}",
        score_line,
        "  (DeepFace: winning-class score only; not dual REAL/SPOOF probs)",
        f"  Check: {check}",
    ]


def _put(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    color: tuple[int, int, int],
    scale: float = 0.55,
    thickness: int = 1,
) -> None:
    cv2.putText(
        img,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_auth(
    frame: np.ndarray,
    result: AuthPipelineResult | None,
    *,
    stale: bool,
    mode: str,
    busy: bool,
    age_ms: float | None,
    camera_state: str,
    preview_fps: float | None,
    auth_hz: float | None,
) -> None:
    h, w = frame.shape[:2]
    panel_w = min(420, max(300, w // 3))
    x0 = 8
    x1 = x0 + panel_w
    y0 = 8
    # Semi-opaque left panel (avoid covering center face when possible)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, h - 8), _PANEL, -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    def line(text: str, y: int, color=_WHITE, scale=0.48, thick=1) -> int:
        _put(frame, text[:58], (x0 + 10, y), color, scale, thick)
        return y + int(22 * max(0.9, scale / 0.48))

    # --- Camera / freshness banner (top-right, clear of left panel) ---
    cam_color = _GREEN if camera_state == "frames_ok" and not stale else _AMBER
    if camera_state == "unavailable":
        cam_color = _RED
    cam_label = f"Camera: {camera_state.upper()}"
    cam_x = w - 12 - int(len(cam_label) * 11 * 0.6)
    _put(frame, cam_label, (max(x1 + 8, cam_x), 28), cam_color, 0.6, 2)
    if preview_fps is not None:
        fps_label = f"Preview FPS: {preview_fps:.1f}"
        fps_x = w - 12 - int(len(fps_label) * 10 * 0.5)
        _put(frame, fps_label, (max(x1 + 8, fps_x), 54), _WHITE, 0.5, 1)

    if result is None or stale:
        decision = "STALE" if result is not None else "WAITING"
        if camera_state == "unavailable":
            decision = "DENY"
            reason = "Camera unavailable"
            color = _RED
        elif camera_state == "initializing":
            decision = "WAITING"
            reason = "Camera initializing"
            color = _AMBER
        else:
            reason = "Result stale / waiting for fresh inference"
            color = _AMBER
        y = y0 + 32
        y = line(decision, y, color, 1.1, 3)
        y = line(reason, y, color, 0.55, 1)
        if result is not None and stale:
            y = line(
                f"Last: {result.outcome.decision.value} / {_reason_label(result.outcome.reason)}",
                y,
                _AMBER,
                0.45,
                1,
            )
            y = line("(not current - do not treat as ACCEPT)", y, _AMBER, 0.45, 1)
        y = line(f"Infer: {'BUSY' if busy else 'idle'}  mode={mode}", y, _WHITE, 0.45)
        y = line("HMAC: N/A (standalone live tester)", y, _WHITE, 0.45)
        y = line("Q/ESC=quit", y, _WHITE, 0.45)
        return

    # Fresh result
    decision = result.outcome.decision.value
    reason = _reason_label(result.outcome.reason)
    if decision == "ACCEPT":
        color = _GREEN
    elif decision == "UNCERTAIN":
        color = _AMBER
    else:
        color = _RED

    det = result.detection
    # Bounding boxes for all faces
    for face in det.faces:
        cv2.rectangle(
            frame,
            (int(face.x1), int(face.y1)),
            (int(face.x2), int(face.y2)),
            color,
            2,
        )
        _put(
            frame,
            f"SCRFD {face.confidence:.2f}",
            (int(face.x1), max(20, int(face.y1) - 8)),
            color,
            0.5,
            1,
        )

    # Identity display name from match (never hardcode)
    id_match = result.identity
    if result.outcome.recognized and (result.outcome.identity or (id_match and id_match.display_name)):
        shown_name = (
            result.outcome.identity
            or (id_match.display_name if id_match else None)
            or "UNKNOWN"
        )
        recog = "MATCH"
        id_color = _GREEN
    else:
        shown_name = "UNKNOWN"
        recog = "NO MATCH"
        id_color = _RED if decision == "DENY" else _AMBER

    dist = id_match.best_distance if id_match is not None else None
    thr = id_match.threshold if id_match is not None else MATCH_THRESHOLD

    # Security checks
    face_check = (
        "PASS"
        if det.decision.value == "FACE_OK" and det.face_count == 1
        else "FAIL"
    )
    fas = result.fasnet
    if fas is None:
        live_check = "N/A"
    elif fas.is_real is True:
        live_check = "PASS"
    elif fas.is_real is False:
        live_check = "FAIL"
    else:
        live_check = "FAIL"
    id_check = (
        "PASS"
        if result.outcome.recognized
        else ("FAIL" if fas is not None and fas.is_real is True else "N/A")
    )
    final_check = "PASS" if decision == "ACCEPT" else "FAIL"

    y = y0 + 36
    y = line(decision, y, color, 1.15, 3)
    y = line(reason, y, color, 0.55, 2)
    y = line("", y)  # spacer via small advance
    y -= 10
    y = line(shown_name.upper(), y, id_color, 1.0, 3)
    y = line(f"Identity: {shown_name}", y, id_color, 0.55, 2)
    y = line(f"Recognition: {recog}", y, id_color, 0.5, 1)
    if dist is not None:
        y = line(f"FaceNet distance: {dist:.2f}", y, _WHITE, 0.5, 1)
    else:
        y = line("FaceNet distance: n/a", y, _WHITE, 0.5, 1)
    y = line(
        f"Threshold: {thr:.2f} [{MATCH_THRESHOLD_STATUS}]",
        y,
        _WHITE,
        0.45,
        1,
    )

    y = line("--- Face ---", y + 4, _WHITE, 0.5, 1)
    y = line(f"Faces: {det.face_count}  gate={det.decision.value}", y, _WHITE, 0.48)
    if det.faces:
        confs = ", ".join(f"{f.confidence:.2f}" for f in det.faces[:3])
        y = line(f"SCRFD conf: {confs}", y, _WHITE, 0.45)

    y = line("--- Liveness (FasNet) ---", y + 4, _WHITE, 0.5, 1)
    for fl in _fasnet_lines(fas):
        y = line(fl, y, _GREEN if live_check == "PASS" else (_RED if live_check == "FAIL" else _WHITE), 0.45)

    y = line("--- Security ---", y + 4, _WHITE, 0.5, 1)
    y = line(f"Face-count: {face_check}", y, _GREEN if face_check == "PASS" else _RED, 0.48)
    y = line(f"FasNet liveness: {live_check}", y, _GREEN if live_check == "PASS" else (_RED if live_check == "FAIL" else _AMBER), 0.48)
    y = line(f"FaceNet identity: {id_check}", y, _GREEN if id_check == "PASS" else (_RED if id_check == "FAIL" else _AMBER), 0.48)
    y = line(f"Final auth: {final_check}", y, _GREEN if final_check == "PASS" else _RED, 0.48)
    y = line("HMAC: N/A (standalone)", y, _WHITE, 0.45)

    y = line("--- Performance ---", y + 4, _WHITE, 0.5, 1)
    y = line(f"SCRFD:  {result.scrfd_ms:.0f} ms", y, _WHITE, 0.48)
    y = line(f"FasNet: {result.fasnet_ms:.0f} ms", y, _WHITE, 0.48)
    y = line(f"FaceNet:{result.facenet_ms:.0f} ms", y, _WHITE, 0.48)
    y = line(f"Total:  {result.total_ms:.0f} ms", y, _WHITE, 0.48)
    if auth_hz is not None:
        y = line(f"Auth rate: {auth_hz:.2f} Hz (evals/s)", y, _WHITE, 0.48)
    if preview_fps is not None:
        y = line(f"Preview:  {preview_fps:.1f} FPS", y, _WHITE, 0.48)
    if age_ms is not None:
        y = line(f"Result age: {age_ms:.0f} ms", y, _WHITE, 0.45)
    y = line(f"Infer: {'BUSY' if busy else 'idle'}  mode={mode}", y, _WHITE, 0.45)
    y = line("Q/ESC=quit" + ("  SPACE=score" if mode == "manual" else ""), y, _WHITE, 0.45)


def _session_bucket(result: AuthPipelineResult) -> str:
    reason = result.outcome.reason
    if result.outcome.decision == AuthDecision.ACCEPT:
        return "accept"
    if reason == "spoof":
        return "spoof"
    if reason == "unknown_identity":
        return "unknown"
    if reason == "no_face":
        return "no_face"
    if reason == "multiple_faces":
        return "multiple_faces"
    if reason.startswith("liveness_error") or reason.startswith("identity_error"):
        return "error"
    if result.outcome.decision.value == "UNCERTAIN":
        return "uncertain"
    return "deny"


def run_continuous(
    pipeline: FriendAuthPipeline,
    cap: cv2.VideoCapture,
    *,
    stale_ms: float,
    max_samples: int | None,
) -> tuple[list[float], Counter[str]]:
    window = "VIGIL live auth"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    latest = _LatestFrame()
    latest_lock = threading.Lock()
    result_lock = threading.Lock()
    stop = threading.Event()

    last_result: AuthPipelineResult | None = None
    last_result_at = 0.0
    samples: list[float] = []
    counts: Counter[str] = Counter()
    infer_busy = False
    preview_times: deque[float] = deque(maxlen=30)
    camera_state = "initializing"

    def worker() -> None:
        nonlocal last_result, last_result_at, infer_busy
        last_seq = 0
        while not stop.is_set():
            with latest_lock:
                frame = None if latest.frame is None else latest.frame.copy()
                seq = latest.seq
            if frame is None or seq == last_seq:
                time.sleep(0.01)
                continue
            last_seq = seq
            infer_busy = True
            try:
                with latest_lock:
                    if latest.seq != seq and latest.frame is not None:
                        frame = latest.frame.copy()
                        last_seq = latest.seq
                t0 = time.perf_counter()
                result = pipeline.process_verified_frame(frame)
                wall = (time.perf_counter() - t0) * 1000.0
                with result_lock:
                    last_result = result
                    last_result_at = time.perf_counter()
                    samples.append(result.total_ms)
                    counts[_session_bucket(result)] += 1
                id_show = result.outcome.identity or "UNKNOWN"
                print(
                    f"auto#{len(samples)} {result.outcome.decision.value} "
                    f"{result.outcome.reason} id={id_show} "
                    f"total_ms={result.total_ms:.1f} wall={wall:.1f} "
                    f"scrfd={result.scrfd_ms:.1f} fas={result.fasnet_ms:.1f} "
                    f"fn={result.facenet_ms:.1f}"
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[ERROR] inference: {type(exc).__name__}: {exc}", file=sys.stderr)
                with result_lock:
                    last_result = None
                    last_result_at = 0.0
            finally:
                infer_busy = False
            if max_samples is not None and len(samples) >= max_samples:
                stop.set()
                break

    thread = threading.Thread(target=worker, name="auth-infer", daemon=True)
    thread.start()
    print(
        f"[INFO] Continuous auth: latest-frame only; stale_after={stale_ms:.0f} ms. "
        "Q/ESC quit."
    )

    try:
        while not stop.is_set():
            ok, frame = cap.read()
            now = time.perf_counter()
            if not ok or frame is None:
                camera_state = "unavailable"
                print("[WARN] Camera read failed; clearing current auth state.")
                with result_lock:
                    last_result = None
                    last_result_at = 0.0
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                draw_auth(
                    blank,
                    None,
                    stale=True,
                    mode="continuous",
                    busy=infer_busy,
                    age_ms=None,
                    camera_state=camera_state,
                    preview_fps=None,
                    auth_hz=None,
                )
                cv2.imshow(window, blank)
                if (cv2.waitKey(1) & 0xFF) in (27, ord("q"), ord("Q")):
                    stop.set()
                    break
                time.sleep(0.05)
                continue

            camera_state = "frames_ok"
            preview_times.append(now)
            with latest_lock:
                latest.frame = frame
                latest.seq += 1

            with result_lock:
                result = last_result
                age_ms = (
                    (now - last_result_at) * 1000.0 if last_result_at > 0 else None
                )
                sample_snap = list(samples[-20:])
            stale = result is None or age_ms is None or age_ms > stale_ms

            preview_fps = None
            if len(preview_times) >= 2:
                dt = preview_times[-1] - preview_times[0]
                if dt > 0:
                    preview_fps = (len(preview_times) - 1) / dt
            auth_hz = None
            if len(sample_snap) >= 1:
                med = statistics.median(sample_snap)
                if med > 0:
                    auth_hz = 1000.0 / med

            display = frame.copy()
            draw_auth(
                display,
                result,
                stale=stale,
                mode="continuous",
                busy=infer_busy,
                age_ms=None if stale else age_ms,
                camera_state=camera_state,
                preview_fps=preview_fps,
                auth_hz=auth_hz,
            )
            cv2.imshow(window, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                stop.set()
                break
    finally:
        stop.set()
        thread.join(timeout=30.0)
        cv2.destroyAllWindows()

    return samples, counts


def run_manual(
    pipeline: FriendAuthPipeline,
    cap: cv2.VideoCapture,
    *,
    stale_ms: float,
    max_samples: int | None,
) -> tuple[list[float], Counter[str]]:
    window = "VIGIL live auth (manual)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    samples: list[float] = []
    counts: Counter[str] = Counter()
    last: AuthPipelineResult | None = None
    last_at = 0.0
    preview_times: deque[float] = deque(maxlen=30)
    print("[INFO] Manual mode: SPACE scores current frame. Q/ESC quit.")

    try:
        while True:
            ok, frame = cap.read()
            now = time.perf_counter()
            if not ok or frame is None:
                camera_state = "unavailable"
                last = None
                display = np.zeros((480, 640, 3), dtype=np.uint8)
                frame = None
            else:
                camera_state = "frames_ok"
                preview_times.append(now)
                display = frame.copy()
            age = (now - last_at) * 1000.0 if last_at > 0 else None
            stale = last is None or age is None or age > stale_ms or frame is None
            preview_fps = None
            if len(preview_times) >= 2:
                dt = preview_times[-1] - preview_times[0]
                if dt > 0:
                    preview_fps = (len(preview_times) - 1) / dt
            auth_hz = None
            if samples:
                med = statistics.median(samples[-20:])
                if med > 0:
                    auth_hz = 1000.0 / med
            draw_auth(
                display,
                last,
                stale=stale,
                mode="manual",
                busy=False,
                age_ms=None if stale else age,
                camera_state=camera_state,
                preview_fps=preview_fps,
                auth_hz=auth_hz,
            )
            cv2.imshow(window, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            if key == 32 and frame is not None:
                last = pipeline.process_verified_frame(frame)
                last_at = time.perf_counter()
                samples.append(last.total_ms)
                counts[_session_bucket(last)] += 1
                print(
                    f"sample#{len(samples)} {last.outcome.decision.value} "
                    f"{last.outcome.reason} id={last.outcome.identity or 'UNKNOWN'} "
                    f"total_ms={last.total_ms:.1f}"
                )
                if max_samples and len(samples) >= max_samples:
                    break
    finally:
        cv2.destroyAllWindows()
    return samples, counts


def main() -> int:
    parser = argparse.ArgumentParser(
        description="VIGIL friend-auth live demo (no HMAC)"
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--enrollment-db", type=Path, default=DEFAULT_ENROLLMENT_PATH)
    parser.add_argument("--threshold", type=float, default=MATCH_THRESHOLD)
    parser.add_argument(
        "--mode",
        choices=("continuous", "manual"),
        default="continuous",
        help="continuous=auto latest-frame (default); manual=SPACE to score",
    )
    parser.add_argument(
        "--stale-ms",
        type=float,
        default=2500.0,
        help="Clear displayed decision if older than this many ms (default 2500)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on scored frames then exit",
    )
    args = parser.parse_args()

    face_mod = load_face_detector_module()
    pipeline = FriendAuthPipeline(
        face_mod.FaceDetector(),
        enrollment_path=args.enrollment_db,
        match_threshold=args.threshold,
    )
    print("[INFO] Warming FasNet + FaceNet (one-time)...")
    t_warm = time.perf_counter()
    init = pipeline.warm_up()
    warm_ms = (time.perf_counter() - t_warm) * 1000.0
    print(
        f"[INFO] warm wall={warm_ms:.0f} ms "
        f"fasnet_init={init['fasnet_init_ms']} facenet_init={init['facenet_init_ms']}"
    )
    for ident_id, rec in pipeline.enrollments.get("identities", {}).items():
        from facenet_identity import display_name_of, identity_embedding_count

        print(
            f"[INFO] Enrolled {ident_id!r} display={display_name_of(rec, ident_id)!r} "
            f"embeddings={len(rec.get('embeddings') or [])}"
        )
    n_id = len(pipeline.enrollments.get("identities", {}))
    if n_id == 0:
        print(
            "[WARN] No enrollments — REAL faces will DENY unknown_identity. "
            "Run: python enroll_identity.py --identity <PersonName> --count 15",
            file=sys.stderr,
        )

    camera_state = "initializing"
    print(f"[INFO] Opening camera (state={camera_state})...")
    cap = _open_camera(args.camera)
    try:
        if args.mode == "continuous":
            samples, counts = run_continuous(
                pipeline,
                cap,
                stale_ms=args.stale_ms,
                max_samples=args.max_samples,
            )
        else:
            samples, counts = run_manual(
                pipeline,
                cap,
                stale_ms=args.stale_ms,
                max_samples=args.max_samples,
            )
    finally:
        cap.release()

    print("\n--- Session summary ---")
    for key in (
        "accept",
        "deny",
        "spoof",
        "unknown",
        "no_face",
        "multiple_faces",
        "uncertain",
        "error",
    ):
        if counts.get(key):
            print(f"  {key}: {counts[key]}")
    print(f"  total_scored: {sum(counts.values())}")
    if samples:
        steady = samples[1:] if len(samples) > 1 else samples
        hz = 1000.0 / statistics.median(steady) if steady and statistics.median(steady) > 0 else 0.0
        print(
            f"Latency n={len(samples)} median={statistics.median(samples):.1f} ms "
            f"mean={statistics.mean(samples):.1f} ms "
            f"min={min(samples):.1f} max={max(samples):.1f}"
        )
        print(
            f"Steady-state (excl. first) n={len(steady)} "
            f"median={statistics.median(steady):.1f} ms "
            f"auth_rate~={hz:.2f} Hz"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
