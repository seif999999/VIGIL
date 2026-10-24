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
from pathlib import Path

import cv2
import numpy as np

from dataclasses import replace

from auth_decision import AuthDecision
from auth_pipeline import AuthPipelineResult, FriendAuthPipeline
from challenge1_runner import outcome_from_challenge, run_challenge1_interactive
from challenge1_session import decide_challenge1_gate
from face_bridge import load_face_detector_module
from facenet_identity import DEFAULT_ENROLLMENT_PATH, MATCH_THRESHOLD, MATCH_THRESHOLD_STATUS
from fasnet_calibration_record import (
    CalibrationSessionRecorder,
    status_line as rec_status_line,
)
from fasnet_decision_policy import (
    active_research_policy,
    binary_native_policy,
    default_shadow_policy,
)
from fasnet_liveness import FasNetClass
from live_runtime import (
    CAMERA_INITIALIZING,
    CAMERA_UNAVAILABLE,
    FRAMES_OK,
    MODELS_LOADING,
    QUIT_KEY_CODES,
    WAITING_FOR_FRAMES,
    InterruptFlag,
    LatestFrame,
    configure_tf_threads_for_inference,
    frame_is_usable,
    poll_key_code,
)

# BGR colors
_GREEN = (40, 200, 40)
_RED = (40, 40, 220)
_AMBER = (0, 180, 255)
_WHITE = (240, 240, 240)
_PANEL = (24, 24, 24)

# Require a few consecutive usable frames before enabling auth (not a grace ACCEPT).
_USABLE_FRAME_STREAK = 3


# Post-open validation budget (bounded; no endless reopen loops).
_CAMERA_VALIDATE_READS = 8
_CAMERA_MIN_USABLE = 1
_CAMERA_SETTLE_S = 0.4


def _camera_backend_candidates() -> list[tuple[str, int | None]]:
    """Prefer MSMF on Windows; DSHOW next; plain open last."""
    backends: list[tuple[str, int | None]] = []
    if hasattr(cv2, "CAP_MSMF"):
        backends.append(("MSMF", int(cv2.CAP_MSMF)))
    if hasattr(cv2, "CAP_DSHOW"):
        backends.append(("DSHOW", int(cv2.CAP_DSHOW)))
    backends.append(("ANY", None))
    return backends


def _read_capture_usability(
    cap: cv2.VideoCapture,
    *,
    n_reads: int = _CAMERA_VALIDATE_READS,
) -> tuple[int, int, float | None, float | None, tuple[int, ...] | None]:
    """
    Sample up to n_reads frames. Returns
    (ok_reads, usable_reads, last_mean, last_std, last_shape).
    Uses the same frame_is_usable() gate as the live preview loop.
    """
    ok_n = 0
    usable_n = 0
    last_mean = last_std = None
    last_shape: tuple[int, ...] | None = None
    for _ in range(max(1, int(n_reads))):
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        ok_n += 1
        sample = frame[::8, ::8, :3]
        last_mean = float(sample.mean())
        last_std = float(sample.std())
        last_shape = tuple(int(x) for x in frame.shape)
        if frame_is_usable(frame):
            usable_n += 1
    return ok_n, usable_n, last_mean, last_std, last_shape


def _videocapture_open(index: int, api: int | None) -> cv2.VideoCapture:
    if api is None:
        return cv2.VideoCapture(index)
    return cv2.VideoCapture(index, api)


def _open_camera_validated(
    index: int,
    backends: list[tuple[str, int | None]],
    *,
    open_fn=_videocapture_open,
    sleep_fn=time.sleep,
    validate_reads: int = _CAMERA_VALIDATE_READS,
    min_usable: int = _CAMERA_MIN_USABLE,
    settle_s: float = _CAMERA_SETTLE_S,
) -> cv2.VideoCapture:
    """
    Try backends in order; keep only a capture that yields usable frames.

    open_fn/sleep_fn are injectable for unit tests (no physical camera).
    """
    attempts: list[str] = []
    last_error = "no backends tried"

    for i, (name, api) in enumerate(backends):
        try:
            cap = open_fn(index, api)
        except Exception as exc:  # noqa: BLE001
            attempts.append(f"{name}:open_exception:{type(exc).__name__}")
            last_error = str(exc)
            continue

        if not cap.isOpened():
            try:
                cap.release()
            except Exception:  # noqa: BLE001
                pass
            attempts.append(f"{name}:not_opened")
            last_error = f"{name} failed to open"
            print(
                f"[WARN] Camera index={index} backend={name} did not open; "
                "trying next backend.",
                file=sys.stderr,
            )
            continue

        ok_n, usable_n, mean, std, shape = _read_capture_usability(
            cap, n_reads=validate_reads
        )
        shape_s = f" shape={shape}" if shape is not None else ""
        stats_s = (
            f" mean={mean:.2f} std={std:.2f}"
            if mean is not None and std is not None
            else ""
        )
        if usable_n >= min_usable:
            print(
                f"[INFO] Camera index={index} selected backend={name} "
                f"validate ok={ok_n}/{validate_reads} "
                f"usable={usable_n}/{validate_reads}{shape_s}{stats_s}"
            )
            if attempts:
                print(
                    f"[INFO] Camera backend fallback path: {' -> '.join(attempts)} "
                    f"-> {name}"
                )
            return cap

        reason = (
            "no_frames_readable"
            if ok_n == 0
            else f"black_or_unusable ok={ok_n} usable={usable_n}"
        )
        attempts.append(f"{name}:{reason}")
        last_error = f"{name} {reason}{shape_s}{stats_s}"
        print(
            f"[WARN] Camera index={index} backend={name} rejected "
            f"({reason}{shape_s}{stats_s}); releasing and trying next.",
            file=sys.stderr,
        )
        try:
            cap.release()
        except Exception:  # noqa: BLE001
            pass

        # Bounded settle only between backends (not after the last attempt).
        if i < len(backends) - 1 and settle_s > 0:
            sleep_fn(settle_s)

    raise RuntimeError(
        f"Cannot open usable camera {index} "
        f"(tried {', '.join(attempts) or 'none'}; last={last_error}). "
        "Camo/OpenCV may need a moment after a prior session, or try another "
        "--camera index."
    )


def _open_camera(
    index: int,
    *,
    validate_reads: int = _CAMERA_VALIDATE_READS,
    min_usable: int = _CAMERA_MIN_USABLE,
    settle_s: float = _CAMERA_SETTLE_S,
) -> cv2.VideoCapture:
    """
    Open camera with MSMF → DSHOW → default preference.

    isOpened()/read() alone are insufficient: DSHOW can return all-black
    buffers. Reject backends that never produce frame_is_usable frames within
    validate_reads, then settle briefly and try the next backend.
    """
    return _open_camera_validated(
        index,
        _camera_backend_candidates(),
        validate_reads=validate_reads,
        min_usable=min_usable,
        settle_s=settle_s,
    )


def _probe_camera_frames(cap: cv2.VideoCapture, *, index: int, n: int = 8) -> None:
    """Log a short post-open usability sample (auth still fail-closed on black)."""
    ok_n, usable_n, last_mean, last_std, last_shape = _read_capture_usability(
        cap, n_reads=n
    )
    shape_s = f" shape={last_shape}" if last_shape is not None else ""
    stats_s = (
        f" last_mean={last_mean:.2f} last_std={last_std:.2f}"
        if last_mean is not None and last_std is not None
        else ""
    )
    print(
        f"[INFO] Camera index={index} probe reads ok={ok_n}/{n} "
        f"usable={usable_n}/{n}{shape_s}{stats_s}"
    )
    if ok_n == 0:
        print(
            "[WARN] Camera opened but no frames could be read yet. "
            "Preview will retry; auth stays disabled until usable frames arrive.",
            file=sys.stderr,
        )
    elif usable_n == 0:
        print(
            "[WARN] Camera frames look black/uniform (not usable). "
            "Auth will not start until textured frames arrive. "
            "If this persists, try another --camera index (Camo index can change).",
            file=sys.stderr,
        )


def _print_session_summary(samples: list[float], counts: Counter[str]) -> None:
    print("\n--- Session summary ---")
    for key in (
        "accept",
        "deny",
        "spoof",
        "unknown",
        "no_face",
        "multiple_faces",
        "uncertain",
        "challenge_required",
        "error",
        "skipped_not_ready",
    ):
        if counts.get(key):
            print(f"  {key}: {counts[key]}")
    print(f"  total_scored: {sum(v for k, v in counts.items() if k != 'skipped_not_ready')}")
    if samples:
        steady = samples[1:] if len(samples) > 1 else samples
        hz = (
            1000.0 / statistics.median(steady)
            if steady and statistics.median(steady) > 0
            else 0.0
        )
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
        "liveness_uncertain": "Liveness uncertain",
        "challenge_required_not_implemented": (
            "CHALLENGE_REQUIRED / NOT IMPLEMENTED"
        ),
        "challenge_1_required": "Challenge 1 required (head movement)",
        "challenge_2_required": "Challenge 2 required (not implemented)",
        "challenge1_passed_real_and_recognized": "Challenge 1 passed",
        "challenge1_spoof": "Challenge 1: spoof on re-check",
        "challenge1_unknown_identity": "Challenge 1: unknown identity",
    }
    if reason in mapping:
        return mapping[reason]
    if reason.startswith("liveness_error"):
        return f"Liveness inference error ({reason.split(':', 1)[-1]})"
    if reason.startswith("identity_error"):
        return f"Identity inference error ({reason.split(':', 1)[-1]})"
    return reason


def _fasnet_lines(fas, outcome=None) -> list[str]:
    """
    DeepFace FasNet returns (is_real, score) where
      score = sum(softmax_model1 + softmax_model2)[argmax_label] / 2
    i.e. uncalibrated confidence for the *winning* class (REAL or SPOOF),
    not a separate REAL probability when the class is SPOOF.

    Debug-mode HUD only — not shown in the default simplified view.
    """
    if fas is None:
        return ["Liveness: N/A", "  (not run)"]
    if fas.predicted_class == FasNetClass.ERROR or fas.is_real is None:
        return [
            "Liveness native: ERROR",
            f"  {fas.error or 'unknown'}",
            "  Check: FAIL",
        ]
    cls = fas.predicted_class.value
    score = fas.antispoof_score
    check = "PASS" if fas.is_real else "FAIL"
    if score is None:
        score_line = f"  {cls} class_conf: n/a (uncalibrated)"
    else:
        score_line = f"  {cls} class_conf: {score:.3f} (uncalibrated)"
    lines = [
        f"Native: {cls}",
        score_line,
        "  (winning-class score; not verified probability)",
        f"  Native check: {check}",
    ]
    if fas.real_score_avg is not None and fas.spoof_score_combined is not None:
        lines.append(
            f"  avg REAL={fas.real_score_avg:.3f} SPOOF*={fas.spoof_score_combined:.3f}"
        )
        lines.append("  (*SPOOF=class0+class2 avg; uncalibrated)")
    if outcome is not None:
        mode = outcome.fasnet_policy_mode or "n/a"
        status = outcome.fasnet_policy_status or "n/a"
        auth_st = outcome.fasnet_auth_state or "n/a"
        shadow_st = outcome.fasnet_shadow_state or "n/a"
        rmin = outcome.fasnet_real_min
        smin = outcome.fasnet_spoof_min
        thr = (
            f"REAL>={rmin:.2f} SPOOF>={smin:.2f}"
            if rmin is not None and smin is not None
            else "n/a"
        )
        lines.append(f"Policy mode: {mode}")
        lines.append(f"  status: {status}")
        lines.append(f"  gates: {thr} (uncalibrated)")
        lines.append(f"  auth state: {auth_st}")
        if mode in ("shadow", "binary_native"):
            lines.append(f"  shadow state: {shadow_st} (diag only)")
        c1 = outcome.challenge_1_status or "NOT_APPLICABLE"
        if outcome.challenge_1_required or c1 not in (
            "NOT_APPLICABLE",
            "NOT_REQUIRED",
        ):
            lines.append(f"Challenge1: {c1}")
            if outcome.challenge_1_required:
                lines.append("  -> REQUIRED / NOT IMPLEMENTED")
    return lines


def _liveness_user_label(outcome, fas) -> str:
    """Map auth liveness to LIVE / SPOOF / UNCERTAIN for the big title."""
    auth_st = getattr(outcome, "fasnet_auth_state", None) if outcome is not None else None
    if auth_st == "REAL":
        return "LIVE"
    if auth_st == "SPOOF":
        return "SPOOF"
    if auth_st == "UNCERTAIN":
        return "UNCERTAIN"
    if outcome is not None and outcome.liveness_class == "REAL":
        return "LIVE"
    if outcome is not None and outcome.liveness_class == "SPOOF":
        return "SPOOF"
    if outcome is not None and outcome.liveness_class == "UNCERTAIN":
        return "UNCERTAIN"
    if fas is not None and fas.is_real is True:
        return "LIVE"
    if fas is not None and fas.is_real is False:
        return "SPOOF"
    return "UNCERTAIN"


def _liveness_result_label(outcome, fas) -> str:
    """Liveness section uses REAL / SPOOF / UNCERTAIN wording."""
    live = _liveness_user_label(outcome, fas)
    return "REAL" if live == "LIVE" else live


def _identity_title_part(outcome) -> str:
    """Right side of big title: name, NOT IDENTIFIED, or UNKNOWN (spoof)."""
    if outcome is None:
        return "NOT IDENTIFIED"
    if outcome.recognized and outcome.identity:
        return str(outcome.identity).upper()
    live = (outcome.fasnet_auth_state or outcome.liveness_class or "").upper()
    if live == "SPOOF":
        return "UNKNOWN"
    return "NOT IDENTIFIED"


def _status_title(outcome, fas) -> str:
    return f"{_liveness_user_label(outcome, fas)} — {_identity_title_part(outcome)}"


def _access_result_label(decision: str) -> str:
    if decision == "ACCEPT":
        return "ALLOW"
    if decision == "CHALLENGE_REQUIRED":
        return "CHALLENGE REQUIRED"
    return "DENY"


def _confidence_value(result: AuthPipelineResult) -> float | None:
    outcome = result.outcome
    if outcome.fasnet_winning_score is not None:
        return float(outcome.fasnet_winning_score)
    fas = result.fasnet
    if fas is not None and fas.antispoof_score is not None:
        return float(fas.antispoof_score)
    return None


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
    record_status: str | None = None,
    debug: bool = False,
) -> None:
    h, w = frame.shape[:2]
    panel_w = min(440, max(300, w // 3))
    x0 = 8
    x1 = x0 + panel_w
    y0 = 8
    # Semi-opaque left panel (avoid covering center face when possible)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, h - 8), _PANEL, -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    def line(text: str, y: int, color=_WHITE, scale=0.48, thick=1) -> int:
        _put(frame, text[:62], (x0 + 10, y), color, scale, thick)
        return y + int(22 * max(0.9, scale / 0.48))

    def section(title: str, y: int) -> int:
        return line(title, y + 4, _WHITE, 0.5, 1)

    # --- Camera / freshness banner (top-right, clear of left panel) ---
    cam_color = _GREEN if camera_state == FRAMES_OK and not stale else _AMBER
    if camera_state == CAMERA_UNAVAILABLE:
        cam_color = _RED
    cam_label = f"Camera: {camera_state.upper()}"
    cam_x = w - 12 - int(len(cam_label) * 11 * 0.6)
    _put(frame, cam_label, (max(x1 + 8, cam_x), 28), cam_color, 0.6, 2)
    if preview_fps is not None:
        fps_label = f"Preview FPS: {preview_fps:.1f}"
        fps_x = w - 12 - int(len(fps_label) * 10 * 0.5)
        _put(frame, fps_label, (max(x1 + 8, fps_x), 54), _WHITE, 0.5, 1)
    if record_status:
        rec_color = _GREEN if "MARKED" in record_status else _AMBER
        _put(frame, record_status[:48], (max(x1 + 8, w - 12 - 280), 80), rec_color, 0.5, 2)
    if debug:
        _put(frame, "DEBUG HUD", (max(x1 + 8, w - 140), 106 if record_status else 80), _AMBER, 0.5, 2)

    help_bits = ["Q/ESC=quit", "D=debug"]
    if mode == "manual":
        help_bits.insert(0, "SPACE=score")
    if record_status:
        help_bits.insert(0, "M=mark")
    help_line = "  ".join(help_bits)

    if result is None or stale:
        decision = "STALE" if result is not None else "WAITING"
        if camera_state == CAMERA_UNAVAILABLE:
            decision = "DENY"
            reason = "Camera unavailable"
            color = _RED
        elif camera_state in (CAMERA_INITIALIZING, "initializing"):
            decision = "WAITING"
            reason = "Camera initializing"
            color = _AMBER
        elif camera_state == WAITING_FOR_FRAMES:
            decision = "WAITING"
            reason = "Waiting for valid camera frames"
            color = _AMBER
        elif camera_state == MODELS_LOADING:
            decision = "LOADING MODELS"
            reason = "Preview live — authentication starts when ready"
            color = _AMBER
        else:
            reason = "Result stale / waiting for fresh inference"
            color = _AMBER
        y = y0 + 32
        y = line(decision, y, color, 1.1, 3)
        y = line(reason, y, color, 0.55, 1)
        if result is not None and stale:
            y = line(
                f"Last: {_status_title(result.outcome, result.fasnet)}",
                y,
                _AMBER,
                0.45,
                1,
            )
            y = line("(not current — do not treat as ALLOW)", y, _AMBER, 0.45, 1)
        if debug:
            y = line(f"Infer: {'BUSY' if busy else 'idle'}  mode={mode}", y, _WHITE, 0.45)
        y = line(help_line, y, _WHITE, 0.45)
        return

    # Fresh result
    decision = result.outcome.decision.value
    access = _access_result_label(decision)
    if decision == "ACCEPT":
        color = _GREEN
    elif decision in ("UNCERTAIN", "CHALLENGE_REQUIRED"):
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
    if result.outcome.recognized and (
        result.outcome.identity or (id_match and id_match.display_name)
    ):
        shown_name = (
            result.outcome.identity
            or (id_match.display_name if id_match else None)
            or "UNKNOWN"
        )
        recog = "MATCHED"
        id_color = _GREEN
    else:
        shown_name = "UNKNOWN"
        recog = "NOT IDENTIFIED"
        id_color = _RED if decision == "DENY" else _AMBER

    dist = id_match.best_distance if id_match is not None else None
    thr = id_match.threshold if id_match is not None else MATCH_THRESHOLD

    fas = result.fasnet
    live_user = _liveness_user_label(result.outcome, fas)
    live_result = _liveness_result_label(result.outcome, fas)
    title = _status_title(result.outcome, fas)
    conf = _confidence_value(result)

    if live_user == "LIVE":
        live_color = _GREEN
    elif live_user == "SPOOF":
        live_color = _RED
    else:
        live_color = _AMBER

    # Security checks (PASS / FAIL only — user-facing gates)
    face_check = (
        "PASS"
        if det.decision.value == "FACE_OK" and det.face_count == 1
        else "FAIL"
    )
    if live_user == "LIVE":
        live_check = "PASS"
    else:
        live_check = "FAIL"
    id_check = "PASS" if result.outcome.recognized else "FAIL"

    y = y0 + 36
    y = line(title, y, color, 1.05, 3)

    y = section("--- Decision ---", y)
    y = line(f"Final liveness: {live_user}", y, live_color, 0.5, 1)
    y = line(f"Access result: {access}", y, color, 0.5, 1)

    y = section("--- Identity ---", y)
    y = line(f"Identity: {shown_name}", y, id_color, 0.5, 1)
    y = line(f"Recognition: {recog}", y, id_color, 0.5, 1)
    if dist is not None:
        y = line(f"FaceNet distance: {dist:.2f}", y, _WHITE, 0.48, 1)
    else:
        y = line("FaceNet distance: n/a", y, _WHITE, 0.48, 1)
    y = line(f"Threshold: {thr:.2f}", y, _WHITE, 0.48, 1)

    y = section("--- Liveness ---", y)
    y = line(f"Liveness result: {live_result}", y, live_color, 0.5, 1)
    if conf is not None:
        y = line(f"Confidence: {conf:.3f}", y, live_color, 0.55, 2)
    else:
        y = line("Confidence: n/a", y, _AMBER, 0.5, 1)

    y = section("--- Security checks ---", y)
    y = line(
        f"Face count: {face_check}",
        y,
        _GREEN if face_check == "PASS" else _RED,
        0.48,
    )
    y = line(
        f"Liveness: {live_check}",
        y,
        _GREEN if live_check == "PASS" else _RED,
        0.48,
    )
    y = line(
        f"Identity: {id_check}",
        y,
        _GREEN if id_check == "PASS" else _RED,
        0.48,
    )

    y = section("--- Performance ---", y)
    y = line(f"SCRFD:   {result.scrfd_ms:.0f} ms", y, _WHITE, 0.48)
    y = line(f"FasNet:  {result.fasnet_ms:.0f} ms", y, _WHITE, 0.48)
    y = line(f"FaceNet: {result.facenet_ms:.0f} ms", y, _WHITE, 0.48)
    y = line(f"Total:   {result.total_ms:.0f} ms", y, _WHITE, 0.48)

    if debug:
        y = section("--- Debug ---", y)
        y = line(f"Reason: {_reason_label(result.outcome.reason)}", y, _AMBER, 0.42)
        y = line(f"Decision: {decision}", y, _AMBER, 0.42)
        y = line(
            f"Faces: {det.face_count}  gate={det.decision.value}",
            y,
            _WHITE,
            0.42,
        )
        y = line(
            f"Threshold status: {MATCH_THRESHOLD_STATUS}",
            y,
            _WHITE,
            0.42,
        )
        for fl in _fasnet_lines(fas, result.outcome):
            y = line(fl, y, live_color, 0.40)
        if auth_hz is not None:
            y = line(f"Auth rate: {auth_hz:.2f} Hz", y, _WHITE, 0.42)
        if preview_fps is not None:
            y = line(f"Preview: {preview_fps:.1f} FPS", y, _WHITE, 0.42)
        if age_ms is not None:
            y = line(f"Result age: {age_ms:.0f} ms", y, _WHITE, 0.42)
        y = line(f"Infer: {'BUSY' if busy else 'idle'}  mode={mode}", y, _WHITE, 0.42)
        if record_status:
            y = line(record_status[:56], y, _AMBER if "PAUSED" in record_status else _GREEN, 0.42)

    y = line(help_line, y + 2, _WHITE, 0.42)


def _session_bucket(result: AuthPipelineResult) -> str:
    reason = result.outcome.reason
    if result.outcome.decision == AuthDecision.ACCEPT:
        return "accept"
    if result.outcome.decision == AuthDecision.CHALLENGE_REQUIRED:
        return "challenge_required"
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


def _apply_challenge1_gate(
    result: AuthPipelineResult, *, enable_challenge1: bool
) -> AuthPipelineResult:
    if not enable_challenge1:
        return result
    gated = decide_challenge1_gate(
        challenge_enabled=True, outcome=result.outcome
    )
    if gated is result.outcome:
        return result
    return replace(result, outcome=gated)


def run_continuous(
    pipeline_holder: dict,
    cap: cv2.VideoCapture,
    *,
    stale_ms: float,
    max_samples: int | None,
    recorder: CalibrationSessionRecorder | None = None,
    models_ready: threading.Event,
    interrupt: InterruptFlag,
    pipeline_error: list,
    enable_challenge1: bool = False,
) -> tuple[list[float], Counter[str], str]:
    """
    Preview runs immediately. Authentication starts only when models_ready is set
    and usable camera frames are flowing (no auth on blank startup buffers).
    pipeline_holder['pipeline'] is filled by the background warm thread.

    Returns (latency_samples, decision_counts, exit_reason).
    """
    window = "VIGIL live auth"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    latest = LatestFrame()
    latest_lock = threading.Lock()
    result_lock = threading.Lock()
    stop = threading.Event()

    last_result: AuthPipelineResult | None = None
    last_result_at = 0.0
    samples: list[float] = []
    counts: Counter[str] = Counter()
    infer_busy = False
    preview_times: deque[float] = deque(maxlen=30)
    camera_state = CAMERA_INITIALIZING
    rec_cam_state = CAMERA_INITIALIZING
    rec_preview_fps: float | None = None
    rec_auth_hz: float | None = None
    metrics_lock = threading.Lock()
    usable_streak = 0
    frames_ready_ev = threading.Event()
    auth_enabled_logged = False
    debug_hud = False
    challenge1_armed = True
    challenge_pause = threading.Event()  # set while Challenge 1 owns the camera
    exit_reason = "unknown"
    reads_ok = 0
    reads_fail = 0
    usable_frames = 0
    unusable_frames = 0
    last_frame_log_at = 0.0
    unusable_warned = False

    def worker() -> None:
        nonlocal last_result, last_result_at, infer_busy, auth_enabled_logged, exit_reason
        last_seq = 0
        while not stop.is_set() and not interrupt.is_set:
            if challenge_pause.is_set():
                time.sleep(0.02)
                continue
            if not models_ready.is_set():
                time.sleep(0.02)
                continue
            if pipeline_error:
                time.sleep(0.05)
                continue
            pipeline: FriendAuthPipeline | None = pipeline_holder.get("pipeline")
            if pipeline is None:
                time.sleep(0.02)
                continue
            if not frames_ready_ev.is_set():
                time.sleep(0.01)
                continue
            with latest_lock:
                if latest.frame is None or latest.seq == last_seq:
                    frame = None
                    seq = latest.seq
                    cap_ts = latest.capture_ts
                else:
                    # Own a copy so OpenCV can reuse the capture buffer safely.
                    frame = latest.frame.copy()
                    seq = latest.seq
                    cap_ts = latest.capture_ts
            if frame is None or seq == last_seq:
                time.sleep(0.01)
                continue
            if not frame_is_usable(frame):
                counts["skipped_not_ready"] += 1
                last_seq = seq
                time.sleep(0.01)
                continue
            if not auth_enabled_logged:
                print("[INFO] Authentication enabled (models ready + valid frames).")
                auth_enabled_logged = True
            last_seq = seq
            infer_busy = True
            try:
                # Drop to newest if several frames arrived while we were copying.
                with latest_lock:
                    if latest.seq != seq and latest.frame is not None:
                        frame = latest.frame.copy()
                        seq = latest.seq
                        cap_ts = latest.capture_ts
                        last_seq = seq
                age_at_start_ms = (time.perf_counter() - cap_ts) * 1000.0 if cap_ts else None
                t0 = time.perf_counter()
                result = _apply_challenge1_gate(
                    pipeline.process_verified_frame(frame),
                    enable_challenge1=enable_challenge1,
                )
                wall = (time.perf_counter() - t0) * 1000.0
                with result_lock:
                    last_result = result
                    last_result_at = time.perf_counter()
                    samples.append(result.total_ms)
                    counts[_session_bucket(result)] += 1
                if recorder is not None:
                    with metrics_lock:
                        cam_s = rec_cam_state
                        pfps = rec_preview_fps
                        ahz = rec_auth_hz
                    saved = recorder.record_eval(
                        result,
                        camera_state=cam_s,
                        preview_fps=pfps,
                        auth_hz=ahz,
                        wall_ms=wall,
                    )
                else:
                    saved = False
                id_show = result.outcome.identity or "UNKNOWN"
                fas = result.fasnet
                score_s = (
                    f" win={fas.antispoof_score:.3f}"
                    if fas is not None and fas.antispoof_score is not None
                    else ""
                )
                age_s = (
                    f" age0={age_at_start_ms:.0f}ms"
                    if age_at_start_ms is not None
                    else ""
                )
                print(
                    f"auto#{len(samples)} seq={seq} {result.outcome.decision.value} "
                    f"{result.outcome.reason} id={id_show} "
                    f"total_ms={result.total_ms:.1f} wall={wall:.1f} "
                    f"scrfd={result.scrfd_ms:.1f} fas={result.fasnet_ms:.1f} "
                    f"fn={result.facenet_ms:.1f}{score_s}{age_s}"
                    + (" [REC]" if saved else "")
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[ERROR] inference: {type(exc).__name__}: {exc}", file=sys.stderr)
                with result_lock:
                    last_result = None
                    last_result_at = 0.0
            finally:
                infer_busy = False
            if max_samples is not None and len(samples) >= max_samples:
                exit_reason = "max_samples"
                stop.set()
                break

    thread = threading.Thread(target=worker, name="auth-infer", daemon=True)
    thread.start()
    print(
        f"[INFO] Continuous preview; auth after models+valid frames; "
        f"stale_after={stale_ms:.0f} ms. Q/ESC/Ctrl+C quit. D toggles debug HUD."
        + (" M toggles MARKED recording." if recorder else "")
    )

    try:
        while not stop.is_set() and not interrupt.is_set:
            try:
                ok, frame = cap.read()
            except KeyboardInterrupt:
                interrupt.set()
                exit_reason = "keyboard_interrupt_during_read"
                break
            now = time.perf_counter()
            if not ok or frame is None:
                reads_fail += 1
                camera_state = CAMERA_UNAVAILABLE
                usable_streak = 0
                frames_ready_ev.clear()
                with metrics_lock:
                    rec_cam_state = camera_state
                    rec_preview_fps = None
                    rec_auth_hz = None
                if reads_fail == 1 or reads_fail % 30 == 0:
                    print(
                        f"[WARN] Camera read failed "
                        f"(fail={reads_fail} ok={reads_ok}); "
                        "clearing current auth state."
                    )
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
                    record_status=rec_status_line(recorder),
                    debug=debug_hud,
                )
                cv2.imshow(window, blank)
                try:
                    key = poll_key_code(1)
                except KeyboardInterrupt:
                    interrupt.set()
                    exit_reason = "keyboard_interrupt_during_waitKey"
                    break
                if key in QUIT_KEY_CODES:
                    exit_reason = f"quit_key code={key}"
                    stop.set()
                    break
                if key in (ord("d"), ord("D")):
                    debug_hud = not debug_hud
                    print(f"[HUD] debug={'on' if debug_hud else 'off'}")
                if recorder is not None and key in (ord("m"), ord("M")):
                    on = recorder.toggle_marked()
                    print(f"[REC] marked={on}")
                time.sleep(0.05)
                continue

            reads_ok += 1
            preview_times.append(now)
            usable = frame_is_usable(frame)
            if usable:
                usable_frames += 1
                usable_streak += 1
            else:
                unusable_frames += 1
                usable_streak = 0
                frames_ready_ev.clear()
                if not unusable_warned and unusable_frames >= 15 and usable_frames == 0:
                    unusable_warned = True
                    print(
                        "[WARN] Still no usable camera frames "
                        f"(ok_reads={reads_ok} unusable={unusable_frames}). "
                        "Auth remains disabled until frames have real texture.",
                        file=sys.stderr,
                    )
            if now - last_frame_log_at >= 5.0:
                last_frame_log_at = now
                print(
                    f"[INFO] Frame status read_ok={reads_ok} read_fail={reads_fail} "
                    f"usable={usable_frames} unusable={unusable_frames} "
                    f"models_ready={models_ready.is_set()} "
                    f"frames_ready={frames_ready_ev.is_set()} "
                    f"scored={len(samples)}"
                )

            if usable_streak >= _USABLE_FRAME_STREAK:
                frames_ready_ev.set()

            if not models_ready.is_set():
                camera_state = MODELS_LOADING if usable else WAITING_FOR_FRAMES
            elif not frames_ready_ev.is_set():
                camera_state = WAITING_FOR_FRAMES
            else:
                camera_state = FRAMES_OK

            # Publish for inference only when auth may run; always keep preview.
            # Store reference; worker copies under lock (OpenCV may reuse read buffer).
            if models_ready.is_set() and frames_ready_ev.is_set() and usable:
                with latest_lock:
                    latest.frame = frame
                    latest.seq += 1
                    latest.capture_ts = now

            with result_lock:
                result = last_result
                age_ms = (
                    (now - last_result_at) * 1000.0 if last_result_at > 0 else None
                )
                sample_snap = list(samples[-20:])
            # Never show a prior ACCEPT as current when not fully ready.
            if camera_state != FRAMES_OK:
                result = None
                age_ms = None
                stale = True
            else:
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
            with metrics_lock:
                rec_cam_state = camera_state
                rec_preview_fps = preview_fps
                rec_auth_hz = auth_hz

            # Opt-in Challenge 1 when active_research (or any path) requests it.
            if (
                enable_challenge1
                and challenge1_armed
                and not stale
                and result is not None
                and result.outcome.challenge_1_required
                and result.outcome.decision == AuthDecision.CHALLENGE_REQUIRED
                and models_ready.is_set()
            ):
                pipe = pipeline_holder.get("pipeline")
                if pipe is not None:
                    challenge_pause.set()
                    challenge1_armed = False
                    print("[CHALLENGE1] Starting interactive head-movement challenge…")
                    try:
                        c1 = run_challenge1_interactive(
                            pipe,
                            cap,
                            window=window,
                            expected_identity=result.outcome.identity,
                            interrupt_check=lambda: interrupt.is_set or stop.is_set(),
                        )
                        c_out = outcome_from_challenge(c1)
                        with result_lock:
                            last_result = replace(result, outcome=c_out)
                            last_result_at = time.perf_counter()
                            counts[_session_bucket(last_result)] += 1
                        result = last_result
                        stale = False
                    finally:
                        challenge_pause.clear()
            elif (
                result is not None
                and result.outcome.decision != AuthDecision.CHALLENGE_REQUIRED
            ):
                challenge1_armed = True

            display = frame.copy()
            draw_auth(
                display,
                result,
                stale=stale,
                mode="continuous",
                busy=infer_busy or not models_ready.is_set() or challenge_pause.is_set(),
                age_ms=None if stale else age_ms,
                camera_state=camera_state,
                preview_fps=preview_fps,
                auth_hz=auth_hz,
                record_status=rec_status_line(recorder),
                debug=debug_hud,
            )
            cv2.imshow(window, display)
            try:
                key = poll_key_code(1)
            except KeyboardInterrupt:
                interrupt.set()
                exit_reason = "keyboard_interrupt_during_waitKey"
                break
            if key in QUIT_KEY_CODES:
                exit_reason = f"quit_key code={key}"
                stop.set()
                break
            if key in (ord("d"), ord("D")):
                debug_hud = not debug_hud
                print(f"[HUD] debug={'on' if debug_hud else 'off'}")
            if recorder is not None and key in (ord("m"), ord("M")):
                on = recorder.toggle_marked()
                print(f"[REC] marked={on} saved_so_far={recorder.marked_records}")
            if pipeline_error:
                print(f"[ERROR] Model init failed: {pipeline_error[0]}", file=sys.stderr)
                exit_reason = f"pipeline_error:{pipeline_error[0]}"
                stop.set()
                break
        else:
            # while-condition failed without an inner break
            if interrupt.is_set and exit_reason == "unknown":
                exit_reason = interrupt.signal_reason()
            elif stop.is_set() and exit_reason == "unknown":
                exit_reason = "stop_event"
    finally:
        if exit_reason == "unknown":
            if interrupt.is_set:
                exit_reason = interrupt.signal_reason()
            elif stop.is_set():
                exit_reason = "stop_event"
            else:
                exit_reason = "loop_ended"
        stop.set()
        # Signal background warm/worker to wind down; do not treat this alone
        # as the user-facing exit reason (already captured above).
        interrupt.set()
        thread.join(timeout=30.0)
        try:
            cv2.destroyAllWindows()
        except Exception:  # noqa: BLE001
            pass
        print(
            f"[INFO] Preview exit reason={exit_reason} | "
            f"frames read_ok={reads_ok} read_fail={reads_fail} "
            f"usable={usable_frames} unusable={unusable_frames} "
            f"scored={len(samples)}"
        )

    return samples, counts, exit_reason


def run_manual(
    pipeline_holder: dict,
    cap: cv2.VideoCapture,
    *,
    stale_ms: float,
    max_samples: int | None,
    recorder: CalibrationSessionRecorder | None = None,
    models_ready: threading.Event,
    interrupt: InterruptFlag,
    pipeline_error: list,
    enable_challenge1: bool = False,
) -> tuple[list[float], Counter[str], str]:
    window = "VIGIL live auth (manual)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    samples: list[float] = []
    counts: Counter[str] = Counter()
    last: AuthPipelineResult | None = None
    last_at = 0.0
    preview_times: deque[float] = deque(maxlen=30)
    debug_hud = False
    challenge1_armed = True
    exit_reason = "unknown"
    reads_ok = 0
    reads_fail = 0
    usable_frames = 0
    unusable_frames = 0
    print(
        "[INFO] Manual mode: SPACE scores when models ready. Q/ESC/Ctrl+C quit. "
        "D toggles debug HUD."
        + (" M toggles MARKED recording." if recorder else "")
        + (" Challenge1 enabled." if enable_challenge1 else "")
    )

    try:
        while not interrupt.is_set:
            try:
                ok, frame = cap.read()
            except KeyboardInterrupt:
                interrupt.set()
                exit_reason = "keyboard_interrupt_during_read"
                break
            now = time.perf_counter()
            if not ok or frame is None:
                reads_fail += 1
                camera_state = CAMERA_UNAVAILABLE
                last = None
                display = np.zeros((480, 640, 3), dtype=np.uint8)
                frame = None
                if reads_fail == 1 or reads_fail % 30 == 0:
                    print(
                        f"[WARN] Camera read failed "
                        f"(fail={reads_fail} ok={reads_ok})."
                    )
            else:
                reads_ok += 1
                usable = frame_is_usable(frame)
                if usable:
                    usable_frames += 1
                else:
                    unusable_frames += 1
                if not models_ready.is_set():
                    camera_state = MODELS_LOADING if usable else WAITING_FOR_FRAMES
                elif not usable:
                    camera_state = WAITING_FOR_FRAMES
                else:
                    camera_state = FRAMES_OK
                preview_times.append(now)
                display = frame.copy()
            age = (now - last_at) * 1000.0 if last_at > 0 else None
            if camera_state != FRAMES_OK:
                stale = True
                show = None
            else:
                show = last
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
                show if camera_state == FRAMES_OK else None,
                stale=stale,
                mode="manual",
                busy=not models_ready.is_set(),
                age_ms=None if stale else age,
                camera_state=camera_state,
                preview_fps=preview_fps,
                auth_hz=auth_hz,
                record_status=rec_status_line(recorder),
                debug=debug_hud,
            )
            cv2.imshow(window, display)
            try:
                key = poll_key_code(1)
            except KeyboardInterrupt:
                interrupt.set()
                exit_reason = "keyboard_interrupt_during_waitKey"
                break
            if key in QUIT_KEY_CODES:
                exit_reason = f"quit_key code={key}"
                break
            if key in (ord("d"), ord("D")):
                debug_hud = not debug_hud
                print(f"[HUD] debug={'on' if debug_hud else 'off'}")
            if recorder is not None and key in (ord("m"), ord("M")):
                on = recorder.toggle_marked()
                print(f"[REC] marked={on} saved_so_far={recorder.marked_records}")
            if key == 32 and frame is not None:
                if not models_ready.is_set() or pipeline_holder.get("pipeline") is None:
                    print("[INFO] Models still loading — SPACE ignored.")
                    continue
                if not frame_is_usable(frame):
                    print("[INFO] Frame not usable yet — SPACE ignored.")
                    continue
                pipe: FriendAuthPipeline = pipeline_holder["pipeline"]
                last = _apply_challenge1_gate(
                    pipe.process_verified_frame(frame),
                    enable_challenge1=enable_challenge1,
                )
                last_at = time.perf_counter()
                samples.append(last.total_ms)
                counts[_session_bucket(last)] += 1
                if (
                    enable_challenge1
                    and challenge1_armed
                    and last.outcome.challenge_1_required
                    and last.outcome.decision == AuthDecision.CHALLENGE_REQUIRED
                ):
                    challenge1_armed = False
                    print("[CHALLENGE1] Starting interactive head-movement challenge…")
                    c1 = run_challenge1_interactive(
                        pipe,
                        cap,
                        window=window,
                        expected_identity=last.outcome.identity,
                        interrupt_check=lambda: interrupt.is_set,
                    )
                    last = replace(last, outcome=outcome_from_challenge(c1))
                    last_at = time.perf_counter()
                    counts[_session_bucket(last)] += 1
                elif last.outcome.decision != AuthDecision.CHALLENGE_REQUIRED:
                    challenge1_armed = True
                saved = False
                if recorder is not None:
                    saved = recorder.record_eval(
                        last,
                        camera_state=camera_state,
                        preview_fps=preview_fps,
                        auth_hz=auth_hz,
                    )
                print(
                    f"sample#{len(samples)} {last.outcome.decision.value} "
                    f"{last.outcome.reason} id={last.outcome.identity or 'UNKNOWN'} "
                    f"total_ms={last.total_ms:.1f}"
                    + (" [REC]" if saved else "")
                )
                if max_samples and len(samples) >= max_samples:
                    exit_reason = "max_samples"
                    break
            if pipeline_error:
                print(f"[ERROR] Model init failed: {pipeline_error[0]}", file=sys.stderr)
                exit_reason = f"pipeline_error:{pipeline_error[0]}"
                break
        else:
            if interrupt.is_set and exit_reason == "unknown":
                exit_reason = interrupt.signal_reason()
    finally:
        if exit_reason == "unknown":
            exit_reason = (
                interrupt.signal_reason() if interrupt.is_set else "loop_ended"
            )
        try:
            cv2.destroyAllWindows()
        except Exception:  # noqa: BLE001
            pass
        print(
            f"[INFO] Preview exit reason={exit_reason} | "
            f"frames read_ok={reads_ok} read_fail={reads_fail} "
            f"usable={usable_frames} unusable={unusable_frames} "
            f"scored={len(samples)}"
        )
    return samples, counts, exit_reason


def _warm_pipeline_background(
    *,
    enrollment_db: Path,
    threshold: float,
    fasnet_policy,
    pipeline_holder: dict,
    models_ready: threading.Event,
    pipeline_error: list,
    interrupt: InterruptFlag,
) -> None:
    try:
        t0 = time.perf_counter()
        print("[INFO] Loading SCRFD + warming FasNet/FaceNet in background...")
        face_mod = load_face_detector_module()
        t_scrfd = time.perf_counter()
        detector = face_mod.FaceDetector()
        print(f"[INFO] SCRFD ready in {(time.perf_counter() - t_scrfd) * 1000:.0f} ms")
        pipeline = FriendAuthPipeline(
            detector,
            enrollment_path=enrollment_db,
            match_threshold=threshold,
            fasnet_policy=fasnet_policy,
        )
        t_warm = time.perf_counter()
        init = pipeline.warm_up()
        warm_ms = (time.perf_counter() - t_warm) * 1000.0
        fas_backend = getattr(pipeline.fasnet, "backend", "unknown")
        fn_backend = getattr(pipeline.facenet, "backend", "unknown")
        print(
            f"[INFO] Model warm wall={warm_ms:.0f} ms "
            f"fasnet_backend={fas_backend} "
            f"facenet_backend={fn_backend} "
            f"fasnet_init={init['fasnet_init_ms']} "
            f"facenet_init={init['facenet_init_ms']} "
            f"total_bg={(time.perf_counter() - t0) * 1000:.0f} ms"
        )
        for ident_id, rec in pipeline.enrollments.get("identities", {}).items():
            from facenet_identity import display_name_of

            print(
                f"[INFO] Enrolled {ident_id!r} display={display_name_of(rec, ident_id)!r} "
                f"embeddings={len(rec.get('embeddings') or [])}"
            )
        if not pipeline.enrollments.get("identities"):
            print(
                "[WARN] No enrollments — REAL faces will DENY unknown_identity. "
                "Run: python enroll_identity.py --identity <PersonName> --count 15",
                file=sys.stderr,
            )
        if interrupt.is_set:
            print(
                "[WARN] Model warm finished but preview already exiting "
                f"({interrupt.signal_reason()}) — not enabling authentication.",
                file=sys.stderr,
            )
            return
        pipeline_holder["pipeline"] = pipeline
        models_ready.set()
        print("[INFO] Models ready — authentication will use valid frames.")
    except Exception as exc:  # noqa: BLE001
        pipeline_error.append(f"{type(exc).__name__}: {exc}")
        print(f"[ERROR] Background model init: {pipeline_error[0]}", file=sys.stderr)


def main() -> int:
    interrupt = InterruptFlag()
    interrupt.install()

    # Configure TF/OpenCV threads before model imports in the warm thread.
    print(f"[INFO] {configure_tf_threads_for_inference()}")

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
    parser.add_argument(
        "--fasnet-policy",
        choices=("shadow", "binary_native", "active_research"),
        default="shadow",
        help=(
            "shadow=default: auth uses native argmax; research 0.85 gates are "
            "diagnostic only. active_research: UNCERTAIN→CHALLENGE_REQUIRED "
            "(opt-in Challenge 1 with --enable-challenge1). Gates remain unvalidated."
        ),
    )
    parser.add_argument(
        "--enable-challenge1",
        action="store_true",
        help=(
            "Opt-in Challenge 1 (spoken + HUD head-movement). Requires "
            "--fasnet-policy active_research to trigger from UNCERTAIN liveness. "
            "Pose thresholds are EMPIRICAL_UNVALIDATED."
        ),
    )
    parser.add_argument(
        "--record-label",
        choices=("REAL", "PRINT", "SCREEN"),
        default=None,
        help=(
            "Start a labeled FasNet confidence session (scores/metadata JSONL only; "
            "no images). Press M to mark the true test interval."
        ),
    )
    parser.add_argument(
        "--record-dir",
        type=Path,
        default=None,
        help="Override calibration output directory (default: liveness/data/fasnet_calibration)",
    )
    args = parser.parse_args()

    if args.fasnet_policy == "active_research":
        fasnet_policy = active_research_policy()
    elif args.fasnet_policy == "binary_native":
        fasnet_policy = binary_native_policy()
    else:
        fasnet_policy = default_shadow_policy()

    if args.enable_challenge1 and args.fasnet_policy != "active_research":
        print(
            "[WARN] --enable-challenge1 is most useful with "
            "--fasnet-policy active_research (UNCERTAIN→CHALLENGE_REQUIRED). "
            "Shadow/binary rarely emit challenge_1_required."
        )

    recorder: CalibrationSessionRecorder | None = None
    samples: list[float] = []
    counts: Counter[str] = Counter()
    session_exit_reason = "not_started"
    cap: cv2.VideoCapture | None = None
    warm_thread: threading.Thread | None = None

    try:
        print(
            f"[INFO] FasNet policy mode={fasnet_policy.mode.value} "
            f"status={fasnet_policy.status} "
            f"gates REAL>={fasnet_policy.confident_real_min} "
            f"SPOOF>={fasnet_policy.confident_spoof_min} (uncalibrated) "
            f"challenge1={'on' if args.enable_challenge1 else 'off'}"
        )

        # Open camera first so preview is not blocked by model init (~30–40s).
        print(f"[INFO] Opening camera index={args.camera} (preview-first)...")
        t_cam = time.perf_counter()
        cap = _open_camera(args.camera)
        print(f"[INFO] Camera opened in {(time.perf_counter() - t_cam) * 1000:.0f} ms")
        _probe_camera_frames(cap, index=args.camera)

        pipeline_holder: dict = {"pipeline": None}
        models_ready = threading.Event()
        pipeline_error: list = []
        warm_thread = threading.Thread(
            target=_warm_pipeline_background,
            kwargs={
                "enrollment_db": args.enrollment_db,
                "threshold": args.threshold,
                "fasnet_policy": fasnet_policy,
                "pipeline_holder": pipeline_holder,
                "models_ready": models_ready,
                "pipeline_error": pipeline_error,
                "interrupt": interrupt,
            },
            name="model-warm",
            daemon=True,
        )
        warm_thread.start()

        if args.record_label:
            # Create immediately so preview stays live; backends filled when known.
            import os

            recorder = CalibrationSessionRecorder.create(
                label=args.record_label,
                camera_index=args.camera,
                out_dir=args.record_dir,
                fasnet_backend=os.environ.get("VIGIL_FASNET_BACKEND") or "default_onnx_if_available",
                facenet_backend=os.environ.get("VIGIL_FACENET_BACKEND") or "default_onnx_if_available",
                auth_policy_mode=fasnet_policy.mode.value,
            )
            print(
                f"[REC] Session {recorder.session_id} -> {recorder.out_path}\n"
                "[REC] Recording PAUSED until you press M (marks the physical test interval).\n"
                "[REC] Press M again to pause during transitions. No images are saved.\n"
                "[REC] Default policy=shadow; candidate thresholds compared offline only."
            )

        if args.mode == "continuous":
            samples, counts, session_exit_reason = run_continuous(
                pipeline_holder,
                cap,
                stale_ms=args.stale_ms,
                max_samples=args.max_samples,
                recorder=recorder,
                models_ready=models_ready,
                interrupt=interrupt,
                pipeline_error=pipeline_error,
                enable_challenge1=args.enable_challenge1,
            )
        else:
            samples, counts, session_exit_reason = run_manual(
                pipeline_holder,
                cap,
                stale_ms=args.stale_ms,
                max_samples=args.max_samples,
                recorder=recorder,
                models_ready=models_ready,
                interrupt=interrupt,
                pipeline_error=pipeline_error,
                enable_challenge1=args.enable_challenge1,
            )
    except KeyboardInterrupt:
        interrupt.set()
        session_exit_reason = "keyboard_interrupt_main"
        print("\n[INFO] Interrupted — shutting down cleanly...")
    except Exception:
        # Unexpected errors must remain visible.
        raise
    finally:
        interrupt.set()
        if warm_thread is not None and warm_thread.is_alive():
            warm_thread.join(timeout=5.0)
        if cap is not None:
            try:
                cap.release()
            except Exception:  # noqa: BLE001
                pass
        try:
            cv2.destroyAllWindows()
        except Exception:  # noqa: BLE001
            pass
        if recorder is not None:
            path = recorder.close()
            print(
                f"[REC] Closed session marked_records={recorder.marked_records} "
                f"path={path}"
            )

    _print_session_summary(samples, counts)
    print(f"[INFO] Session exit reason: {session_exit_reason}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted — session exit reason=keyboard_interrupt_main")
        raise SystemExit(0)
