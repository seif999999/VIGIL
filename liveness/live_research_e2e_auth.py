"""LOCAL RESEARCH end-to-end authentication test (camera).

Integrates SCRFD + frozen FasNet 5s median PAD + FaceNet (9.0) + Challenge 1.
Challenge 2 is NOT implemented — never simulated or reported as passed.

Does NOT change default live_friend_auth behavior, thresholds, models, or enrollment.
Production activation remains off.

Controls:
  Q/ESC — quit
  R     — reset / start a new attempt after ACCEPT/DENY
  D     — toggle debug lines

See RESEARCH_E2E.md.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from auth_pipeline import FriendAuthPipeline
from challenge1_session import Challenge1FailReason
from face_bridge import load_face_detector_module
from facenet_identity import (
    DEFAULT_ENROLLMENT_PATH,
    MATCH_THRESHOLD,
    MATCH_THRESHOLD_STATUS,
    FaceNetIdentity,
    identity_embedding_count,
    load_enrollments,
)
from fasnet_decision_policy import active_research_policy
from fasnet_liveness import FasNetClass, FasNetLiveness
from live_friend_auth import _open_camera
from live_runtime import (
    InterruptFlag,
    QUIT_KEY_CODES,
    frame_is_usable,
    poll_key_code,
    preview_window_closed,
    run_with_interrupt_guard,
)
from research_e2e_decision import (
    FACENET_UNCERTAIN_GAP,
    KNOWN_LIMITATION,
    STAGE_STATUS,
    E2EDecision,
    E2EPhase,
    IdentityWindowSummary,
    frozen_pad_banner,
    route_after_challenge1,
    route_after_pad_window,
    summarize_identity,
)

# BGR (match live_friend_auth left panel)
_GREEN = (40, 200, 40)
_RED = (40, 40, 220)
_AMBER = (0, 180, 255)
_WHITE = (240, 240, 240)
_PANEL = (24, 24, 24)
_BAR_BG = (60, 60, 60)
_BAR_FG = (0, 200, 255)
from research_e2e_challenge1 import (
    RESEARCH_C1_STEP_TIMEOUT_MS,
    ResearchChallenge1Timing,
    run_research_challenge1_interactive,
)
from research_e2e_pad import (
    FROZEN_PAD_MIN_VALID,
    FROZEN_PAD_REAL_MIN,
    FROZEN_PAD_SPOOF_MIN,
    FROZEN_PAD_WINDOW_S,
    FrozenPadWindowBuffer,
    eval_from_observation,
    utc_now_iso,
)

WINDOW = "VIGIL Research E2E Auth"


def create_scrfd_detector() -> Any:
    """Same SCRFD entry as live_friend_auth: ``FaceDetector()``."""
    return load_face_detector_module().FaceDetector()


@dataclass
class AttemptState:
    phase: E2EPhase = E2EPhase.COLLECTING_PAD
    last_decision: E2EDecision = E2EDecision.PENDING
    last_reason: str = "starting"
    joint_summary: str = "—"
    pad_state: str = "—"
    pad_median: float | None = None
    pad_n_valid: int = 0
    pad_n_frames_last: int = 0
    window_index: int = 1
    windows_completed: int = 0
    identity: IdentityWindowSummary | None = None
    invalid_retries: int = 0
    post_challenge1: bool = False
    expected_identity: str | None = None
    challenge1_note: str = ""
    challenge2_status: str = STAGE_STATUS["challenge2_voice"]
    final_at_utc: str | None = None
    session_t0: float = field(default_factory=time.perf_counter)
    live_fasnet_real: float | None = None
    live_fasnet_spoof: float | None = None
    live_fasnet_class: str | None = None
    event_log: list[str] = field(default_factory=list)
    distances_in_window: list[tuple[float, str | None]] = field(default_factory=list)

    def log(self, msg: str) -> None:
        line = f"{utc_now_iso()}  {msg}"
        self.event_log.append(line)
        print(f"[E2E] {line}")


def _crop_face(frame_bgr: np.ndarray, face: Any) -> np.ndarray | None:
    x1, y1, x2, y2 = int(face.x1), int(face.y1), int(face.x2), int(face.y2)
    h, w = frame_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame_bgr[y1:y2, x1:x2]
    return crop if crop.size else None


def _put(
    frame: np.ndarray,
    text: str,
    org: tuple[int, int],
    color=_WHITE,
    scale: float = 0.48,
    thick: int = 1,
) -> None:
    cv2.putText(
        frame,
        text[:64],
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thick,
        cv2.LINE_AA,
    )


def _draw_progress_bar(
    frame: np.ndarray,
    *,
    x: int,
    y: int,
    width: int,
    height: int,
    frac: float,
) -> None:
    frac = max(0.0, min(1.0, frac))
    cv2.rectangle(frame, (x, y), (x + width, y + height), _BAR_BG, -1)
    fill = int(width * frac)
    if fill > 0:
        cv2.rectangle(frame, (x, y), (x + fill, y + height), _BAR_FG, -1)
    cv2.rectangle(frame, (x, y), (x + width, y + height), _WHITE, 1)


def _draw_hud(
    frame: np.ndarray,
    *,
    attempt: AttemptState,
    pad_buf: FrozenPadWindowBuffer,
    perf_now: float,
    face_decision: str,
    face_count: int,
    live_distance: float | None,
    live_recognized: bool | None,
    camera_state: str,
    debug: bool,
) -> None:
    """Detailed left panel styled after live_friend_auth.draw_auth."""
    h, w = frame.shape[:2]
    panel_w = min(460, max(320, w // 3))
    x0, y0 = 8, 8
    x1 = x0 + panel_w
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, h - 8), _PANEL, -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    def line(text: str, y: int, color=_WHITE, scale=0.48, thick=1) -> int:
        _put(frame, text, (x0 + 10, y), color, scale, thick)
        return y + int(22 * max(0.9, scale / 0.48))

    def section(title: str, y: int) -> int:
        return line(title, y + 4, _WHITE, 0.5, 1)

    elapsed_win = pad_buf.elapsed_s(perf_now)
    remain = max(0.0, FROZEN_PAD_WINDOW_S - elapsed_win)
    frac = (
        min(1.0, elapsed_win / FROZEN_PAD_WINDOW_S)
        if attempt.phase
        in (
            E2EPhase.COLLECTING_PAD,
            E2EPhase.COLLECTING_POST_C1_PAD,
            E2EPhase.RETRY,
        )
        else (1.0 if attempt.pad_state not in ("—",) else 0.0)
    )
    session_s = max(0.0, perf_now - attempt.session_t0)

    if attempt.phase == E2EPhase.ACCEPT:
        title, color = "ACCEPT", _GREEN
    elif attempt.phase in (E2EPhase.DENY, E2EPhase.CHALLENGE2_BLOCKED):
        title, color = attempt.phase.value, _RED
    elif attempt.phase in (
        E2EPhase.AWAITING_CHALLENGE1,
        E2EPhase.CHALLENGE1_RUNNING,
    ):
        title, color = "CHALLENGE 1", _AMBER
    else:
        title, color = "EVALUATING", _AMBER

    idn = attempt.identity
    id_state = idn.state.value if idn is not None else "—"
    id_name = (idn.best_identity if idn and idn.best_identity else "—")
    win_l2 = idn.best_distance if idn and idn.best_distance is not None else None

    y = y0 + 32
    y = line("RESEARCH E2E (not production)", y, _AMBER, 0.45, 1)
    y = line(title, y, color, 1.05, 3)
    y = line(f"Phase: {attempt.phase.value}", y, color, 0.5, 1)
    y = line(f"Decision: {attempt.last_decision.value}", y, color, 0.5, 1)
    y = line(f"Reason: {attempt.last_reason}", y, _AMBER, 0.42, 1)
    y = line(f"Joint: {attempt.joint_summary}", y, _WHITE, 0.45, 1)

    y = section("--- PAD (frozen 5s median) ---", y)
    y = line(
        f"Window #{attempt.window_index}  completed={attempt.windows_completed}",
        y,
        _WHITE,
        0.48,
    )
    y = line(
        f"Countdown: {remain:.1f}s / {FROZEN_PAD_WINDOW_S:.0f}s",
        y,
        _AMBER if remain > 0 else _GREEN,
        0.55,
        2,
    )
    _draw_progress_bar(frame, x=x0 + 10, y=y - 8, width=panel_w - 28, height=14, frac=frac)
    y += 22
    y = line(
        f"Obs in window: {pad_buf.n_frames()}  "
        f"valid(last)={attempt.pad_n_valid}/{FROZEN_PAD_MIN_VALID}",
        y,
        _WHITE,
        0.45,
    )
    med = f"{attempt.pad_median:.3f}" if attempt.pad_median is not None else "—"
    y = line(f"Aggregated PAD: {attempt.pad_state}  median_real={med}", y, _WHITE, 0.48)
    live_r = (
        f"{attempt.live_fasnet_real:.3f}"
        if attempt.live_fasnet_real is not None
        else "—"
    )
    y = line(
        f"Live FasNet: class={attempt.live_fasnet_class or '—'}  real={live_r}",
        y,
        _WHITE,
        0.45,
    )
    y = line(
        f"Gates: REAL>={FROZEN_PAD_REAL_MIN} SPOOF>={FROZEN_PAD_SPOOF_MIN}",
        y,
        _WHITE,
        0.42,
    )

    y = section("--- FaceNet identity ---", y)
    live_l2 = f"{live_distance:.2f}" if live_distance is not None else "—"
    live_tag = (
        "MATCH"
        if live_recognized
        else ("UNKNOWN" if live_recognized is False else "—")
    )
    y = line(f"Live L2: {live_l2}  →  {live_tag}", y, _WHITE, 0.5, 1)
    y = line(
        f"Threshold: {MATCH_THRESHOLD:.1f} ({MATCH_THRESHOLD_STATUS})",
        y,
        _WHITE,
        0.42,
    )
    win_l2_s = f"{win_l2:.2f}" if win_l2 is not None else "—"
    y = line(
        f"Window ID: {id_state}  best_L2={win_l2_s}  cand={id_name}",
        y,
        _GREEN if id_state == "MATCH" else (_RED if id_state == "UNKNOWN" else _AMBER),
        0.48,
    )
    y = line("Identity UNCERTAIN zone: NOT AVAILABLE", y, _AMBER, 0.40)

    y = section("--- Challenges ---", y)
    y = line(
        f"C1: {attempt.challenge1_note or 'idle (1 instruction; only if PAD UNCERTAIN+MATCH)'}",
        y,
        _WHITE,
        0.42,
    )
    y = line(f"C2: {attempt.challenge2_status}", y, _AMBER, 0.42)

    y = section("--- Capture ---", y)
    y = line(f"Faces: {face_count}  {face_decision}  cam={camera_state}", y, _WHITE, 0.45)
    y = line(f"Session elapsed: {session_s:.0f}s", y, _WHITE, 0.45)
    if attempt.final_at_utc:
        y = line(f"Final at: {attempt.final_at_utc}", y, color, 0.40)

    if debug:
        y = section("--- Debug ---", y)
        y = line(KNOWN_LIMITATION[:58] + "…", y, _AMBER, 0.38)
        y = line(FACENET_UNCERTAIN_GAP[:58] + "…", y, _AMBER, 0.38)

    y = line("Q/ESC=quit  R=reset  D=debug", y + 4, _WHITE, 0.42)

    # Top-right camera banner (clear of left panel)
    cam_color = _GREEN if camera_state == "frames_ok" else _AMBER
    _put(frame, f"Camera: {camera_state.upper()}", (max(x1 + 8, w - 220), 28), cam_color, 0.55, 2)


def _reset_attempt(attempt: AttemptState, pad_buf: FrozenPadWindowBuffer) -> None:
    attempt.phase = E2EPhase.COLLECTING_PAD
    attempt.last_decision = E2EDecision.PENDING
    attempt.last_reason = "reset"
    attempt.joint_summary = "—"
    attempt.pad_state = "—"
    attempt.pad_median = None
    attempt.pad_n_valid = 0
    attempt.pad_n_frames_last = 0
    attempt.window_index = attempt.windows_completed + 1
    attempt.identity = None
    attempt.invalid_retries = 0
    attempt.post_challenge1 = False
    attempt.expected_identity = None
    attempt.challenge1_note = ""
    attempt.final_at_utc = None
    attempt.distances_in_window.clear()
    pad_buf.reset()
    attempt.log(f"attempt reset -> COLLECTING_PAD window=#{attempt.window_index}")


def _apply_pad_route(
    attempt: AttemptState,
    pad_buf: FrozenPadWindowBuffer,
    decision_state: str,
    median_real: float | None,
    n_valid: int,
    n_frames: int,
) -> None:
    identity = summarize_identity(attempt.distances_in_window)
    attempt.identity = identity
    attempt.pad_state = decision_state
    attempt.pad_median = median_real
    attempt.pad_n_valid = n_valid
    attempt.pad_n_frames_last = n_frames
    attempt.windows_completed += 1
    route = route_after_pad_window(
        pad_state=decision_state,
        identity=identity,
        post_challenge1=attempt.post_challenge1,
        invalid_retry_count=attempt.invalid_retries,
    )
    attempt.last_decision = route.decision
    attempt.last_reason = route.reason
    attempt.phase = route.phase
    attempt.joint_summary = route.joint_summary
    attempt.log(
        f"window#{attempt.window_index} PAD={decision_state} median={median_real} "
        f"n_valid={n_valid}/{n_frames} id={identity.state.value} "
        f"L2={identity.best_distance} -> {route.decision.value} ({route.reason})"
    )
    if route.decision == E2EDecision.RETRY:
        attempt.invalid_retries += 1
        attempt.distances_in_window.clear()
        pad_buf.reset()
        attempt.window_index = attempt.windows_completed + 1
        attempt.phase = (
            E2EPhase.COLLECTING_POST_C1_PAD
            if attempt.post_challenge1
            else E2EPhase.COLLECTING_PAD
        )
        attempt.last_decision = E2EDecision.PENDING
        attempt.last_reason = route.reason
    elif route.decision in (
        E2EDecision.ACCEPT,
        E2EDecision.DENY,
        E2EDecision.CHALLENGE2_UNAVAILABLE,
    ):
        attempt.final_at_utc = utc_now_iso()
        attempt.distances_in_window.clear()
        pad_buf.reset()
    elif route.decision == E2EDecision.CHALLENGE1_REQUIRED:
        attempt.expected_identity = identity.best_identity
        attempt.distances_in_window.clear()
        pad_buf.reset()
        attempt.window_index = attempt.windows_completed + 1


def run_session(args: argparse.Namespace, interrupt: InterruptFlag) -> int:
    banner = frozen_pad_banner()
    print("[INFO] VIGIL LOCAL RESEARCH E2E - production_activation=False")
    print(f"[INFO] stages={banner['stages']}")
    print(f"[INFO] frozen PAD: {FROZEN_PAD_WINDOW_S}s median REAL>={FROZEN_PAD_REAL_MIN} "
          f"SPOOF>={FROZEN_PAD_SPOOF_MIN} min_valid={FROZEN_PAD_MIN_VALID}")
    print(f"[INFO] FaceNet threshold={MATCH_THRESHOLD} ({MATCH_THRESHOLD_STATUS})")
    print(f"[WARN] {KNOWN_LIMITATION}")
    print(f"[WARN] {FACENET_UNCERTAIN_GAP}")
    print("[WARN] Challenge 2 voice: NOT_IMPLEMENTED - never auto-passed")
    print("[INFO] Challenge 1 only when PAD=UNCERTAIN + FaceNet=MATCH (not REAL+MATCH)")

    enroll_path = Path(args.enrollment_db)
    db = load_enrollments(enroll_path)
    identities = db.get("identities") or {}
    n_emb = sum(identity_embedding_count(db, i) for i in identities)
    if n_emb <= 0:
        print(f"[FAIL] No enrollments in {enroll_path}", file=sys.stderr)
        return 2

    detector = create_scrfd_detector()
    # Pipeline used for Challenge 1 interactive re-eval (single-frame diagnostic).
    # Final ACCEPT is gated by frozen temporal PAD in this tool, not C1's ACCEPT.
    pipeline = FriendAuthPipeline(
        detector,
        enrollment_path=enroll_path,
        match_threshold=MATCH_THRESHOLD,
        fasnet_policy=active_research_policy(),
    )
    print("[INFO] Warming FasNet + FaceNet...")
    pipeline.warm_up()
    fasnet: FasNetLiveness = pipeline.fasnet
    facenet: FaceNetIdentity = pipeline.facenet

    pad_buf = FrozenPadWindowBuffer()
    attempt = AttemptState()
    attempt.log("session start")

    cap = _open_camera(args.camera)
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    last_eval_t = 0.0
    # 0 = evaluate every usable preview frame (tests / high-rate research).
    eval_interval_s = max(0.0, float(args.min_interval_ms) / 1000.0)
    debug = False
    face_decision = "NO_FACE"
    face_count = 0
    live_distance: float | None = None
    live_recognized: bool | None = None
    box_xyxy: tuple[int, int, int, int] | None = None
    camera_state = "frames_ok"

    try:
        while not interrupt.is_set:
            # Terminal phases: wait for R or quit
            if attempt.phase in (
                E2EPhase.ACCEPT,
                E2EPhase.DENY,
                E2EPhase.CHALLENGE2_BLOCKED,
            ):
                ok, frame = cap.read()
                display = (
                    frame.copy()
                    if ok and frame is not None
                    else np.zeros((480, 640, 3), dtype=np.uint8)
                )
                _draw_hud(
                    display,
                    attempt=attempt,
                    pad_buf=pad_buf,
                    perf_now=time.perf_counter(),
                    face_decision=face_decision,
                    face_count=face_count,
                    live_distance=live_distance,
                    live_recognized=live_recognized,
                    camera_state=camera_state,
                    debug=debug,
                )
                cv2.imshow(WINDOW, display)
                key = poll_key_code(1)
                if key in QUIT_KEY_CODES:
                    break
                if key in (ord("r"), ord("R")):
                    _reset_attempt(attempt, pad_buf)
                if key in (ord("d"), ord("D")):
                    debug = not debug
                if preview_window_closed(WINDOW):
                    break
                continue

            # Launch Challenge 1 when required
            if attempt.phase == E2EPhase.AWAITING_CHALLENGE1:
                attempt.phase = E2EPhase.CHALLENGE1_RUNNING
                attempt.challenge1_note = "running..."
                attempt.log(
                    f"starting Challenge 1 (expected_identity={attempt.expected_identity})"
                )
                c1_timing = ResearchChallenge1Timing(
                    step_timeout_ms=float(args.c1_timeout_ms),
                    baseline_ms=float(args.c1_baseline_ms),
                    hold_ms=float(args.c1_hold_ms),
                    post_speech_pause_s=float(args.c1_post_speech_pause_s),
                )
                c1 = run_research_challenge1_interactive(
                    pipeline,
                    cap,
                    window=WINDOW,
                    expected_identity=attempt.expected_identity,
                    timing=c1_timing,
                    speak=not args.no_tts,
                    interrupt_check=lambda: interrupt.is_set,
                )
                gesture_ok = bool(c1.succeeded_gesture)
                fail = c1.fail_reason.value if c1.fail_reason else "NONE"
                c1_out = c1.outcome.decision.value if c1.outcome else "None"
                c1_reason = c1.outcome.reason if c1.outcome else ""
                attempt.challenge1_note = (
                    f"gesture={'OK' if gesture_ok else 'FAIL'} "
                    f"c1_diag={c1_out}/{c1_reason} fail={fail}"
                )
                attempt.log(attempt.challenge1_note)
                # Gesture-only gate; ignore C1 single-frame ACCEPT for final decision.
                # Ctrl+C during C1 sets fail_reason=CANCELLED via interrupt_check.
                if c1.fail_reason == Challenge1FailReason.CANCELLED:
                    attempt.phase = E2EPhase.DENY
                    attempt.last_decision = E2EDecision.DENY
                    attempt.last_reason = "challenge1_cancelled"
                    attempt.final_at_utc = utc_now_iso()
                    continue
                route = route_after_challenge1(
                    succeeded_gesture=gesture_ok,
                    fail_reason=fail,
                )
                attempt.last_decision = route.decision
                attempt.last_reason = route.reason
                attempt.phase = route.phase
                if route.phase == E2EPhase.COLLECTING_POST_C1_PAD:
                    attempt.post_challenge1 = True
                    attempt.invalid_retries = 0
                    pad_buf.reset()
                    attempt.distances_in_window.clear()
                    attempt.window_index = attempt.windows_completed + 1
                    attempt.log(
                        f"post-C1: collecting fresh PAD window #{attempt.window_index}"
                    )
                elif route.phase == E2EPhase.DENY:
                    attempt.final_at_utc = utc_now_iso()
                continue

            ok, frame = cap.read()
            if not ok or frame is None or not frame_is_usable(frame):
                camera_state = "waiting_for_frames"
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                _draw_hud(
                    blank,
                    attempt=attempt,
                    pad_buf=pad_buf,
                    perf_now=time.perf_counter(),
                    face_decision="NO_FRAME",
                    face_count=0,
                    live_distance=None,
                    live_recognized=None,
                    camera_state=camera_state,
                    debug=debug,
                )
                cv2.imshow(WINDOW, blank)
            else:
                camera_state = "frames_ok"
                now = time.perf_counter()
                if now - last_eval_t >= eval_interval_s:
                    last_eval_t = now
                    det = detector.detect(frame)
                    face_count = int(det.face_count)
                    face_decision = det.decision.value
                    live_distance = None
                    live_recognized = None
                    box_xyxy = None

                    fas_class: str | None = None
                    real_avg: float | None = None
                    spoof_c: float | None = None
                    fas_err: str | None = None

                    collecting = attempt.phase in (
                        E2EPhase.COLLECTING_PAD,
                        E2EPhase.COLLECTING_POST_C1_PAD,
                        E2EPhase.RETRY,
                    )
                    if collecting and attempt.phase == E2EPhase.RETRY:
                        attempt.phase = (
                            E2EPhase.COLLECTING_POST_C1_PAD
                            if attempt.post_challenge1
                            else E2EPhase.COLLECTING_PAD
                        )

                    if face_decision == "FACE_OK" and face_count == 1 and det.faces:
                        f0 = det.faces[0]
                        box_xyxy = (int(f0.x1), int(f0.y1), int(f0.x2), int(f0.y2))
                        # FasNet on full frame (native crops) — always for PAD.
                        fas = fasnet.analyze_full_frame(frame, f0.bbox)
                        fas_err = fas.error
                        fas_class = fas.predicted_class.value
                        real_avg = fas.real_score_avg
                        spoof_c = fas.spoof_score_combined
                        attempt.live_fasnet_class = fas_class
                        attempt.live_fasnet_real = real_avg
                        attempt.live_fasnet_spoof = spoof_c

                        # FaceNet independent of FasNet prediction.
                        crop = _crop_face(frame, f0)
                        if crop is not None:
                            match = facenet.match_crop(
                                crop,
                                pipeline.enrollments,
                                threshold=MATCH_THRESHOLD,
                                enrollment_index=pipeline.enrollment_index,
                            )
                            if match.error is None and match.best_distance is not None:
                                live_distance = match.best_distance
                                live_recognized = bool(match.recognized)
                                if collecting:
                                    attempt.distances_in_window.append(
                                        (
                                            float(match.best_distance),
                                            match.best_identity_candidate
                                            or match.identity,
                                        )
                                    )
                    elif face_decision != "FACE_OK":
                        fas_class = None
                        attempt.live_fasnet_class = None
                        attempt.live_fasnet_real = None
                        attempt.live_fasnet_spoof = None

                    if collecting:
                        row = eval_from_observation(
                            face_decision=face_decision,
                            fasnet_native_class=fas_class,
                            fasnet_real_score_avg=real_avg,
                            fasnet_spoof_score_combined=spoof_c,
                            fasnet_error=fas_err,
                        )
                        if (
                            face_decision == "FACE_OK"
                            and fas_class == FasNetClass.ERROR.value
                        ):
                            row["fasnet_native_class"] = "ERROR"
                        n_before = pad_buf.n_frames()
                        win = pad_buf.add(perf_t=now, eval_row=row)
                        if win is not None:
                            _apply_pad_route(
                                attempt,
                                pad_buf,
                                win.state,
                                win.median_real,
                                win.n_valid,
                                n_frames=win.n_frames,
                            )
                        elif n_before == 0 and pad_buf.n_frames() == 1:
                            # Timer reset: first observation of a new window.
                            attempt.log(
                                f"PAD window #{attempt.window_index} started "
                                f"(post_c1={attempt.post_challenge1})"
                            )

                display = frame.copy()
                if box_xyxy is not None:
                    x1, y1, x2, y2 = box_xyxy
                    cv2.rectangle(
                        display,
                        (x1, y1),
                        (x2, y2),
                        (80, 220, 80) if live_recognized else (80, 80, 240),
                        2,
                    )
                _draw_hud(
                    display,
                    attempt=attempt,
                    pad_buf=pad_buf,
                    perf_now=time.perf_counter(),
                    face_decision=face_decision,
                    face_count=face_count,
                    live_distance=live_distance,
                    live_recognized=live_recognized,
                    camera_state=camera_state,
                    debug=debug,
                )
                cv2.imshow(WINDOW, display)

            key = poll_key_code(1)
            if key in QUIT_KEY_CODES:
                break
            if key in (ord("r"), ord("R")):
                _reset_attempt(attempt, pad_buf)
            if key in (ord("d"), ord("D")):
                debug = not debug
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
        attempt.log(
            f"session end phase={attempt.phase.value} "
            f"decision={attempt.last_decision.value} reason={attempt.last_reason}"
        )
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "LOCAL RESEARCH end-to-end auth test: frozen 5s FasNet PAD + FaceNet "
            "9.0 + Challenge 1. Challenge 2 not implemented. Default live auth unchanged."
        )
    )
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--enrollment-db", type=Path, default=DEFAULT_ENROLLMENT_PATH)
    p.add_argument(
        "--min-interval-ms",
        type=float,
        default=100.0,
        help="Min ms between FasNet/FaceNet evaluations",
    )
    p.add_argument(
        "--no-tts",
        action="store_true",
        help="Disable Challenge 1 spoken instructions",
    )
    p.add_argument(
        "--c1-timeout-ms",
        type=float,
        default=RESEARCH_C1_STEP_TIMEOUT_MS,
        help="Research Challenge 1 move-window ms after speech (default 18000)",
    )
    p.add_argument(
        "--c1-baseline-ms",
        type=float,
        default=700.0,
        help="Research Challenge 1 baseline settle ms before instruction",
    )
    p.add_argument(
        "--c1-hold-ms",
        type=float,
        default=400.0,
        help="Research Challenge 1 hold ms at target pose",
    )
    p.add_argument(
        "--c1-post-speech-pause-s",
        type=float,
        default=0.35,
        help="Pause after TTS before move window opens",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    interrupt = InterruptFlag()
    interrupt.install()
    code = 0

    def _run() -> None:
        nonlocal code
        code = run_session(args, interrupt)

    run_with_interrupt_guard(_run, interrupt)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
