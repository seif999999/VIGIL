"""Challenge 1 runner for LOCAL research E2E only.

Differs from production ``challenge1_runner``:
  - Exactly one randomized instruction (LOOK_LEFT/RIGHT/UP/DOWN)
  - Blocking TTS finishes before the movement response window starts
  - Longer configurable timeouts (research-mode)
  - Clearer LISTEN / MOVE HUD

Does not change default live_friend_auth Challenge 1 behavior.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from auth_pipeline import FriendAuthPipeline
from challenge1_pose import HeadDirection, PoseThresholds
from challenge1_runner import draw_challenge1_hud
from challenge1_session import (
    Challenge1Config,
    Challenge1HudState,
    Challenge1Phase,
    Challenge1Result,
    Challenge1Session,
    sample_instruction_sequence,
)
from challenge1_tts import speak_instruction
from fasnet_decision_policy import evaluate_fasnet_policy

# Research-mode defaults (physical testing — not production).
RESEARCH_C1_N_STEPS = 1
RESEARCH_C1_STEP_TIMEOUT_MS = 18000.0
RESEARCH_C1_BASELINE_MS = 700.0
RESEARCH_C1_HOLD_MS = 400.0
RESEARCH_C1_POST_SPEECH_PAUSE_S = 0.35


@dataclass(frozen=True)
class ResearchChallenge1Timing:
    step_timeout_ms: float = RESEARCH_C1_STEP_TIMEOUT_MS
    baseline_ms: float = RESEARCH_C1_BASELINE_MS
    hold_ms: float = RESEARCH_C1_HOLD_MS
    post_speech_pause_s: float = RESEARCH_C1_POST_SPEECH_PAUSE_S


def research_challenge1_config(
    timing: ResearchChallenge1Timing | None = None,
) -> Challenge1Config:
    t = timing or ResearchChallenge1Timing()
    return Challenge1Config(
        thresholds=PoseThresholds(
            hold_ms=t.hold_ms,
            step_timeout_ms=t.step_timeout_ms,
        ),
        baseline_ms=t.baseline_ms,
        start_timer_on_instruct=False,  # arm move timer only after TTS
    )


def sample_single_instruction(
    *,
    rng=None,
) -> tuple[HeadDirection, ...]:
    """Exactly one direction — multi-step sequences disabled for research E2E."""
    return sample_instruction_sequence(n_steps=RESEARCH_C1_N_STEPS, rng=rng)


def draw_research_challenge1_hud(
    frame: np.ndarray,
    hud: Challenge1HudState,
    *,
    listen_mode: bool = False,
    move_window_s: float | None = None,
    debug: bool = False,
) -> None:
    """Production C1 HUD plus explicit LISTEN / MOVE cues for research."""
    draw_challenge1_hud(frame, hud, debug=debug)
    h, w = frame.shape[:2]
    banner_y = 28
    if listen_mode:
        text = f"LISTEN: {hud.current_instruction or '—'}"
        color = (0, 200, 255)
    elif hud.phase in ("WAIT_MOVE", "HOLD"):
        rem_s = hud.remaining_ms / 1000.0
        total = move_window_s if move_window_s and move_window_s > 0 else None
        if total is not None:
            text = f"MOVE NOW: {hud.current_instruction}  ({rem_s:.1f}s / {total:.0f}s)"
        else:
            text = f"MOVE NOW: {hud.current_instruction}  ({rem_s:.1f}s left)"
        color = (80, 220, 80)
    else:
        text = f"C1 {hud.phase}: {hud.current_instruction or '—'}"
        color = (220, 220, 220)

    cv2.rectangle(frame, (8, h - 56), (min(w - 8, 520), h - 8), (20, 20, 20), -1)
    cv2.putText(
        frame,
        text[:70],
        (18, h - 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )
    # Countdown bar during move window
    if hud.phase in ("WAIT_MOVE", "HOLD") and move_window_s and move_window_s > 0:
        frac = max(0.0, min(1.0, hud.remaining_ms / (move_window_s * 1000.0)))
        bar_x, bar_y, bar_w, bar_h = 18, h - 48, min(480, w - 40), 10
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
        fill = int(bar_w * frac)
        if fill > 0:
            cv2.rectangle(
                frame,
                (bar_x, bar_y),
                (bar_x + fill, bar_y + bar_h),
                (0, 200, 255),
                -1,
            )


def run_research_challenge1_interactive(
    pipeline: FriendAuthPipeline,
    cap: cv2.VideoCapture,
    *,
    window: str = "VIGIL Research E2E Auth",
    expected_identity: str | None = None,
    timing: ResearchChallenge1Timing | None = None,
    speak: bool = True,
    interrupt_check: Callable[[], bool] | None = None,
) -> Challenge1Result:
    """
    Single-instruction Challenge 1 for research E2E.

    Spoken instruction completes before WAIT_MOVE. Gesture alone never ACCEPT —
    parent E2E tool still requires fresh frozen PAD + FaceNet after return.
    """
    timing = timing or ResearchChallenge1Timing()
    config = research_challenge1_config(timing)
    instructions = sample_single_instruction()
    assert len(instructions) == 1
    session = Challenge1Session(
        instructions,
        config=config,
        expected_identity=expected_identity,
    )
    debug = False
    spoken_for_step = -1
    listen_mode = False
    move_window_s = timing.step_timeout_ms / 1000.0

    print(
        f"[RESEARCH-C1] session={session.session_id} "
        f"single_instruction={instructions[0].value} "
        f"move_timeout_ms={timing.step_timeout_ms:.0f} "
        f"blocking_tts={speak}"
    )

    while not session.done:
        if interrupt_check and interrupt_check():
            session.cancel()
            break
        ok, frame = cap.read()
        if not ok or frame is None:
            session.observe_detection(
                face_count=0,
                landmarks=None,
                camera_ok=False,
            )
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            draw_research_challenge1_hud(
                blank,
                session.hud(),
                listen_mode=listen_mode,
                move_window_s=move_window_s,
                debug=debug,
            )
            cv2.imshow(window, blank)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                session.cancel()
            continue

        t_cap = time.perf_counter()
        det = pipeline.face_detector.detect(frame)
        age_ms = (time.perf_counter() - t_cap) * 1000.0
        lm = None
        if det.face_count == 1:
            lm = det.faces[0].landmarks

        phase = session.observe_detection(
            face_count=det.face_count,
            landmarks=lm,
            frame_age_ms=age_ms,
            camera_ok=True,
        )

        if phase == Challenge1Phase.INSTRUCT and spoken_for_step != session.hud().current_index:
            spoken_for_step = session.hud().current_index
            instr = session.instructions[session.hud().current_index]
            listen_mode = True
            # Show instruction before speech; do not open move window yet.
            draw_research_challenge1_hud(
                frame,
                session.hud(),
                listen_mode=True,
                move_window_s=move_window_s,
                debug=debug,
            )
            if det.face_count == 1:
                f = det.faces[0]
                cv2.rectangle(
                    frame,
                    (int(f.x1), int(f.y1)),
                    (int(f.x2), int(f.y2)),
                    (0, 200, 255),
                    2,
                )
            cv2.imshow(window, frame)
            cv2.waitKey(1)
            print(f"[RESEARCH-C1] LISTEN -> speaking: {instr.spoken!r}")
            if speak:
                # Blocking: movement window must not start mid-sentence.
                speak_instruction(instr.spoken, blocking=True)
            if timing.post_speech_pause_s > 0:
                time.sleep(timing.post_speech_pause_s)
            listen_mode = False
            session.on_instruction_spoken()
            print(
                f"[RESEARCH-C1] MOVE window open ({move_window_s:.0f}s) "
                f"for {instr.value}"
            )
            continue

        if phase == Challenge1Phase.PASSED_STEP:
            session.advance_after_step_pass()
            spoken_for_step = -1

        if session.phase == Challenge1Phase.REEVAL:
            # Diagnostic single-frame re-eval only; parent E2E still requires
            # fresh frozen 5s PAD + FaceNet before research ACCEPT.
            result = pipeline.process_verified_frame(frame)
            policy_ev = evaluate_fasnet_policy(result.fasnet, pipeline.fasnet_policy)
            session.submit_reeval(
                auth_liveness=policy_ev.auth_liveness_state,
                fasnet=result.fasnet,
                identity=result.identity,
                face_count=result.detection.face_count,
                base_outcome=result.outcome,
            )

        draw_research_challenge1_hud(
            frame,
            session.hud(),
            listen_mode=listen_mode,
            move_window_s=move_window_s,
            debug=debug,
        )
        if det.face_count == 1:
            f = det.faces[0]
            cv2.rectangle(
                frame,
                (int(f.x1), int(f.y1)),
                (int(f.x2), int(f.y2)),
                (0, 200, 255),
                2,
            )
        cv2.imshow(window, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q"), ord("Q")):
            session.cancel()
            break
        if key in (ord("d"), ord("D")):
            debug = not debug

    result = session.result()
    print(
        f"[RESEARCH-C1] done gesture_ok={result.succeeded_gesture} "
        f"fail={result.fail_reason.value} "
        f"decision={result.outcome.decision.value if result.outcome else None} "
        f"reason={result.outcome.reason if result.outcome else None}"
    )
    return result
