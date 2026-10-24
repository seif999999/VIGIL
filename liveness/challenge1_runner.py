"""Interactive Challenge 1 loop (camera preview + SCRFD pose + TTS + re-eval).

Called only when auth already returned CHALLENGE_REQUIRED and --enable-challenge1.
HMAC: use only frames from the same trusted capture path as the parent session.
"""

from __future__ import annotations

import time
from typing import Any, Callable

import cv2
import numpy as np

from auth_decision import AuthDecision, AuthOutcome
from auth_pipeline import AuthPipelineResult, FriendAuthPipeline
from challenge1_session import (
    Challenge1Config,
    Challenge1FailReason,
    Challenge1HudState,
    Challenge1Phase,
    Challenge1Result,
    Challenge1Session,
    sample_instruction_sequence,
)
from challenge1_tts import speak_instruction
from fasnet_decision_policy import evaluate_fasnet_policy


def draw_challenge1_hud(
    frame: np.ndarray,
    hud: Challenge1HudState,
    *,
    debug: bool = False,
) -> None:
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (min(460, w - 8), h - 8), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    def put(text: str, y: int, color=(255, 255, 255), scale=0.55, thick=1) -> int:
        cv2.putText(
            frame,
            text[:58],
            (18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thick,
            cv2.LINE_AA,
        )
        return y + int(24 * max(0.85, scale / 0.55))

    y = 40
    y = put("CHALLENGE 1 — HEAD MOVEMENT", y, (0, 200, 255), 0.7, 2)
    seq = " → ".join(hud.instructions)
    y = put(f"Sequence: {seq}", y, (220, 220, 220), 0.5)
    y = put(
        f"Step {hud.current_index + 1}/{len(hud.instructions)}: "
        f"{hud.current_instruction or '-'}",
        y,
        (0, 255, 255),
        0.65,
        2,
    )
    y = put(f"Phase: {hud.phase}", y, (200, 200, 200), 0.48)
    y = put(f"Time left: {hud.remaining_ms / 1000.0:.1f}s", y, (200, 200, 200), 0.5)
    prog = max(0.0, min(1.5, hud.progress))
    y = put(f"Motion progress: {prog:.0%}", y, (180, 255, 180), 0.5)
    mv = "YES" if hud.movement_verified or hud.step_verified else "NO"
    y = put(f"Movement verified: {mv}", y, (0, 255, 0) if mv == "YES" else (0, 180, 255), 0.55, 2)
    if hud.reeval_liveness:
        y = put(f"Fresh liveness: {hud.reeval_liveness}", y, (200, 200, 255), 0.5)
    if hud.reeval_identity:
        y = put(f"Fresh identity: {hud.reeval_identity}", y, (200, 200, 255), 0.5)
    if hud.final_decision:
        col = (0, 255, 0) if hud.final_decision == "ACCEPT" else (0, 0, 255)
        if hud.final_decision == "CHALLENGE_REQUIRED":
            col = (0, 200, 255)
        y = put(f"Result: {hud.final_decision}", y, col, 0.65, 2)
        if hud.final_reason:
            y = put(f"Reason: {hud.final_reason}", y, col, 0.45)
    if hud.fail_reason and hud.fail_reason != "NONE":
        y = put(f"Fail: {hud.fail_reason}", y, (0, 0, 255), 0.5)
    y = put("Q/ESC=abort challenge   D=debug", y, (180, 180, 180), 0.45)
    if debug:
        y = put(f"session={hud.session_id[:8]}…", y, (160, 160, 160), 0.4)


def run_challenge1_interactive(
    pipeline: FriendAuthPipeline,
    cap: cv2.VideoCapture,
    *,
    window: str = "VIGIL live auth",
    expected_identity: str | None = None,
    n_steps: int | None = None,
    config: Challenge1Config | None = None,
    speak: bool = True,
    interrupt_check: Callable[[], bool] | None = None,
) -> Challenge1Result:
    """
    Take over the preview briefly: pose-track with SCRFD, then one+ full re-evals.
    Does not ACCEPT on gesture alone.
    """
    instructions = sample_instruction_sequence(n_steps=n_steps)
    session = Challenge1Session(
        instructions,
        config=config,
        expected_identity=expected_identity,
    )
    debug = False
    spoken_for_step = -1

    print(
        f"[CHALLENGE1] session={session.session_id} "
        f"instructions={[d.value for d in instructions]}"
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
            draw_challenge1_hud(blank, session.hud(), debug=debug)
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
            if speak:
                speak_instruction(instr.spoken, blocking=False)
            session.on_instruction_spoken()

        if phase == Challenge1Phase.PASSED_STEP:
            session.advance_after_step_pass()
            spoken_for_step = -1

        if session.phase == Challenge1Phase.REEVAL:
            # Fresh full auth on current frame (SCRFD+FasNet+FaceNet)
            result = pipeline.process_verified_frame(frame)
            policy_ev = evaluate_fasnet_policy(result.fasnet, pipeline.fasnet_policy)
            # Gate challenge1 status through decide — use auth_liveness from policy
            session.submit_reeval(
                auth_liveness=policy_ev.auth_liveness_state,
                fasnet=result.fasnet,
                identity=result.identity,
                face_count=result.detection.face_count,
                base_outcome=result.outcome,
            )

        draw_challenge1_hud(frame, session.hud(), debug=debug)
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
        f"[CHALLENGE1] done gesture_ok={result.succeeded_gesture} "
        f"fail={result.fail_reason.value} "
        f"decision={result.outcome.decision.value if result.outcome else None} "
        f"reason={result.outcome.reason if result.outcome else None}"
    )
    return result


def outcome_from_challenge(result: Challenge1Result) -> AuthOutcome:
    if result.outcome is not None:
        return result.outcome
    return AuthOutcome(
        AuthDecision.DENY,
        f"challenge1_failed:{result.fail_reason.value}",
        0,
        None,
        False,
        None,
        challenge_1_required=False,
        challenge_1_status=f"FAILED_{result.fail_reason.value}",
    )
