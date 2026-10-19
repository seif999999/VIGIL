"""VIGIL Stage 2: live webcam capture with HMAC gate before display."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

from protocol import (
    VerifyReason,
    create_frame_package,
    generate_session_id,
    load_key,
    FrameVerifier,
    HEADER_SIZE,
    HMAC_SIZE,
)

BASE_DIR = Path(__file__).resolve().parent
KEY_PATH = BASE_DIR / "secrets" / "hmac.key"
WINDOW_TITLE = "VIGIL - HMAC Verified Camera"


class LiveStats:
    def __init__(self) -> None:
        self.captured = 0
        self.verified = 0
        self.rejected = 0
        self._fps_window_start = time.perf_counter()
        self._fps_window_frames = 0
        self.display_fps = 0.0

    def mark_capture(self) -> None:
        self.captured += 1

    def mark_verified(self) -> None:
        self.verified += 1
        self._fps_window_frames += 1
        elapsed = time.perf_counter() - self._fps_window_start
        if elapsed >= 1.0:
            self.display_fps = self._fps_window_frames / elapsed
            self._fps_window_start = time.perf_counter()
            self._fps_window_frames = 0

    def mark_rejected(self) -> None:
        self.rejected += 1

    def average_fps(self, elapsed_seconds: float) -> float:
        if elapsed_seconds <= 0:
            return 0.0
        return self.verified / elapsed_seconds


def _session_hex_prefix(session_id: bytes, n: int = 8) -> str:
    return session_id[: n // 2].hex()


def _draw_overlay(
    frame_bgr,
    session_id: bytes,
    frame_counter: int,
    fps: float,
) -> None:
    lines = [
        f"Session: {_session_hex_prefix(session_id)}",
        f"Frame: {frame_counter}",
        "HMAC: VERIFIED",
        f"FPS: {fps:.1f}",
    ]
    y = 28
    for line in lines:
        cv2.putText(
            frame_bgr,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        y += 28


def _load_hmac_key() -> bytes:
    try:
        return load_key(KEY_PATH)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        print(
            "[ERROR] Generate a key first: python keygen.py",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc


def _open_camera(index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera index {index}.", file=sys.stderr)
        raise SystemExit(1)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap


def _encode_jpeg(frame) -> bytes | None:
    ok, encoded = cv2.imencode(".jpg", frame)
    if not ok or encoded is None:
        return None
    return encoded.tobytes()


def _decode_verified_jpeg(jpeg_bytes: bytes):
    import numpy as np

    buffer = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    decoded = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    return decoded


def run(camera_index: int = 0, duration_seconds: float | None = None) -> int:
    secret_key = _load_hmac_key()
    session_id = generate_session_id()
    verifier = FrameVerifier(secret_key, session_id)
    sender_counter = 0
    stats = LiveStats()

    cap = _open_camera(camera_index)
    session_prefix = _session_hex_prefix(session_id)
    print(f"[INFO] Session started: {session_prefix}...")
    print(f"[INFO] Camera index: {camera_index}")
    print("[INFO] Press Q or ESC to stop.")

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)

    exit_code = 0
    started_at = time.perf_counter()
    try:
        while True:
            if duration_seconds is not None:
                if time.perf_counter() - started_at >= duration_seconds:
                    print(f"[INFO] Duration limit ({duration_seconds:.0f}s) reached.")
                    break

            # Pump HighGUI events before capture (Windows DShow/MSMF + imshow coupling).
            cv2.pollKey()

            ok, frame = cap.read()
            if not ok or frame is None:
                print("[WARN] Frame capture failed; retrying...", file=sys.stderr)
                time.sleep(0.05)
                continue

            stats.mark_capture()

            jpeg_bytes = _encode_jpeg(frame)
            if jpeg_bytes is None:
                print(
                    f"[WARN] JPEG encode failed for capture #{stats.captured}",
                    file=sys.stderr,
                )
                continue

            package = create_frame_package(
                jpeg_bytes,
                session_id,
                sender_counter,
                secret_key,
            )
            result = verifier.verify(package)

            if result.reason != VerifyReason.ACCEPT:
                stats.mark_rejected()
                print(
                    f"[REJECT] frame={sender_counter} reason={result.reason.value}",
                    file=sys.stderr,
                )
                continue

            accepted_counter = result.frame_counter
            assert accepted_counter is not None
            sender_counter = accepted_counter + 1

            verified_jpeg = package[HEADER_SIZE : len(package) - HMAC_SIZE]
            verified_bgr = _decode_verified_jpeg(verified_jpeg)
            if verified_bgr is None:
                stats.mark_rejected()
                print(
                    f"[REJECT] frame={sender_counter} reason=JPEG_DECODE_FAILED",
                    file=sys.stderr,
                )
                continue

            stats.mark_verified()
            _draw_overlay(
                verified_bgr,
                session_id,
                accepted_counter,
                stats.display_fps,
            )
            cv2.imshow(WINDOW_TITLE, verified_bgr)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

        result_label = "PASS" if stats.rejected == 0 else "FAIL"
        print("\n--- VIGIL Stage 2 Summary ---")
        print(f"Session: {session_prefix}...")
        print(f"Frames captured: {stats.captured}")
        print(f"Frames verified: {stats.verified}")
        print(f"Frames rejected: {stats.rejected}")
        elapsed_total = time.perf_counter() - started_at
        print(f"Approx. FPS: {stats.average_fps(elapsed_total):.1f}")
        print(f"Result: {result_label}")

    return exit_code


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VIGIL live HMAC-verified webcam (Stage 2)",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam device index (default: 0)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Optional auto-stop after N seconds (default: run until Q/ESC)",
    )
    args = parser.parse_args()
    sys.exit(run(camera_index=args.camera, duration_seconds=args.duration))


if __name__ == "__main__":
    main()
