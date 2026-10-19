"""VIGIL Stage 3: trusted webcam capture — signs frames and sends them over TCP."""

from __future__ import annotations

import argparse
import socket
import struct
import sys
import time
from pathlib import Path

import cv2

from protocol import (
    create_frame_package,
    load_key,
    SESSION_ID_SIZE,
)

BASE_DIR = Path(__file__).resolve().parent
LIVENESS_DIR = BASE_DIR.parent / "liveness"
KEY_PATH = BASE_DIR / "secrets" / "hmac.key"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5000
LENGTH_PREFIX_SIZE = 4


def recv_exact(sock: socket.socket, nbytes: int) -> bytes | None:
    """Receive exactly nbytes or None if the connection closes early."""
    chunks: list[bytes] = []
    remaining = nbytes
    while remaining > 0:
        try:
            chunk = sock.recv(remaining)
        except (ConnectionResetError, ConnectionAbortedError, OSError):
            return None
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _session_hex_prefix(session_id: bytes, n: int = 8) -> str:
    return session_id[: n // 2].hex()


def _load_hmac_key() -> bytes:
    try:
        return load_key(KEY_PATH)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        print("[ERROR] Generate a key first: python keygen.py", file=sys.stderr)
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
        return None
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap


def _encode_jpeg(frame) -> bytes | None:
    ok, encoded = cv2.imencode(".jpg", frame)
    if not ok or encoded is None:
        return None
    return encoded.tobytes()


def _send_framed_package(sock: socket.socket, package: bytes) -> None:
    record = struct.pack(">I", len(package)) + package
    sock.sendall(record)


def connect_verifier_session(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
) -> tuple[socket.socket, bytes, bytes]:
    """Connect to verifier; return (socket, session_id, secret_key)."""
    secret_key = _load_hmac_key()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((host, port))
    except OSError as exc:
        print(
            f"[ERROR] Could not connect to verifier at {host}:{port} ({exc})",
            file=sys.stderr,
        )
        print("[ERROR] Start verifier_server.py first.", file=sys.stderr)
        sock.close()
        raise SystemExit(1) from exc

    session_bytes = recv_exact(sock, SESSION_ID_SIZE)
    if session_bytes is None or len(session_bytes) != SESSION_ID_SIZE:
        print("[ERROR] Failed to receive session ID from verifier.", file=sys.stderr)
        sock.close()
        raise SystemExit(1)
    return sock, session_bytes, secret_key


def stream_signed_bgr_frames(
    sock: socket.socket,
    session_id: bytes,
    secret_key: bytes,
    read_bgr_frame,
    *,
    duration_seconds: float | None = None,
    source_label: str = "CAPTURE",
) -> int:
    """Sign and send BGR frames using existing HMAC frame packages."""
    session_prefix = _session_hex_prefix(session_id)
    print(f"[INFO] Using verifier session: {session_prefix}...")
    print(f"[INFO] Source: {source_label}")

    frame_counter = 0
    captured = 0
    encoded = 0
    packages_sent = 0
    started_at = time.perf_counter()

    try:
        while True:
            if duration_seconds is not None:
                if time.perf_counter() - started_at >= duration_seconds:
                    print(
                        f"[INFO] Duration limit ({duration_seconds:.0f}s) reached."
                    )
                    break

            cv2.pollKey()
            frame = read_bgr_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            captured += 1
            jpeg_bytes = _encode_jpeg(frame)
            if jpeg_bytes is None:
                print(
                    f"[WARN] JPEG encode failed for capture #{captured}",
                    file=sys.stderr,
                )
                continue

            encoded += 1
            package = create_frame_package(
                jpeg_bytes,
                session_id,
                frame_counter,
                secret_key,
            )

            try:
                _send_framed_package(sock, package)
            except OSError as exc:
                print(f"[ERROR] Send failed: {exc}", file=sys.stderr)
                break

            packages_sent += 1
            frame_counter += 1

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        sock.close()

        if packages_sent > 0 and packages_sent == encoded:
            result_label = "PASS"
        else:
            result_label = "FAIL"

        print("\n--- VIGIL Stage 3 Capture Summary ---")
        print(f"Source: {source_label}")
        print(f"Session: {session_prefix}...")
        print(f"Frames captured: {captured}")
        print(f"Frames encoded: {encoded}")
        print(f"Packages sent: {packages_sent}")
        print(f"Result: {result_label}")

    return 0


def run(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    camera_index: int = 0,
    duration_seconds: float | None = None,
    windows_device: str | None = None,
) -> int:
    cap = None
    winrt_cap = None
    if windows_device:
        if str(LIVENESS_DIR) not in sys.path:
            sys.path.insert(0, str(LIVENESS_DIR))
        from windows_camera_capture import (  # noqa: WPS433
            open_winrt_capture,
            print_winrt_capture_info,
        )

        try:
            winrt_cap = open_winrt_capture(windows_device)
        except RuntimeError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return 1
        print_winrt_capture_info(winrt_cap.capture_info)
        print(f"[INFO] Windows device: {windows_device}")
    else:
        cap = _open_camera(camera_index)
        if cap is None:
            return 1
        print(f"[INFO] Camera index: {camera_index}")

    try:
        sock, session_id, secret_key = connect_verifier_session(host, port)

        def read_frame():
            if winrt_cap is not None:
                ok, frame = winrt_cap.read()
            else:
                ok, frame = cap.read()
            if not ok or frame is None:
                print("[WARN] Frame capture failed; retrying...", file=sys.stderr)
                time.sleep(0.05)
                return None
            return frame

        source = "WINDOWS" if winrt_cap else "OPENCV"
        return stream_signed_bgr_frames(
            sock,
            session_id,
            secret_key,
            read_frame,
            duration_seconds=duration_seconds,
            source_label=source,
        )
    finally:
        if winrt_cap is not None:
            winrt_cap.release()
        elif cap is not None:
            cap.release()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VIGIL Stage 3 trusted capture (TCP client)",
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="Verifier address")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Verifier port")
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
        help="Optional auto-stop after N seconds",
    )
    parser.add_argument(
        "--windows-device",
        type=str,
        default=None,
        help='WinRT MediaCapture device, e.g. "Nazmy (Windows Virtual Camera)"',
    )
    args = parser.parse_args()
    if args.windows_device and args.camera != 0:
        print(
            "[WARN] --windows-device set; ignoring --camera index.",
            file=sys.stderr,
        )
    sys.exit(
        run(
            host=args.host,
            port=args.port,
            camera_index=args.camera,
            duration_seconds=args.duration,
            windows_device=args.windows_device,
        )
    )


if __name__ == "__main__":
    main()
