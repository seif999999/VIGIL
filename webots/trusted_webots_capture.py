"""Bridge: Webots camera stream → existing HMAC trusted capture (verifier TCP)."""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
HMAC_DIR = BASE_DIR.parent / "hmac"
sys.path.insert(0, str(HMAC_DIR))
sys.path.insert(0, str(BASE_DIR))

from frame_diagnostic import analyze_bgr_frame, print_diagnostic_report, save_diagnostic_png  # noqa: E402
from frame_protocol import recv_bgr_frame  # noqa: E402
from trusted_capture import (  # noqa: E402
    DEFAULT_HOST,
    DEFAULT_PORT,
    connect_verifier_session,
    stream_signed_bgr_frames,
)

BRIDGE_DEFAULT_PORT = 5010
LOG_DIR = BASE_DIR / "run_logs"


def _print_scrfd_on_diagnostic(frame_bgr) -> None:
    face_dir = BASE_DIR.parent / "face_detection"
    try:
        sys.path.insert(0, str(face_dir))
        from detector import FaceDetector  # noqa: WPS433

        det = FaceDetector().detect(frame_bgr)
        print(
            f"SCRFD on diagnostic frame: faces={det.face_count} "
            f"decision={det.decision.value}"
        )
    except Exception as exc:
        print(f"SCRFD on diagnostic frame: skipped ({exc})")


def _wait_for_webots(
    listen_port: int,
    timeout_s: float,
) -> tuple[socket.socket, socket.socket]:
    """Return (bridge_listener_closed, webots_conn)."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", listen_port))
    server.listen(1)
    server.settimeout(timeout_s)
    print(
        f"[INFO] Webots bridge listening on 127.0.0.1:{listen_port} "
        f"(timeout {timeout_s:.0f}s — start Webots world now)"
    )
    try:
        conn, addr = server.accept()
    except socket.timeout as exc:
        server.close()
        raise TimeoutError(
            "No Webots controller connected. "
            "Open webots/worlds/vigil_indoor.wbt and run simulation."
        ) from exc
    print(f"[INFO] Webots controller connected from {addr[0]}:{addr[1]}")
    server.close()
    return conn


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Webots → HMAC signed stream to VIGIL verifier",
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="Verifier address")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Verifier port")
    parser.add_argument(
        "--bridge-port",
        type=int,
        default=BRIDGE_DEFAULT_PORT,
        help="Local port for Webots controller BGR frames",
    )
    parser.add_argument(
        "--bridge-timeout",
        type=float,
        default=120.0,
        help="Seconds to wait for Webots controller connection",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Stop after N seconds of streaming",
    )
    parser.add_argument(
        "--save-diagnostic",
        type=Path,
        default=None,
        help="Save first received Webots BGR frame as PNG and print pixel stats",
    )
    args = parser.parse_args()

    webots_conn = _wait_for_webots(args.bridge_port, args.bridge_timeout)
    sock, session_id, secret_key = connect_verifier_session(args.host, args.port)

    frames_from_webots = 0
    diagnostic_saved = False
    started = time.perf_counter()

    def read_frame():
        nonlocal frames_from_webots, diagnostic_saved
        frame = recv_bgr_frame(webots_conn)
        if frame is not None:
            frames_from_webots += 1
            if args.save_diagnostic is not None and not diagnostic_saved:
                diagnostic_saved = True
                out = args.save_diagnostic
                if not out.is_absolute():
                    out = BASE_DIR / out
                save_diagnostic_png(frame, out)
                stats = analyze_bgr_frame(frame)
                print_diagnostic_report(stats, out)
                _print_scrfd_on_diagnostic(frame)
        return frame

    try:
        rc = stream_signed_bgr_frames(
            sock,
            session_id,
            secret_key,
            read_frame,
            duration_seconds=args.duration,
            source_label="WEBOTS",
        )
    finally:
        webots_conn.close()

    elapsed = time.perf_counter() - started
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stats_path = LOG_DIR / f"webots_capture_{ts}.json"
    stats = {
        "timestamp_utc": ts,
        "source": "WEBOTS",
        "note": (
            "Simulation target is a rendered poster texture — "
            "not a live human; liveness scores are not physical-ground-truth."
        ),
        "webots_frames_received": frames_from_webots,
        "elapsed_s": elapsed,
        "verifier_host": args.host,
        "verifier_port": args.port,
        "bridge_port": args.bridge_port,
    }
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"[INFO] Bridge stats: {stats_path}")
    sys.exit(rc)


if __name__ == "__main__":
    main()
