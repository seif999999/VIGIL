"""VIGIL Stage 3: TCP verifier — receives HMAC packages and displays verified frames."""

from __future__ import annotations

import argparse
import socket
import struct
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from protocol import (
    VerifyReason,
    FrameVerifier,
    generate_session_id,
    load_key,
    HEADER_SIZE,
    HMAC_SIZE,
    MAX_JPEG_SIZE,
    MIN_PACKAGE_SIZE,
    SESSION_ID_SIZE,
)

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
FACE_DETECTION_DIR = PROJECT_ROOT / "face_detection"
LIVENESS_DIR = PROJECT_ROOT / "liveness"
KEY_PATH = BASE_DIR / "secrets" / "hmac.key"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5000
WINDOW_TITLE = "VIGIL - Integrated Visual Authenticity (TCP)"
LENGTH_PREFIX_SIZE = 4
MAX_HMAC_PACKAGE_SIZE = HEADER_SIZE + MAX_JPEG_SIZE + HMAC_SIZE


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


def _decode_verified_jpeg(jpeg_bytes: bytes):
    buffer = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)


def _draw_overlay(
    frame_bgr,
    session_id: bytes,
    frame_counter: int,
    fps: float,
    *,
    y_start: int = 28,
) -> None:
    lines = [
        f"Session: {_session_hex_prefix(session_id)}",
        f"Frame: {frame_counter}",
        "HMAC: VERIFIED",
        "Transport: TCP",
        f"FPS: {fps:.1f}",
    ]
    y = y_start
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


class VerifierStats:
    def __init__(self) -> None:
        self.packages_received = 0
        self.verified = 0
        self.rejected = 0
        self._fps_window_start = time.perf_counter()
        self._fps_window_frames = 0
        self.display_fps = 0.0

    def mark_verified(self) -> None:
        self.verified += 1
        self._fps_window_frames += 1
        elapsed = time.perf_counter() - self._fps_window_start
        if elapsed >= 1.0:
            self.display_fps = self._fps_window_frames / elapsed
            self._fps_window_start = time.perf_counter()
            self._fps_window_frames = 0

    def average_fps(self, elapsed_seconds: float) -> float:
        if elapsed_seconds <= 0:
            return 0.0
        return self.verified / elapsed_seconds


def run(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    enable_face_detection: bool = False,
    enable_liveness: bool = False,
    enable_friend_auth: bool = False,
    provisional_real_min: float = 0.85,
    provisional_spoof_min: float = 0.85,
    enrollment_db: Path | None = None,
) -> int:
    secret_key = _load_hmac_key()
    session_id = generate_session_id()
    verifier = FrameVerifier(secret_key, session_id)
    session_prefix = _session_hex_prefix(session_id)
    stats = VerifierStats()
    started_at = time.perf_counter()
    if (enable_liveness or enable_friend_auth) and not enable_face_detection:
        enable_face_detection = True
        print("[INFO] Perception flags require face detection; SCRFD enabled.")
    if enable_friend_auth and enable_liveness:
        print(
            "[WARN] Both --enable-friend-auth and --enable-liveness set; "
            "friend-auth (FasNet+FaceNet) takes precedence for display/decision. "
            "Legacy mn3 remains available via --enable-liveness alone.",
            file=sys.stderr,
        )
        enable_liveness = False

    detector_calls = 0
    liveness_calls = 0
    friend_auth_calls = 0
    face_detector = None
    draw_detections = None
    liveness_detector = None
    run_liveness_if_face_ok = None
    draw_demo_liveness_readout = None
    resolve_demo_outcome = None
    provisional_policy = None
    ProvisionalDemoOutcome = None
    friend_pipeline = None
    auth_decision_counts: dict[str, int] = {}
    if enable_face_detection or enable_liveness or enable_friend_auth:
        if str(LIVENESS_DIR) not in sys.path:
            sys.path.insert(0, str(LIVENESS_DIR))
    if enable_face_detection:
        from face_bridge import load_face_detector_module  # noqa: WPS433

        face_mod = load_face_detector_module()
        draw_detections = face_mod.draw_detections
        face_detector = face_mod.FaceDetector()
        print("[INFO] SCRFD face detection enabled (post-HMAC only).")
    if enable_friend_auth:
        if face_detector is None:
            print(
                "[ERROR] --enable-friend-auth requires face detection (SCRFD).",
                file=sys.stderr,
            )
            return 1
        from auth_pipeline import FriendAuthPipeline  # noqa: WPS433
        from facenet_identity import (  # noqa: WPS433
            DEFAULT_ENROLLMENT_PATH,
            MATCH_THRESHOLD,
            MATCH_THRESHOLD_STATUS,
        )

        enroll_path = enrollment_db or DEFAULT_ENROLLMENT_PATH
        friend_pipeline = FriendAuthPipeline(
            face_detector, enrollment_path=enroll_path
        )
        init_ms = friend_pipeline.warm_up()
        n_ids = len(friend_pipeline.enrollments.get("identities", {}))
        print(
            "[INFO] Friend-auth enabled (post-HMAC): SCRFD → FasNet → FaceNet. "
            f"threshold={MATCH_THRESHOLD} [{MATCH_THRESHOLD_STATUS}] "
            f"enrollments={n_ids} @ {enroll_path}"
        )
        print(
            f"[INFO] Model warm-up fasnet_ms={init_ms.get('fasnet_init_ms')} "
            f"facenet_ms={init_ms.get('facenet_init_ms')}"
        )
        if n_ids == 0:
            print(
                "[WARN] No enrollments — ACCEPT impossible until enroll_identity.py runs.",
                file=sys.stderr,
            )
    if enable_liveness:
        if face_detector is None:
            print(
                "[ERROR] --enable-liveness requires face detection (SCRFD).",
                file=sys.stderr,
            )
            return 1
        from decision_policy import (  # noqa: WPS433
            ProvisionalDecisionPolicy,
            ProvisionalDemoOutcome as _PDO,
            resolve_demo_outcome as _resolve_demo,
        )
        from detector import LivenessDetector  # noqa: WPS433
        from overlay import draw_demo_liveness_readout as _draw_demo  # noqa: WPS433
        from pipeline import run_liveness_if_face_ok as _run_live  # noqa: WPS433

        ProvisionalDemoOutcome = _PDO
        resolve_demo_outcome = _resolve_demo
        provisional_policy = ProvisionalDecisionPolicy(
            confident_real_min=provisional_real_min,
            confident_spoof_min=provisional_spoof_min,
        )
        draw_demo_liveness_readout = _draw_demo
        run_liveness_if_face_ok = _run_live
        liveness_detector = LivenessDetector()
        print("[INFO] anti-spoof-mn3 liveness enabled (FACE_OK gate only).")
        print(
            "[INFO] Provisional demo policy (uncalibrated): "
            f"REAL>={provisional_real_min}, SPOOF>={provisional_spoof_min}; "
            "else UNCERTAIN_REVIEW — not identity authorization."
        )

    verify_latencies_ms: list[float] = []
    scrfd_latencies_ms: list[float] = []
    liveness_latencies_ms: list[float] = []
    fasnet_latencies_ms: list[float] = []
    facenet_latencies_ms: list[float] = []
    friend_total_ms: list[float] = []
    combined_perception_ms: list[float] = []
    demo_outcome_counts: dict[str, int] = {}
    if ProvisionalDemoOutcome is not None:
        demo_outcome_counts = {o.value: 0 for o in ProvisionalDemoOutcome}

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        server.bind((host, port))
    except OSError as exc:
        print(f"[ERROR] Could not bind {host}:{port} ({exc})", file=sys.stderr)
        server.close()
        return 1

    server.listen(1)
    print(f"[INFO] Listening on {host}:{port}")
    print(f"[INFO] Session: {session_prefix}...")
    print("[INFO] Waiting for trusted capture...")

    conn: socket.socket | None = None
    try:
        conn, addr = server.accept()
        print(f"[INFO] Connected from {addr[0]}:{addr[1]}")
        conn.sendall(session_id)

        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
        running = True

        while running:
            cv2.pollKey()
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break

            length_prefix = recv_exact(conn, LENGTH_PREFIX_SIZE)
            if length_prefix is None:
                print("[INFO] Capture client disconnected.")
                break

            (pkg_len,) = struct.unpack(">I", length_prefix)
            if pkg_len < MIN_PACKAGE_SIZE or pkg_len > MAX_HMAC_PACKAGE_SIZE:
                print(
                    f"[ERROR] Invalid network package length: {pkg_len}",
                    file=sys.stderr,
                )
                break

            package = recv_exact(conn, pkg_len)
            if package is None or len(package) != pkg_len:
                print("[ERROR] Truncated network package.", file=sys.stderr)
                break

            stats.packages_received += 1
            t_verify0 = time.perf_counter_ns()
            result = verifier.verify(package)
            verify_ms = (time.perf_counter_ns() - t_verify0) / 1_000_000

            if result.reason != VerifyReason.ACCEPT:
                stats.rejected += 1
                frame_no = result.frame_counter
                suffix = f" frame={frame_no}" if frame_no is not None else ""
                print(
                    f"[REJECT]{suffix} reason={result.reason.value}",
                    file=sys.stderr,
                )
                print(
                    "[SECURITY] Authentication stream rejected; terminating current session.",
                    file=sys.stderr,
                )
                running = False
                break

            accepted_counter = result.frame_counter
            assert accepted_counter is not None
            verified_jpeg = package[HEADER_SIZE : len(package) - HMAC_SIZE]
            verified_bgr = _decode_verified_jpeg(verified_jpeg)
            if verified_bgr is None:
                stats.rejected += 1
                print(
                    f"[REJECT] frame={accepted_counter} reason=JPEG_DECODE_FAILED",
                    file=sys.stderr,
                )
                continue

            stats.mark_verified()
            verify_latencies_ms.append(verify_ms)
            display_bgr = verified_bgr
            perception_ms = 0.0
            if face_detector is not None and draw_detections is not None:
                t_perc0 = time.perf_counter_ns()
                if friend_pipeline is not None:
                    friend_auth_calls += 1
                    detector_calls += 1
                    auth = friend_pipeline.process_verified_frame(verified_bgr)
                    scrfd_latencies_ms.append(auth.scrfd_ms)
                    if auth.fasnet is not None:
                        fasnet_latencies_ms.append(auth.fasnet_ms)
                    if auth.identity is not None and auth.identity.error is None:
                        facenet_latencies_ms.append(auth.facenet_ms)
                    friend_total_ms.append(auth.total_ms)
                    perception_ms = auth.total_ms
                    combined_perception_ms.append(perception_ms)
                    key = f"{auth.outcome.decision.value}:{auth.outcome.reason}"
                    auth_decision_counts[key] = auth_decision_counts.get(key, 0) + 1
                    display_bgr = draw_detections(verified_bgr, auth.detection)
                    color = (
                        (0, 255, 0)
                        if auth.outcome.decision.value == "ACCEPT"
                        else (0, 0, 255)
                    )
                    dist = (
                        auth.identity.best_distance
                        if auth.identity is not None
                        else None
                    )
                    lines = [
                        f"AUTH: {auth.outcome.decision.value} ({auth.outcome.reason})",
                        f"FasNet: {auth.fasnet.predicted_class.value if auth.fasnet else 'n/a'} "
                        f"score={auth.fasnet.antispoof_score if auth.fasnet else None}",
                        f"ID: {auth.outcome.identity} dist={dist}",
                        f"ms SCRFD={auth.scrfd_ms:.0f} Fas={auth.fasnet_ms:.0f} "
                        f"FN={auth.facenet_ms:.0f} tot={auth.total_ms:.0f}",
                    ]
                    y = display_bgr.shape[0] - len(lines) * 22 - 8
                    y = max(160, y)
                    for line in lines:
                        cv2.putText(
                            display_bgr,
                            line,
                            (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color,
                            1,
                            cv2.LINE_AA,
                        )
                        y += 22
                else:
                    detector_calls += 1
                    det_result = face_detector.detect(verified_bgr)
                    scrfd_latencies_ms.append(det_result.inference_ms)
                    display_bgr = draw_detections(verified_bgr, det_result)
                    live_result = None
                    if (
                        liveness_detector is not None
                        and run_liveness_if_face_ok is not None
                    ):
                        live_result = run_liveness_if_face_ok(
                            verified_bgr, det_result, liveness_detector
                        )
                        if live_result is not None:
                            liveness_calls += 1
                            liveness_latencies_ms.append(live_result.inference_ms)
                    perception_ms = (time.perf_counter_ns() - t_perc0) / 1_000_000
                    combined_perception_ms.append(perception_ms)
                    if (
                        draw_demo_liveness_readout is not None
                        and provisional_policy is not None
                        and resolve_demo_outcome is not None
                    ):
                        demo = resolve_demo_outcome(
                            det_result.decision.value,
                            live_result,
                            provisional_policy,
                        )
                        demo_outcome_counts[demo.value] = (
                            demo_outcome_counts.get(demo.value, 0) + 1
                        )
                        display_bgr = draw_demo_liveness_readout(
                            display_bgr,
                            face_decision_value=det_result.decision.value,
                            liveness=live_result,
                            policy=provisional_policy,
                            total_ms=perception_ms,
                        )
            hmac_y = 110 if face_detector is not None else 28
            _draw_overlay(
                display_bgr,
                session_id,
                accepted_counter,
                stats.display_fps,
                y_start=hmac_y,
            )
            if face_detector is not None:
                cv2.putText(
                    display_bgr,
                    f"HMAC verify: {verify_ms:.1f} ms",
                    (12, hmac_y + 5 * 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
                if perception_ms > 0:
                    cv2.putText(
                        display_bgr,
                        f"SCRFD+liveness: {perception_ms:.1f} ms",
                        (12, hmac_y + 6 * 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )
            cv2.imshow(WINDOW_TITLE, display_bgr)

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        if conn is not None:
            try:
                conn.close()
            except OSError:
                pass
        server.close()
        cv2.destroyAllWindows()

        elapsed = time.perf_counter() - started_at
        if stats.rejected == 0:
            result_label = "PASS"
        else:
            result_label = "SECURITY_REJECT"
        print("\n--- VIGIL Stage 3 Verifier Summary ---")
        print(f"Session: {session_prefix}...")
        print(f"Packages received: {stats.packages_received}")
        print(f"Frames verified: {stats.verified}")
        print(f"Frames rejected: {stats.rejected}")
        print(f"Approx. FPS: {stats.average_fps(elapsed):.1f}")
        if enable_face_detection:
            print(f"Detector calls: {detector_calls}")
        if enable_liveness:
            print(f"Liveness calls (FACE_OK only): {liveness_calls}")
        if enable_friend_auth:
            print(f"Friend-auth calls (post-HMAC): {friend_auth_calls}")
        if verify_latencies_ms:
            print(
                f"Median HMAC verify latency: "
                f"{statistics.median(verify_latencies_ms):.3f} ms"
            )
        if scrfd_latencies_ms:
            print(
                f"Median SCRFD latency: "
                f"{statistics.median(scrfd_latencies_ms):.3f} ms"
            )
        if liveness_latencies_ms:
            print(
                f"Median liveness latency: "
                f"{statistics.median(liveness_latencies_ms):.3f} ms"
            )
        if fasnet_latencies_ms:
            print(
                f"Median FasNet latency: "
                f"{statistics.median(fasnet_latencies_ms):.3f} ms"
            )
        if facenet_latencies_ms:
            print(
                f"Median FaceNet latency: "
                f"{statistics.median(facenet_latencies_ms):.3f} ms"
            )
        if friend_total_ms:
            print(
                f"Median friend-auth total: "
                f"{statistics.median(friend_total_ms):.3f} ms"
            )
        if combined_perception_ms:
            print(
                f"Median SCRFD+liveness wall time: "
                f"{statistics.median(combined_perception_ms):.3f} ms"
            )
        if demo_outcome_counts:
            print("Provisional demo outcome counts (uncalibrated policy):")
            for key in sorted(demo_outcome_counts):
                print(f"  {key}: {demo_outcome_counts[key]}")
        if auth_decision_counts:
            print("Friend-auth decision counts:")
            for key in sorted(auth_decision_counts):
                print(f"  {key}: {auth_decision_counts[key]}")
        print(f"Result: {result_label}")

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="VIGIL Stage 3 HMAC verifier (TCP)")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Bind address")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="TCP port")
    parser.add_argument(
        "--enable-face-detection",
        action="store_true",
        help="Run SCRFD on HMAC-verified frames only",
    )
    parser.add_argument(
        "--enable-liveness",
        action="store_true",
        help="Run anti-spoof-mn3 after SCRFD FACE_OK (implies face detection)",
    )
    parser.add_argument(
        "--enable-visual-baseline",
        action="store_true",
        help="Full pipeline: SCRFD + mn3 liveness + provisional demo policy (post-HMAC)",
    )
    parser.add_argument(
        "--enable-friend-auth",
        action="store_true",
        help=(
            "Experimental: SCRFD + DeepFace FasNet + FaceNet identity "
            "(post-HMAC only; ACCEPT = REAL and recognized)"
        ),
    )
    parser.add_argument(
        "--enrollment-db",
        type=Path,
        default=None,
        help="FaceNet enrollment JSON (default: liveness/data/enrollments.json)",
    )
    parser.add_argument(
        "--provisional-real-min",
        type=float,
        default=0.85,
        help="Provisional min real score for confident REAL (uncalibrated, mn3 only)",
    )
    parser.add_argument(
        "--provisional-spoof-min",
        type=float,
        default=0.85,
        help="Provisional min spoof score for confident SPOOF (uncalibrated, mn3 only)",
    )
    args = parser.parse_args()
    enable_face = (
        args.enable_face_detection
        or args.enable_visual_baseline
        or args.enable_friend_auth
    )
    enable_live = args.enable_liveness or args.enable_visual_baseline
    sys.exit(
        run(
            host=args.host,
            port=args.port,
            enable_face_detection=enable_face,
            enable_liveness=enable_live,
            enable_friend_auth=args.enable_friend_auth,
            provisional_real_min=args.provisional_real_min,
            provisional_spoof_min=args.provisional_spoof_min,
            enrollment_db=args.enrollment_db,
        )
    )


if __name__ == "__main__":
    main()
