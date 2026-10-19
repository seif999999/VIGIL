"""VIGIL Stage 5: HMAC security and localhost TCP overhead benchmarks."""

from __future__ import annotations

import argparse
import json
import platform
import socket
import statistics
import struct
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from protocol import (
    VerifyReason,
    create_frame_package,
    generate_session_id,
    load_key,
    FrameVerifier,
    HEADER_SIZE,
    HMAC_SIZE,
    MIN_PACKAGE_SIZE,
    MAX_JPEG_SIZE,
)

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore

BASE_DIR = Path(__file__).resolve().parent
KEY_PATH = BASE_DIR / "secrets" / "hmac.key"
DEFAULT_IMAGE = BASE_DIR / "test_images" / "original_1.jpg"
RESULTS_DIR = BASE_DIR / "benchmark_results"
LENGTH_PREFIX_SIZE = 4
MAX_HMAC_PACKAGE_SIZE = HEADER_SIZE + MAX_JPEG_SIZE + HMAC_SIZE
WARMUP_CRYPTO = 100
WARMUP_TCP = 1


def recv_exact(sock: socket.socket, nbytes: int) -> bytes | None:
    buf = bytearray()
    while len(buf) < nbytes:
        chunk = sock.recv(nbytes - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def send_framed(sock: socket.socket, payload: bytes) -> None:
    sock.sendall(struct.pack(">I", len(payload)) + payload)


@dataclass
class TimingStats:
    count: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float


def stats_from_ns(samples_ns: list[int]) -> TimingStats:
    if not samples_ns:
        return TimingStats(0, 0.0, 0.0, 0.0, 0.0, 0.0)
    ms = [x / 1_000_000 for x in samples_ns]
    ms_sorted = sorted(ms)
    p95_idx = max(0, int(len(ms_sorted) * 0.95) - 1)
    return TimingStats(
        count=len(ms),
        mean_ms=statistics.mean(ms),
        median_ms=statistics.median(ms),
        p95_ms=ms_sorted[p95_idx],
        min_ms=min(ms),
        max_ms=max(ms),
    )


def capture_camera_jpegs(camera_index: int, frame_count: int) -> list[bytes]:
    if cv2 is None:
        print("[ERROR] OpenCV required for camera capture.", file=sys.stderr)
        raise SystemExit(1)
    cap = cv2.VideoCapture(camera_index, cv2.CAP_MSMF)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera index {camera_index}.", file=sys.stderr)
        raise SystemExit(1)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 30)

    jpegs: list[bytes] = []
    try:
        while len(jpegs) < frame_count:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            ok_enc, encoded = cv2.imencode(".jpg", frame)
            if ok_enc and encoded is not None:
                jpegs.append(encoded.tobytes())
    finally:
        cap.release()
    if not jpegs:
        print("[ERROR] No JPEG frames captured.", file=sys.stderr)
        raise SystemExit(1)
    return jpegs


def run_crypto_microbenchmark(
    jpeg_bytes: bytes,
    secret_key: bytes,
    session_id: bytes,
    iterations: int,
) -> dict:
    warmup_verifier = FrameVerifier(secret_key, session_id)
    for i in range(WARMUP_CRYPTO):
        pkg = create_frame_package(jpeg_bytes, session_id, i, secret_key)
        assert warmup_verifier.verify(pkg).reason == VerifyReason.ACCEPT

    sign_ns: list[int] = []
    verify_ns: list[int] = []
    combined_ns: list[int] = []
    verifier = FrameVerifier(secret_key, session_id)

    for i in range(iterations):
        t0 = time.perf_counter_ns()
        package = create_frame_package(jpeg_bytes, session_id, i, secret_key)
        t1 = time.perf_counter_ns()
        sign_ns.append(t1 - t0)

        t2 = time.perf_counter_ns()
        result = verifier.verify(package)
        t3 = time.perf_counter_ns()
        if result.reason != VerifyReason.ACCEPT:
            raise RuntimeError(f"Benchmark verify failed at counter {i}: {result.reason}")
        verify_ns.append(t3 - t2)
        combined_ns.append(t3 - t0)

    sign_stats = stats_from_ns(sign_ns)
    verify_stats = stats_from_ns(verify_ns)
    combined_stats = stats_from_ns(combined_ns)
    combined_median_ms = combined_stats.median_ms
    throughput = 1000.0 / combined_median_ms if combined_median_ms > 0 else 0.0

    return {
        "sign": asdict(sign_stats),
        "verify": asdict(verify_stats),
        "combined_per_operation": asdict(combined_stats),
        "combined_median_sign_plus_verify_ms": combined_median_ms,
        "security_processing_throughput_fps": throughput,
    }


def compute_package_overhead(jpeg_size: int) -> dict:
    baseline_tcp_record = LENGTH_PREFIX_SIZE + jpeg_size
    hmac_package = HEADER_SIZE + jpeg_size + HMAC_SIZE
    secured_tcp_record = LENGTH_PREFIX_SIZE + hmac_package
    security_added_bytes = hmac_package - jpeg_size
    overhead_vs_jpeg_pct = (security_added_bytes / jpeg_size * 100.0) if jpeg_size else 0.0
    overhead_vs_baseline_record_pct = (
        security_added_bytes / baseline_tcp_record * 100.0
    ) if baseline_tcp_record else 0.0
    return {
        "jpeg_bytes": jpeg_size,
        "baseline_tcp_record_bytes": baseline_tcp_record,
        "hmac_header_bytes": HEADER_SIZE,
        "hmac_tag_bytes": HMAC_SIZE,
        "secured_hmac_package_bytes": hmac_package,
        "secured_tcp_record_bytes": secured_tcp_record,
        "security_added_bytes_per_frame": security_added_bytes,
        "overhead_vs_jpeg_payload_percent": overhead_vs_jpeg_pct,
        "overhead_vs_baseline_tcp_record_percent": overhead_vs_baseline_record_pct,
    }


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _tcp_baseline_server(port: int, frame_count: int, jpeg_size: int, done: threading.Event, out: dict) -> None:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", port))
    server.listen(1)
    conn, _ = server.accept()
    received = 0
    bytes_in = 0
    t0 = 0
    while received < frame_count:
        prefix = recv_exact(conn, LENGTH_PREFIX_SIZE)
        if prefix is None:
            break
        if received == 0:
            t0 = time.perf_counter()
        (length,) = struct.unpack(">I", prefix)
        if length != jpeg_size:
            raise RuntimeError("Baseline TCP length mismatch")
        payload = recv_exact(conn, length)
        if payload is None or len(payload) != length:
            break
        bytes_in += LENGTH_PREFIX_SIZE + length
        received += 1
    elapsed = time.perf_counter() - t0 if t0 else 0.0
    conn.close()
    server.close()
    out["received"] = received
    out["bytes"] = bytes_in
    out["elapsed_s"] = elapsed
    done.set()


def _tcp_baseline_client(port: int, jpeg_bytes: bytes, frame_count: int) -> None:
    sock = socket.create_connection(("127.0.0.1", port), timeout=30)
    record = struct.pack(">I", len(jpeg_bytes)) + jpeg_bytes
    for _ in range(frame_count):
        sock.sendall(record)
    sock.shutdown(socket.SHUT_RDWR)
    sock.close()


def _tcp_secured_server(
    port: int,
    frame_count: int,
    secret_key: bytes,
    session_id: bytes,
    done: threading.Event,
    out: dict,
) -> None:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", port))
    server.listen(1)
    conn, _ = server.accept()
    verifier = FrameVerifier(secret_key, session_id)
    verified = 0
    rejected = 0
    bytes_in = 0
    t0 = 0
    while verified < frame_count:
        prefix = recv_exact(conn, LENGTH_PREFIX_SIZE)
        if prefix is None:
            break
        if verified == 0 and rejected == 0:
            t0 = time.perf_counter()
        (pkg_len,) = struct.unpack(">I", prefix)
        if pkg_len < MIN_PACKAGE_SIZE or pkg_len > MAX_HMAC_PACKAGE_SIZE:
            raise RuntimeError("Invalid secured package length")
        package = recv_exact(conn, pkg_len)
        if package is None or len(package) != pkg_len:
            break
        bytes_in += LENGTH_PREFIX_SIZE + pkg_len
        result = verifier.verify(package)
        if result.reason != VerifyReason.ACCEPT:
            rejected += 1
            break
        verified += 1
    elapsed = time.perf_counter() - t0 if t0 else 0.0
    conn.close()
    server.close()
    out["verified"] = verified
    out["rejected"] = rejected
    out["bytes"] = bytes_in
    out["elapsed_s"] = elapsed
    done.set()


def _tcp_secured_client(
    port: int,
    jpeg_bytes: bytes,
    frame_count: int,
    secret_key: bytes,
    session_id: bytes,
) -> None:
    sock = socket.create_connection(("127.0.0.1", port), timeout=30)
    for i in range(frame_count):
        package = create_frame_package(jpeg_bytes, session_id, i, secret_key)
        send_framed(sock, package)
    sock.shutdown(socket.SHUT_RDWR)
    sock.close()


def run_tcp_trial_baseline(jpeg_bytes: bytes, frame_count: int) -> dict:
    port = _pick_free_port()
    done = threading.Event()
    out: dict = {}
    server = threading.Thread(
        target=_tcp_baseline_server,
        args=(port, frame_count, len(jpeg_bytes), done, out),
        daemon=True,
    )
    server.start()
    time.sleep(0.05)
    _tcp_baseline_client(port, jpeg_bytes, frame_count)
    done.wait(timeout=60)
    fps = out["received"] / out["elapsed_s"] if out.get("elapsed_s") else 0.0
    return {
        "sent": frame_count,
        "received": out.get("received", 0),
        "elapsed_s": out.get("elapsed_s", 0.0),
        "fps": fps,
        "bytes": out.get("bytes", 0),
        "mib_per_s": (out.get("bytes", 0) / (1024 * 1024)) / out["elapsed_s"]
        if out.get("elapsed_s")
        else 0.0,
    }


def run_tcp_trial_secured(
    jpeg_bytes: bytes,
    frame_count: int,
    secret_key: bytes,
    session_id: bytes,
) -> dict:
    port = _pick_free_port()
    done = threading.Event()
    out: dict = {}
    server = threading.Thread(
        target=_tcp_secured_server,
        args=(port, frame_count, secret_key, session_id, done, out),
        daemon=True,
    )
    server.start()
    time.sleep(0.05)
    _tcp_secured_client(port, jpeg_bytes, frame_count, secret_key, session_id)
    done.wait(timeout=60)
    fps = out["verified"] / out["elapsed_s"] if out.get("elapsed_s") else 0.0
    return {
        "sent": frame_count,
        "verified": out.get("verified", 0),
        "rejected": out.get("rejected", 0),
        "elapsed_s": out.get("elapsed_s", 0.0),
        "fps": fps,
        "bytes": out.get("bytes", 0),
        "mib_per_s": (out.get("bytes", 0) / (1024 * 1024)) / out["elapsed_s"]
        if out.get("elapsed_s")
        else 0.0,
    }


def run_tcp_benchmarks(
    jpeg_bytes: bytes,
    secret_key: bytes,
    session_id: bytes,
    tcp_frames: int,
    tcp_trials: int,
) -> dict:
    for _ in range(WARMUP_TCP):
        run_tcp_trial_baseline(jpeg_bytes, min(10, tcp_frames))
        run_tcp_trial_secured(jpeg_bytes, min(10, tcp_frames), secret_key, session_id)

    baseline_trials = []
    secured_trials = []
    for _ in range(tcp_trials):
        baseline_trials.append(run_tcp_trial_baseline(jpeg_bytes, tcp_frames))
        secured_trials.append(run_tcp_trial_secured(jpeg_bytes, tcp_frames, secret_key, session_id))

    base_fps = [t["fps"] for t in baseline_trials]
    sec_fps = [t["fps"] for t in secured_trials]
    median_base = statistics.median(base_fps)
    median_sec = statistics.median(sec_fps)
    diff = median_sec - median_base
    rel_pct = (diff / median_base * 100.0) if median_base else 0.0

    return {
        "localhost_note": "127.0.0.1 loopback — not WAN/latency representative",
        "frames_per_trial": tcp_frames,
        "trials": tcp_trials,
        "baseline_trials": baseline_trials,
        "secured_trials": secured_trials,
        "baseline_median_fps": median_base,
        "secured_median_fps": median_sec,
        "median_fps_difference": diff,
        "relative_throughput_change_percent": rel_pct,
    }


def print_summary(data: dict) -> None:
    print("\n--- VIGIL Stage 5 Benchmark ---\n")
    inp = data["input"]
    print("Input:")
    print(f"  Source: {inp['source']}")
    print(f"  JPEG size: {inp['jpeg_bytes']} bytes")
    if inp.get("jpeg_count", 1) > 1:
        print(f"  JPEG count (camera batch): {inp['jpeg_count']}")
        print(f"  Average JPEG size: {inp.get('avg_jpeg_bytes', inp['jpeg_bytes'])} bytes")
    print(f"  Iterations: {inp['iterations']}")
    print()

    crypto = data["crypto_microbenchmark"]
    print("Security processing:")
    print("Package/sign:")
    print(f"  Median: {crypto['sign']['median_ms']:.4f} ms")
    print(f"  Mean:   {crypto['sign']['mean_ms']:.4f} ms")
    print(f"  P95:    {crypto['sign']['p95_ms']:.4f} ms")
    print("Verify:")
    print(f"  Median: {crypto['verify']['median_ms']:.4f} ms")
    print(f"  Mean:   {crypto['verify']['mean_ms']:.4f} ms")
    print(f"  P95:    {crypto['verify']['p95_ms']:.4f} ms")
    print("Combined:")
    print(
        f"  Median security cost/frame (sign+verify medians): "
        f"{crypto['combined_median_sign_plus_verify_ms']:.4f} ms"
    )
    print(
        f"  Security-processing throughput: "
        f"{crypto['security_processing_throughput_fps']:.1f} frames/s"
    )
    print()

    oh = data["package_overhead"]
    print("Package overhead:")
    print(f"  JPEG payload: {oh['jpeg_bytes']} bytes")
    print(f"  HMAC header: {oh['hmac_header_bytes']} bytes")
    print(f"  HMAC tag: {oh['hmac_tag_bytes']} bytes")
    print(f"  Security overhead: {oh['security_added_bytes_per_frame']} bytes/frame")
    print(f"  Security overhead vs JPEG: {oh['overhead_vs_jpeg_payload_percent']:.4f} %")
    print()

    tcp = data["tcp_localhost"]
    print("Localhost TCP (127.0.0.1):")
    print(f"  Baseline median FPS: {tcp['baseline_median_fps']:.2f}")
    print(f"  Secured median FPS:  {tcp['secured_median_fps']:.2f}")
    print(f"  Relative throughput change: {tcp['relative_throughput_change_percent']:.2f} %")
    print()

    val = data["validation"]
    print("Validation:")
    print(f"  Baseline sent/received: {val['baseline_sent']}/{val['baseline_received']}")
    print(f"  Secured sent/verified: {val['secured_sent']}/{val['secured_verified']}")
    print(f"  Security rejects: {val['secured_rejected']}")
    print(f"  Result: {val['result']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="VIGIL Stage 5 HMAC benchmark")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE, help="JPEG file path")
    parser.add_argument("--iterations", type=int, default=1000, help="Crypto microbenchmark iterations")
    parser.add_argument("--tcp-frames", type=int, default=500, help="Frames per TCP trial")
    parser.add_argument("--tcp-trials", type=int, default=5, help="TCP benchmark trials")
    parser.add_argument("--camera", type=int, default=None, help="Optional webcam index for JPEG collection")
    parser.add_argument(
        "--capture-frames",
        type=int,
        default=300,
        help="Frames to capture when --camera is set",
    )
    args = parser.parse_args()

    jpeg_list: list[bytes] = []
    source = "file"

    if args.camera is not None:
        source = f"camera:{args.camera}"
        jpeg_list = capture_camera_jpegs(args.camera, args.capture_frames)
        jpeg_bytes = jpeg_list[0]
        avg_jpeg = int(statistics.mean(len(j) for j in jpeg_list))
    else:
        if not args.image.is_file():
            print(f"[ERROR] Image not found: {args.image}", file=sys.stderr)
            sys.exit(1)
        jpeg_bytes = args.image.read_bytes()
        jpeg_list = [jpeg_bytes]
        avg_jpeg = len(jpeg_bytes)

    secret_key = load_key(KEY_PATH)
    session_id = generate_session_id()

    crypto = run_crypto_microbenchmark(jpeg_bytes, secret_key, session_id, args.iterations)
    overhead = compute_package_overhead(len(jpeg_bytes))
    tcp = run_tcp_benchmarks(
        jpeg_bytes, secret_key, session_id, args.tcp_frames, args.tcp_trials
    )

    last_base = tcp["baseline_trials"][-1]
    last_sec = tcp["secured_trials"][-1]
    validation = {
        "baseline_sent": last_base["sent"],
        "baseline_received": last_base["received"],
        "secured_sent": last_sec["sent"],
        "secured_verified": last_sec["verified"],
        "secured_rejected": last_sec["rejected"],
        "result": "PASS"
        if last_base["sent"] == last_base["received"]
        and last_sec["sent"] == last_sec["verified"]
        and last_sec["rejected"] == 0
        else "FAIL",
    }

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / f"stage5_{timestamp}.json"

    payload = {
        "timestamp_utc": timestamp,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "opencv_version": cv2.__version__ if cv2 else None,
        "input": {
            "source": source,
            "image_path": str(args.image) if args.camera is None else None,
            "jpeg_bytes": len(jpeg_bytes),
            "jpeg_count": len(jpeg_list),
            "avg_jpeg_bytes": avg_jpeg,
            "iterations": args.iterations,
            "tcp_frames": args.tcp_frames,
            "tcp_trials": args.tcp_trials,
            "crypto_warmup": WARMUP_CRYPTO,
            "tcp_warmup_trials": WARMUP_TCP,
        },
        "timing_mechanism": "time.perf_counter_ns()",
        "crypto_microbenchmark": crypto,
        "package_overhead": overhead,
        "tcp_localhost": tcp,
        "validation": validation,
    }

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print_summary(payload)
    print(f"\nJSON results: {json_path}")

    if validation["result"] != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
