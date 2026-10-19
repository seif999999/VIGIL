"""VIGIL Stage 4: attacker-in-the-middle proxy on the digital TCP path (no secret key)."""

from __future__ import annotations

import argparse
import secrets
import socket
import struct
import sys
from pathlib import Path

from protocol import (
    HEADER_SIZE,
    HMAC_SIZE,
    MAX_JPEG_SIZE,
    MIN_PACKAGE_SIZE,
    PROTOCOL_VERSION,
    SESSION_ID_SIZE,
)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_LISTEN_HOST = "127.0.0.1"
DEFAULT_LISTEN_PORT = 5001
DEFAULT_VERIFIER_HOST = "127.0.0.1"
DEFAULT_VERIFIER_PORT = 5000
DEFAULT_ATTACK_AFTER = 20
LENGTH_PREFIX_SIZE = 4
MAX_HMAC_PACKAGE_SIZE = HEADER_SIZE + MAX_JPEG_SIZE + HMAC_SIZE
DEFAULT_REPLAY_FILE = BASE_DIR / "replay_artifacts" / "old_session_package.bin"
OLD_FRAME_SAVE_AT = 5
OLD_FRAME_REPLAY_AFTER = 20

class VerifierConnectionClosed(Exception):
    """Upstream verifier closed the TCP connection (expected after security reject)."""


def _is_upstream_closed_error(exc: OSError) -> bool:
    if isinstance(exc, (ConnectionAbortedError, ConnectionResetError, BrokenPipeError)):
        return True
    winerror = getattr(exc, "winerror", None)
    if winerror in (10053, 10054):
        return True
    if exc.errno in (104, 32, 54):
        return True
    return False


ATTACK_MODES = frozenset(
    {
        "normal",
        "tamper-jpeg",
        "replace-jpeg",
        "inject-fake",
        "duplicate",
        "old-frame-replay",
        "out-of-order",
        "record-old-session",
        "old-session-replay",
    }
)


def recv_exact(sock: socket.socket, nbytes: int) -> bytes | None:
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


def send_framed(sock: socket.socket, package: bytes) -> None:
    try:
        sock.sendall(struct.pack(">I", len(package)) + package)
    except OSError as exc:
        if _is_upstream_closed_error(exc):
            raise VerifierConnectionClosed from exc
        raise


def recv_framed(sock: socket.socket) -> bytes | None:
    prefix = recv_exact(sock, LENGTH_PREFIX_SIZE)
    if prefix is None:
        return None
    (pkg_len,) = struct.unpack(">I", prefix)
    if pkg_len < MIN_PACKAGE_SIZE or pkg_len > MAX_HMAC_PACKAGE_SIZE:
        print(f"[ERROR] Invalid package length from capture: {pkg_len}", file=sys.stderr)
        return None
    package = recv_exact(sock, pkg_len)
    if package is None or len(package) != pkg_len:
        return None
    return package


def _session_hex_prefix(session_id: bytes, n: int = 8) -> str:
    return session_id[: n // 2].hex()


def parse_package_fields(package: bytes) -> tuple[int, bytes, int, bytes, bytes] | None:
    if len(package) < MIN_PACKAGE_SIZE:
        return None
    version = package[0]
    session_id = package[1 : 1 + SESSION_ID_SIZE]
    frame_counter, jpeg_length = struct.unpack(
        ">QI", package[1 + SESSION_ID_SIZE : HEADER_SIZE]
    )
    expected_len = HEADER_SIZE + jpeg_length + HMAC_SIZE
    if len(package) != expected_len:
        return None
    jpeg_start = HEADER_SIZE
    jpeg_end = jpeg_start + jpeg_length
    jpeg_bytes = package[jpeg_start:jpeg_end]
    hmac_tag = package[jpeg_end : jpeg_end + HMAC_SIZE]
    return version, session_id, frame_counter, jpeg_bytes, hmac_tag


def rebuild_header(version: int, session_id: bytes, frame_counter: int, jpeg_length: int) -> bytes:
    return (
        struct.pack(">B", version)
        + session_id
        + struct.pack(">QI", frame_counter, jpeg_length)
    )


def tamper_one_jpeg_byte(package: bytes) -> bytes | None:
    fields = parse_package_fields(package)
    if fields is None:
        return None
    _, _, counter, jpeg_bytes, _ = fields
    if len(jpeg_bytes) == 0:
        print("[ERROR] Cannot tamper empty JPEG payload.", file=sys.stderr)
        return None
    tampered = bytearray(package)
    tampered[HEADER_SIZE] ^= 0x01
    return bytes(tampered)


def replace_jpeg_payload(package: bytes, replacement_jpeg: bytes) -> bytes | None:
    fields = parse_package_fields(package)
    if fields is None:
        return None
    version, session_id, frame_counter, _, old_tag = fields
    header = rebuild_header(version, session_id, frame_counter, len(replacement_jpeg))
    return header + replacement_jpeg + old_tag


def build_fake_injected_package(
    session_id: bytes, frame_counter: int, jpeg_bytes: bytes
) -> bytes:
    header = rebuild_header(PROTOCOL_VERSION, session_id, frame_counter, len(jpeg_bytes))
    fake_tag = secrets.token_bytes(HMAC_SIZE)
    return header + jpeg_bytes + fake_tag


class ProxyRuntime:
    def __init__(
        self,
        mode: str,
        attack_after: int,
        replay_file: Path,
        replacement_image: Path | None = None,
    ) -> None:
        self.mode = mode
        self.attack_after = attack_after
        self.replay_file = replay_file
        self.replacement_image = replacement_image
        self.attack_executed = False
        self.packages_intercepted = 0
        self.packages_forwarded = 0
        self.packages_injected_replayed = 0
        self.session_id: bytes | None = None
        self.saved_old_frame: bytes | None = None
        self.out_of_order_a: bytes | None = None
        self.saved_replay_bytes: bytes | None = None
        if mode == "old-session-replay":
            if not replay_file.is_file():
                print(f"[ERROR] Replay file not found: {replay_file}", file=sys.stderr)
                raise SystemExit(1)
            self.saved_replay_bytes = replay_file.read_bytes()

    def process(self, package: bytes, upstream: socket.socket) -> bool:
        """Forward (possibly modified) traffic to verifier. Returns False to stop."""
        self.packages_intercepted += 1

        if self.mode == "old-session-replay" and not self.attack_executed:
            self.attack_executed = True
            print("[ATTACK] Sending saved Session A package before first Session B frame.")
            send_framed(upstream, self.saved_replay_bytes)  # type: ignore[arg-type]
            self.packages_injected_replayed += 1

        if self.mode == "record-old-session" and self.packages_intercepted == 1:
            self.replay_file.parent.mkdir(parents=True, exist_ok=True)
            self.replay_file.write_bytes(package)
            print(f"[INFO] Saved first signed package to {self.replay_file}")

        if self.mode == "old-frame-replay":
            if self.packages_intercepted == OLD_FRAME_SAVE_AT:
                self.saved_old_frame = package
                print(f"[INFO] Saved early frame package (intercept #{self.packages_intercepted}).")
            elif (
                self.saved_old_frame is not None
                and not self.attack_executed
                and self.packages_intercepted
                >= OLD_FRAME_SAVE_AT + OLD_FRAME_REPLAY_AFTER
            ):
                self.attack_executed = True
                print("[ATTACK] Replaying saved old valid frame package.")
                send_framed(upstream, self.saved_old_frame)
                self.packages_injected_replayed += 1

        if self.mode == "out-of-order" and not self.attack_executed:
            if self.packages_intercepted == self.attack_after:
                self.out_of_order_a = package
                return True
            if self.packages_intercepted == self.attack_after + 1:
                b_pkg = package
                a_pkg = self.out_of_order_a
                assert a_pkg is not None
                self.attack_executed = True
                fields = parse_package_fields(b_pkg)
                counter = fields[2] if fields else "?"
                print(f"[ATTACK] Out-of-order: sending counter N+1 ({counter}) before N.")
                send_framed(upstream, b_pkg)
                self.packages_forwarded += 1
                send_framed(upstream, a_pkg)
                self.packages_forwarded += 1
                print("[ATTACK] Re-sending original N+1 package.")
                send_framed(upstream, b_pkg)
                self.packages_forwarded += 1
                return True

        if (
            not self.attack_executed
            and self.mode
            in {"tamper-jpeg", "replace-jpeg", "inject-fake", "duplicate"}
            and self.packages_intercepted == self.attack_after
        ):
            return self._execute_attack(package, upstream)

        send_framed(upstream, package)
        self.packages_forwarded += 1
        return True

    def _execute_attack(self, package: bytes, upstream: socket.socket) -> bool:
        fields = parse_package_fields(package)
        if fields is None:
            print("[ERROR] Attack skipped: malformed package.", file=sys.stderr)
            send_framed(upstream, package)
            self.packages_forwarded += 1
            return True

        _, session_id, frame_counter, jpeg_bytes, _ = fields
        self.attack_executed = True

        if self.mode == "tamper-jpeg":
            tampered = tamper_one_jpeg_byte(package)
            if tampered is None:
                send_framed(upstream, package)
            else:
                print(f"[ATTACK] Tampered one JPEG byte for counter={frame_counter}")
                send_framed(upstream, tampered)
            self.packages_forwarded += 1
            return True

        if self.mode == "replace-jpeg":
            assert self.replacement_image is not None
            replacement = self.replacement_image.read_bytes()
            replaced = replace_jpeg_payload(package, replacement)
            if replaced is None:
                send_framed(upstream, package)
            else:
                print(f"[ATTACK] Replaced JPEG for counter={frame_counter} (HMAC unchanged).")
                send_framed(upstream, replaced)
            self.packages_forwarded += 1
            return True

        if self.mode == "inject-fake":
            fake_jpeg = b"\xff\xd8\xff\xe0fake-inject"
            fake_pkg = build_fake_injected_package(session_id, frame_counter, fake_jpeg)
            print(f"[ATTACK] Injecting fake package counter={frame_counter} (invalid HMAC).")
            send_framed(upstream, fake_pkg)
            self.packages_injected_replayed += 1
            print(f"[INFO] Forwarding genuine package counter={frame_counter}.")
            send_framed(upstream, package)
            self.packages_forwarded += 2
            return True

        if self.mode == "duplicate":
            print(f"[ATTACK] Duplicating package counter={frame_counter}.")
            send_framed(upstream, package)
            send_framed(upstream, package)
            self.packages_forwarded += 2
            self.packages_injected_replayed += 1
            return True

        send_framed(upstream, package)
        self.packages_forwarded += 1
        return True


def run(
    listen_host: str,
    listen_port: int,
    verifier_host: str,
    verifier_port: int,
    mode: str,
    attack_after: int,
    replay_file: Path,
    replacement_image: Path | None = None,
) -> int:
    if mode not in ATTACK_MODES:
        print(f"[ERROR] Unknown mode: {mode}", file=sys.stderr)
        return 1

    if mode == "replace-jpeg":
        if replacement_image is None:
            print(
                "[ERROR] replace-jpeg requires --replacement-image PATH",
                file=sys.stderr,
            )
            return 1
        if not replacement_image.is_file():
            print(
                f"[ERROR] Replacement image not found: {replacement_image}",
                file=sys.stderr,
            )
            return 1

    print(f"[INFO] Proxy mode: {mode}")
    print(f"[INFO] Attack-after intercept count: {attack_after}")

    upstream = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        upstream.connect((verifier_host, verifier_port))
    except OSError as exc:
        print(
            f"[ERROR] Could not connect to verifier at {verifier_host}:{verifier_port} ({exc})",
            file=sys.stderr,
        )
        print("[ERROR] Start verifier_server.py first.", file=sys.stderr)
        upstream.close()
        return 1

    session_id = recv_exact(upstream, SESSION_ID_SIZE)
    if session_id is None or len(session_id) != SESSION_ID_SIZE:
        print("[ERROR] Failed to receive session from verifier.", file=sys.stderr)
        upstream.close()
        return 1

    print(f"[INFO] Session forwarded: {_session_hex_prefix(session_id)}...")

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        listener.bind((listen_host, listen_port))
    except OSError as exc:
        print(f"[ERROR] Could not bind {listen_host}:{listen_port} ({exc})", file=sys.stderr)
        upstream.close()
        listener.close()
        return 1

    listener.listen(1)
    print(f"[INFO] Listening for trusted capture on {listen_host}:{listen_port}")

    runtime = ProxyRuntime(
        mode=mode,
        attack_after=attack_after,
        replay_file=replay_file,
        replacement_image=replacement_image,
    )
    runtime.session_id = session_id

    downstream: socket.socket | None = None
    try:
        downstream, addr = listener.accept()
        print(f"[INFO] Capture connected from {addr[0]}:{addr[1]}")
        downstream.sendall(session_id)

        while True:
            package = recv_framed(downstream)
            if package is None:
                print("[INFO] Capture disconnected.")
                break
            try:
                if not runtime.process(package, upstream):
                    break
            except VerifierConnectionClosed:
                print(
                    "[INFO] Verifier closed the connection after security rejection."
                )
                print("[INFO] Proxy stopping cleanly.")
                break

        if not runtime.attack_executed and mode not in ("normal", "record-old-session"):
            print(
                f"[WARN] Attack did not execute (only {runtime.packages_intercepted} packages intercepted).",
                file=sys.stderr,
            )

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        if downstream is not None:
            try:
                downstream.close()
            except OSError:
                pass
        try:
            upstream.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        upstream.close()
        listener.close()

        attack_label = "YES" if runtime.attack_executed else "NO"
        if mode == "normal":
            attack_label = "N/A"
        print("\n--- VIGIL Stage 4 Proxy Summary ---")
        print(f"Mode: {mode}")
        print(f"Real packages intercepted: {runtime.packages_intercepted}")
        print(f"Packages forwarded: {runtime.packages_forwarded}")
        print(f"Packages injected/replayed: {runtime.packages_injected_replayed}")
        print(f"Attack executed: {attack_label}")
        print("Result: COMPLETE")

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VIGIL Stage 4 attacker proxy (no HMAC secret)",
    )
    parser.add_argument("--listen-host", default=DEFAULT_LISTEN_HOST)
    parser.add_argument("--listen-port", type=int, default=DEFAULT_LISTEN_PORT)
    parser.add_argument("--verifier-host", default=DEFAULT_VERIFIER_HOST)
    parser.add_argument("--verifier-port", type=int, default=DEFAULT_VERIFIER_PORT)
    parser.add_argument(
        "--mode",
        default="normal",
        choices=sorted(ATTACK_MODES),
    )
    parser.add_argument(
        "--attack-after",
        type=int,
        default=DEFAULT_ATTACK_AFTER,
        help="Intercept count at which one-shot attack runs (default: 20)",
    )
    parser.add_argument(
        "--replay-file",
        type=Path,
        default=DEFAULT_REPLAY_FILE,
        help="Path for record-old-session / old-session-replay package bytes",
    )
    parser.add_argument(
        "--replacement-image",
        type=Path,
        default=None,
        help="JPEG file for replace-jpeg mode (required when mode is replace-jpeg)",
    )
    args = parser.parse_args()
    sys.exit(
        run(
            listen_host=args.listen_host,
            listen_port=args.listen_port,
            verifier_host=args.verifier_host,
            verifier_port=args.verifier_port,
            mode=args.mode,
            attack_after=args.attack_after,
            replay_file=args.replay_file,
            replacement_image=args.replacement_image,
        )
    )


if __name__ == "__main__":
    main()
