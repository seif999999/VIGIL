"""Probe Windows/OpenCV webcam backends and report ACTUAL frame sizes."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass

import cv2

RESOLUTIONS: tuple[tuple[int, int], ...] = (
    (1280, 720),
    (640, 480),
    (640, 360),
    (320, 240),
)

FPS_VALUES: tuple[int, ...] = (30, 15)

FOURCC_MODES: tuple[tuple[str, int | None], ...] = (
    ("default", None),
    ("MJPG", cv2.VideoWriter_fourcc(*"MJPG")),
)


@dataclass(frozen=True)
class BackendSpec:
    label: str
    api: int | None  # None => cv2.VideoCapture(index) only


BACKENDS: tuple[BackendSpec, ...] = (
    BackendSpec("MSMF", cv2.CAP_MSMF),
    BackendSpec("DSHOW", cv2.CAP_DSHOW),
    BackendSpec("CAP_ANY", None),
)


@dataclass
class ProbeRow:
    backend: str
    requested: str
    fourcc: str
    requested_fps: int
    actual_frame: str
    valid: str
    actual_fps_prop: str
    actual_fourcc_prop: str
    frames_read: int


def _fourcc_to_str(code: float) -> str:
    if code is None or code <= 0:
        return "n/a"
    icode = int(code)
    chars = "".join(chr((icode >> (8 * i)) & 0xFF) for i in range(4))
    if all(c.isprintable() and c != "\x00" for c in chars):
        return chars.strip("\x00") or "n/a"
    return f"0x{icode:08x}"


def _open_backend(index: int, spec: BackendSpec) -> cv2.VideoCapture | None:
    if spec.api is None:
        cap = cv2.VideoCapture(index)
    else:
        cap = cv2.VideoCapture(index, spec.api)
    if not cap.isOpened():
        cap.release()
        return None
    return cap


def _read_actual_frame(
    cap: cv2.VideoCapture, *, warmup_reads: int = 10
) -> tuple[bool, int, int, int]:
    """Return (valid, width, height, frames_successfully_read)."""
    ok_count = 0
    last_shape: tuple[int, int] | None = None
    for _ in range(warmup_reads):
        ok, frame = cap.read()
        if ok and frame is not None and frame.size > 0:
            ok_count += 1
            h, w = frame.shape[:2]
            last_shape = (int(w), int(h))
        time.sleep(0.02)
    if last_shape is None:
        return False, 0, 0, ok_count
    w, h = last_shape
    return True, w, h, ok_count


def _probe_one(
    index: int,
    spec: BackendSpec,
    width: int,
    height: int,
    fps: int,
    fourcc_label: str,
    fourcc_code: int | None,
    *,
    warmup_reads: int,
) -> ProbeRow:
    cap = _open_backend(index, spec)
    if cap is None:
        return ProbeRow(
            backend=spec.label,
            requested=f"{width}x{height}",
            fourcc=fourcc_label,
            requested_fps=fps,
            actual_frame="(open failed)",
            valid="NO",
            actual_fps_prop="n/a",
            actual_fourcc_prop="n/a",
            frames_read=0,
        )

    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if fourcc_code is not None:
            cap.set(cv2.CAP_PROP_FOURCC, fourcc_code)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
        cap.set(cv2.CAP_PROP_FPS, float(fps))

        valid, aw, ah, n_read = _read_actual_frame(cap, warmup_reads=warmup_reads)
        fps_prop = cap.get(cv2.CAP_PROP_FPS)
        fcc_prop = _fourcc_to_str(cap.get(cv2.CAP_PROP_FOURCC))

        actual = f"{aw}x{ah}" if valid else "(no valid frame)"
        return ProbeRow(
            backend=spec.label,
            requested=f"{width}x{height}",
            fourcc=fourcc_label,
            requested_fps=fps,
            actual_frame=actual,
            valid="YES" if valid else "NO",
            actual_fps_prop=f"{fps_prop:.2f}" if fps_prop and fps_prop > 0 else "n/a",
            actual_fourcc_prop=fcc_prop,
            frames_read=n_read,
        )
    finally:
        cap.release()


def _print_table(rows: list[ProbeRow]) -> None:
    headers = (
        "Backend",
        "Requested",
        "FourCC",
        "Req FPS",
        "Actual frame",
        "Valid",
        "FPS prop",
        "FourCC prop",
        "Frames",
    )
    fmt = (
        "{:<8} | {:<11} | {:<7} | {:<7} | {:<14} | {:<5} | {:<8} | {:<10} | {:>6}"
    )
    print(fmt.format(*headers))
    print("-" * 95)
    for r in rows:
        print(
            fmt.format(
                r.backend,
                r.requested,
                r.fourcc,
                str(r.requested_fps),
                r.actual_frame,
                r.valid,
                r.actual_fps_prop,
                r.actual_fourcc_prop,
                r.frames_read,
            )
        )


def _parse_actual_pixels(actual_frame: str) -> int:
    if "x" not in actual_frame:
        return 0
    try:
        w, h = actual_frame.split("x", 1)
        return int(w) * int(h)
    except ValueError:
        return 0


def _pick_best(rows: list[ProbeRow]) -> ProbeRow | None:
    valid_rows = [r for r in rows if r.valid == "YES" and "x" in r.actual_frame]
    if not valid_rows:
        return None

    def sort_key(r: ProbeRow) -> tuple:
        pixels = _parse_actual_pixels(r.actual_frame)
        # Prefer more frames read, then higher resolution, then MJPG, then 30 fps
        mjpg = 1 if r.fourcc == "MJPG" else 0
        return (pixels, r.frames_read, mjpg, r.requested_fps)

    return max(valid_rows, key=sort_key)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe webcam backends/resolutions (actual size from frame.shape)",
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument(
        "--warmup-reads",
        type=int,
        default=10,
        help="Frames to read per configuration before measuring",
    )
    args = parser.parse_args()

    print(f"[INFO] Probing camera index {args.camera}")
    print("[INFO] Authoritative resolution: last valid frame.shape (not cap.get alone)\n")

    rows: list[ProbeRow] = []
    for spec in BACKENDS:
        for width, height in RESOLUTIONS:
            for fps in FPS_VALUES:
                for fourcc_label, fourcc_code in FOURCC_MODES:
                    row = _probe_one(
                        args.camera,
                        spec,
                        width,
                        height,
                        fps,
                        fourcc_label,
                        fourcc_code,
                        warmup_reads=args.warmup_reads,
                    )
                    rows.append(row)

    _print_table(rows)

    best = _pick_best(rows)
    print("\n--- BEST WORKING CONFIGURATION ---")
    if best is None:
        print("None (no valid frames in any configuration)")
    else:
        print(f"backend: {best.backend}")
        print(f"requested resolution: {best.requested}")
        print(f"actual resolution: {best.actual_frame}")
        print(f"FourCC (requested): {best.fourcc}")
        print(f"FourCC (property): {best.actual_fourcc_prop}")
        print(f"requested FPS: {best.requested_fps}")
        print(f"FPS (property): {best.actual_fps_prop}")
        print(f"valid frames read in warmup: {best.frames_read}")

    max_pixels = max((_parse_actual_pixels(r.actual_frame) for r in rows), default=0)
    if max_pixels <= 160 * 120:
        print(
            "\n[RESULT] No configuration achieved resolution higher than 160x120 "
            "(or no valid frames above that size)."
        )
    else:
        best_wh = best.actual_frame if best else "n/a"
        print(f"\n[RESULT] Higher than 160x120 was achieved: best actual {best_wh}")

    return None


if __name__ == "__main__":
    main()
    sys.exit(0)
