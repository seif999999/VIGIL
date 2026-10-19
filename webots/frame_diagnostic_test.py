"""Unit checks for frame_diagnostic helpers."""

from __future__ import annotations

import sys

import numpy as np

from frame_diagnostic import analyze_bgr_frame


def main() -> None:
    black = np.zeros((48, 64, 3), dtype=np.uint8)
    scene = np.zeros((48, 64, 3), dtype=np.uint8)
    scene[10:40, 15:50] = (40, 120, 200)

    s0 = analyze_bgr_frame(black)
    s1 = analyze_bgr_frame(scene)
    ok = (
        s0["valid"]
        and s0["likely_black_feed"]
        and s1["valid"]
        and not s1["likely_black_feed"]
        and not s1["likely_uniform"]
    )
    print("--- frame_diagnostic_test ---")
    print(f"black_feed_flag: {'PASS' if s0['likely_black_feed'] else 'FAIL'}")
    print(f"scene_content_flag: {'PASS' if not s1['likely_black_feed'] else 'FAIL'}")
    print(f"Result: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
