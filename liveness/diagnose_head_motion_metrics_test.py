"""Unit checks for head-motion diagnostic helpers (no camera)."""

from __future__ import annotations

import sys

import numpy as np

from diagnose_head_motion_liveness import landmark_yaw_proxy, laplacian_variance


def main() -> None:
    gray_blur = np.zeros((64, 64), dtype=np.uint8)
    gray_sharp = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
    assert laplacian_variance(gray_blur) < laplacian_variance(gray_sharp)

    lm_center = np.array(
        [[30, 30], [50, 30], [40, 40], [32, 50], [48, 50]], dtype=np.float32
    )
    lm_yaw = np.array(
        [[20, 30], [50, 30], [48, 40], [22, 50], [48, 50]], dtype=np.float32
    )
    y0 = landmark_yaw_proxy(lm_center)
    y1 = landmark_yaw_proxy(lm_yaw)
    ok = y0 is not None and y1 is not None and abs(y1) > abs(y0)
    print("--- diagnose_head_motion_metrics_test ---")
    print(f"laplacian_sensitivity: PASS")
    print(f"yaw_proxy: {'PASS' if ok else 'FAIL'}")
    print(f"Result: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
