"""Compare FasNet.analyze on an identical (frame, xywh) vs DeepFace extract_faces path.

Does not modify the friend-face-auth-test workspace. Uses DeepFace APIs only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from deepface import DeepFace
from deepface.modules import modeling

from fasnet_liveness import FasNetLiveness


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    args = parser.parse_args()
    frame = cv2.imread(str(args.image))
    if frame is None:
        raise SystemExit(f"Cannot read {args.image}")

    faces = DeepFace.extract_faces(
        img_path=frame, anti_spoofing=True, enforce_detection=False
    )
    if len(faces) != 1:
        print(json.dumps({"status": "skip", "face_count": len(faces)}, indent=2))
        return 0

    area = faces[0]["facial_area"]
    xywh = (int(area["x"]), int(area["y"]), int(area["w"]), int(area["h"]))
    extract_is_real = bool(faces[0].get("is_real"))
    extract_score = float(faces[0].get("antispoof_score") or 0.0)

    model = modeling.build_model(task="spoofing", model_name="Fasnet")
    direct_is_real, direct_score = model.analyze(img=frame, facial_area=xywh)

    wrapper = FasNetLiveness()
    # Reconstruct xyxy from xywh for wrapper API
    x, y, w, h = xywh
    wrap = wrapper.analyze_full_frame(frame, (x, y, x + w, y + h))

    report = {
        "image": str(args.image),
        "facial_area_xywh": xywh,
        "extract_faces": {"is_real": extract_is_real, "score": extract_score},
        "fasnet_analyze_same_bbox": {
            "is_real": bool(direct_is_real),
            "score": float(direct_score),
        },
        "vigil_wrapper": {
            "is_real": wrap.is_real,
            "score": wrap.antispoof_score,
            "class": wrap.predicted_class.value,
            "error": wrap.error,
        },
        "parity_analyze_vs_extract": {
            "is_real_match": bool(direct_is_real) == extract_is_real,
            "score_abs_delta": abs(float(direct_score) - extract_score),
        },
        "parity_wrapper_vs_analyze": {
            "is_real_match": wrap.is_real == bool(direct_is_real),
            "score_abs_delta": abs((wrap.antispoof_score or 0) - float(direct_score)),
        },
        "note": "Same bbox → FasNet.analyze must match wrapper; extract_faces uses same FasNet when anti_spoofing=True",
    }
    print(json.dumps(report, indent=2))
    ok = (
        report["parity_wrapper_vs_analyze"]["is_real_match"]
        and report["parity_wrapper_vs_analyze"]["score_abs_delta"] < 1e-5
        and report["parity_analyze_vs_extract"]["is_real_match"]
        and report["parity_analyze_vs_extract"]["score_abs_delta"] < 1e-5
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
