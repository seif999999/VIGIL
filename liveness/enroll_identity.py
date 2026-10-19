"""Enroll / expand a FaceNet identity into local JSON (gitignored data/).

Appends embeddings by default (does not wipe prior samples).
Validates a single SCRFD face before accepting a capture.
Does not load friend's embeddings pickle or copy personal datasets into git.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from face_bridge import load_face_detector_module
from facenet_identity import (
    DEFAULT_ENROLLMENT_PATH,
    MATCH_THRESHOLD,
    MATCH_THRESHOLD_STATUS,
    NEAR_DUPLICATE_DISTANCE,
    FaceNetIdentity,
    backfill_image_fingerprints,
    enrolled_provenance_sets,
    file_sha256,
    identity_embedding_count,
    is_near_duplicate,
    load_enrollments,
    normalize_path_key,
    save_enrollments,
)


def _open_camera(index: int) -> cv2.VideoCapture:
    backends = []
    if hasattr(cv2, "CAP_DSHOW"):
        backends.append(cv2.CAP_DSHOW)
    backends.append(cv2.CAP_ANY)
    last = None
    for backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                return cap
            cap.release()
            last = f"backend={backend} open but read failed"
        else:
            last = f"backend={backend} failed"
    raise RuntimeError(last or f"cannot open camera {index}")


def _validate_single_face(frame_bgr: np.ndarray, face_detector) -> tuple[bool, str]:
    det = face_detector.detect(frame_bgr)
    if det.face_count == 0 or det.decision.value == "NO_FACE":
        return False, "no_face"
    if det.face_count > 1 or det.decision.value == "MULTIPLE_FACES":
        return False, f"multiple_faces({det.face_count})"
    if det.decision.value != "FACE_OK" or det.face_count != 1:
        return False, f"face_gate={det.decision.value}"
    return True, "FACE_OK"


def capture_and_enroll(
    *,
    camera_index: int,
    identity: str,
    target_new: int,
    out_dir: Path,
    facenet: FaceNetIdentity,
    face_detector,
    enrollment_db: Path,
    duplicate_distance: float,
    allow_near_duplicates: bool,
) -> tuple[int, int, int]:
    """
    Interactive capture loop.
    Returns (accepted_new, rejected, near_dup_forced).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    db = load_enrollments(enrollment_db)
    prior = identity_embedding_count(db, identity)
    existing = list(
        ((db.get("identities") or {}).get(identity) or {}).get("embeddings") or []
    )
    session_embeds: list[list[float]] = list(existing)
    pending: list[tuple[list[float], str, str]] = []
    backfill_image_fingerprints(db, identity)

    accepted = 0
    rejected = 0
    near_forced = 0
    status = "SPACE=capture  ENTER=finish early  ESC=abort"
    last_reject = ""

    cap = _open_camera(camera_index)
    window = "VIGIL enroll — validate + append"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    print(
        f"[INFO] Identity {identity!r}: {prior} existing embedding(s). "
        f"Target +{target_new} new (frontal / slight L-R / distance / glasses). "
        f"Auth threshold unchanged: {MATCH_THRESHOLD} [{MATCH_THRESHOLD_STATUS}]"
    )
    print(
        "Keys: SPACE=capture candidate | ENTER=save & finish | ESC=abort without save"
    )

    try:
        while accepted < target_new:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[ERROR] Camera read failed")
                break
            display = frame.copy()
            lines = [
                f"{identity}: new {accepted}/{target_new}  (prior {prior})",
                status,
                last_reject,
            ]
            y = 28
            for line in lines:
                if not line:
                    continue
                cv2.putText(
                    display,
                    line[:90],
                    (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0) if accepted else (0, 255, 255),
                    2,
                )
                y += 26
            cv2.imshow(window, display)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC — abort, no DB write
                print("[ABORT] No database changes written.")
                pending.clear()
                accepted = 0
                break
            if key in (13, 10):  # ENTER — finish early if we have something
                if not pending:
                    print("[INFO] Nothing accepted yet; keep capturing or ESC to abort.")
                    last_reject = "Need >=1 accepted before ENTER"
                    continue
                print(f"[INFO] Finishing early with {accepted} new sample(s).")
                break
            if key != 32:  # not SPACE
                continue

            # --- validate face count ---
            face_ok, reason = _validate_single_face(frame, face_detector)
            if not face_ok:
                rejected += 1
                last_reject = f"REJECT {reason} — retake"
                status = "Retake: single clear face required"
                print(f"  [REJECT] {reason}")
                continue

            emb, ms, err = facenet.embed_bgr(frame, enforce_detection=True)
            if emb is None:
                rejected += 1
                last_reject = f"REJECT embed: {err}"
                status = "Retake: FaceNet embed failed"
                print(f"  [REJECT] embed failed: {err}")
                continue

            near, dist = is_near_duplicate(
                emb, session_embeds, duplicate_distance=duplicate_distance
            )
            if near and not allow_near_duplicates:
                # Brief review: show freeze; A=accept anyway, any other=retake
                last_reject = f"NEAR-DUP dist={dist:.2f} — A=force  other=retake"
                status = last_reject
                freeze = display.copy()
                cv2.putText(
                    freeze,
                    f"Near-duplicate (d={dist:.2f}<{duplicate_distance})",
                    (12, freeze.shape[0] - 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 165, 255),
                    2,
                )
                cv2.imshow(window, freeze)
                print(
                    f"  [NEAR-DUP] dist={dist:.3f} < {duplicate_distance}. "
                    "Press A to force-accept, any other key to retake."
                )
                k2 = cv2.waitKey(0) & 0xFF
                if k2 not in (ord("a"), ord("A")):
                    rejected += 1
                    last_reject = "Retake after near-dup"
                    continue
                near_forced += 1

            stamp = time.strftime("%Y%m%d_%H%M%S")
            path = out_dir / f"{identity}_{stamp}_{prior + accepted + 1:03d}.jpg"
            if not cv2.imwrite(str(path), frame):
                rejected += 1
                last_reject = "REJECT write failed"
                print(f"  [REJECT] could not write {path}")
                continue

            fp = file_sha256(path)
            pending.append((emb, str(path.resolve()), fp))
            session_embeds.append(emb)
            accepted += 1
            last_reject = ""
            status = f"ACCEPTED #{accepted} ({ms:.0f} ms) — continue"
            print(f"  [ACCEPT] {path.name} embed_ms={ms:.0f}")
            time.sleep(0.25)
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if not pending:
        print(
            f"[FAIL] No new embeddings saved. rejected={rejected} "
            f"prior_unchanged={prior}"
        )
        return 0, rejected, near_forced

    try:
        facenet.append_embeddings(
            identity, pending, enrollments=db, path=enrollment_db
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] Database update failed; prior file preserved. ({exc})")
        raise

    after = identity_embedding_count(load_enrollments(enrollment_db), identity)
    print(
        f"[OK] Enrollment complete for {identity!r}: "
        f"+{accepted} new, total={after}, rejected={rejected}, "
        f"near_dup_forced={near_forced} -> {enrollment_db}"
    )
    return accepted, rejected, near_forced


def _looks_like_vigil_capture(name: str, identity: str) -> bool:
    """Heuristic: prior pipeline captures named identity_YYYYMMDD_*."""
    prefix = f"{identity}_"
    if not name.startswith(prefix):
        return False
    rest = name[len(prefix) :]
    return len(rest) >= 8 and rest[:8].isdigit()


def enroll_from_images_dir(
    *,
    identity: str,
    images_dir: Path,
    facenet: FaceNetIdentity,
    face_detector,
    enrollment_db: Path,
    skip_near_duplicates: bool,
    duplicate_distance: float,
    import_ambiguous_legacy: bool,
) -> dict[str, int]:
    """
    Import JPG/JPEG/PNG from a folder with SHA256 provenance.

    Skips already-enrolled fingerprints/paths.
    Skips ambiguous legacy VIGIL captures not in provenance unless
    --import-ambiguous-legacy is set (asks for clarification by default).
    Near-duplicates are flagged; imported unless --skip-near-duplicates.
    Embeds SCRFD face crop with enforce_detection=False (same as live auth).
    Never modifies/deletes source photographs.
    """
    if not images_dir.is_dir():
        raise SystemExit(f"Not a directory: {images_dir}")

    paths = sorted(
        p
        for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not paths:
        raise SystemExit(f"No JPG/JPEG/PNG files in {images_dir}")

    db = load_enrollments(enrollment_db)
    prior = identity_embedding_count(db, identity)
    filled = backfill_image_fingerprints(db, identity)
    if filled:
        save_enrollments(db, enrollment_db)
        print(f"[INFO] Backfilled {filled} fingerprint(s) for existing source_images")
        db = load_enrollments(enrollment_db)

    known_fps, known_paths = enrolled_provenance_sets(db, identity)
    print(
        f"[INFO] {identity!r} prior embeddings={prior}; "
        f"fingerprints={len(known_fps)}; scanning {len(paths)} files in {images_dir}"
    )

    already: list[str] = []
    ambiguous: list[str] = []
    rejected: list[tuple[str, str]] = []
    near_flagged: list[str] = []
    pending: list[tuple[list[float], str, str]] = []
    session_embeds = list(
        ((db.get("identities") or {}).get(identity) or {}).get("embeddings") or []
    )

    for p in paths:
        try:
            fp = file_sha256(p)
        except OSError as exc:
            rejected.append((p.name, f"unreadable:{exc}"))
            continue
        path_key = normalize_path_key(p)
        if fp in known_fps or path_key in known_paths:
            already.append(p.name)
            continue

        if _looks_like_vigil_capture(p.name, identity) and not import_ambiguous_legacy:
            ambiguous.append(p.name)
            continue

        img = cv2.imread(str(p))
        if img is None:
            rejected.append((p.name, "unreadable_decode"))
            continue
        face_ok, reason = _validate_single_face(img, face_detector)
        if not face_ok:
            rejected.append((p.name, reason))
            continue

        det = face_detector.detect(img)
        face = det.faces[0]
        x1, y1, x2, y2 = int(face.x1), int(face.y1), int(face.x2), int(face.y2)
        h, w = img.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            rejected.append((p.name, "empty_face_crop"))
            continue

        emb, ms, err = facenet.embed_bgr(crop, enforce_detection=False)
        if emb is None:
            rejected.append((p.name, f"embed:{err}"))
            continue

        near, dist = is_near_duplicate(
            emb, session_embeds, duplicate_distance=duplicate_distance
        )
        if near:
            near_flagged.append(f"{p.name} dist={dist:.3f}")
            if skip_near_duplicates:
                rejected.append((p.name, f"near_duplicate:{dist:.3f}"))
                continue

        pending.append((emb, str(p.resolve()), fp))
        session_embeds.append(emb)
        known_fps.add(fp)
        print(f"  [ACCEPT] {p.name} embed_ms={ms:.0f}")

    print("\n--- Import summary ---")
    print(f"  already_enrolled_skipped: {len(already)}")
    for n in already:
        print(f"    - {n}")
    print(f"  ambiguous_legacy_skipped: {len(ambiguous)}")
    if ambiguous:
        print(
            "  [AMBIGUITY] These look like prior VIGIL captures but are NOT in "
            "enrollment provenance. Not imported (avoid guessing). Re-run with "
            "--import-ambiguous-legacy only if you intend to enroll them:"
        )
        for n in ambiguous:
            print(f"    - {n}")
    print(f"  rejected: {len(rejected)}")
    for n, why in rejected:
        print(f"    - {n}: {why}")
    if near_flagged and not skip_near_duplicates:
        print(f"  near_duplicate_flagged_but_imported: {len(near_flagged)}")
        for line in near_flagged:
            print(f"    - {line}")
    print(f"  newly_imported: {len(pending)}")

    if not pending:
        print(
            f"[FAIL] Nothing imported. prior_unchanged={prior}. "
            f"DB not modified (except optional fingerprint backfill)."
        )
        return {
            "imported": 0,
            "already": len(already),
            "ambiguous": len(ambiguous),
            "rejected": len(rejected),
            "prior": prior,
            "total": prior,
        }

    facenet.append_embeddings(
        identity, pending, enrollments=db, path=enrollment_db
    )
    after = identity_embedding_count(load_enrollments(enrollment_db), identity)
    print(
        f"[OK] images-dir done: prior={prior} imported={len(pending)} "
        f"total_now={after} -> {enrollment_db}"
    )
    return {
        "imported": len(pending),
        "already": len(already),
        "ambiguous": len(ambiguous),
        "rejected": len(rejected),
        "prior": prior,
        "total": after,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VIGIL FaceNet enrollment — append-safe, validated captures"
    )
    parser.add_argument(
        "--identity",
        required=True,
        help="Person name / identity id (e.g. Seif). Used as DB key and default display name.",
    )
    parser.add_argument(
        "--display-name",
        default=None,
        help="Optional UI display name (defaults to --identity)",
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument(
        "--count",
        type=int,
        default=15,
        help="Number of NEW embeddings to add (default 15; try 15–20)",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help="Import JPG/JPEG/PNG from this folder (append + fingerprint skip)",
    )
    parser.add_argument(
        "--from-dir",
        type=Path,
        default=None,
        help="Alias for --images-dir",
    )
    parser.add_argument(
        "--enrollment-db",
        type=Path,
        default=DEFAULT_ENROLLMENT_PATH,
        help="JSON enrollment path (default: liveness/data/enrollments.json)",
    )
    parser.add_argument(
        "--photo-dir",
        type=Path,
        default=None,
        help="Where to store capture photos (default: data/enroll_photos/<id>)",
    )
    parser.add_argument(
        "--duplicate-distance",
        type=float,
        default=NEAR_DUPLICATE_DISTANCE,
        help="Flag near-duplicates below this euclidean distance (not auth threshold)",
    )
    parser.add_argument(
        "--allow-near-duplicates",
        action="store_true",
        help="Do not prompt on near-duplicates during live capture",
    )
    parser.add_argument(
        "--skip-near-duplicates",
        action="store_true",
        help="When using --images-dir, skip near-duplicates instead of importing them",
    )
    parser.add_argument(
        "--import-ambiguous-legacy",
        action="store_true",
        help=(
            "Also import VIGIL-named captures in the folder that are missing from "
            "provenance (default: skip and report — do not guess)"
        ),
    )
    args = parser.parse_args()

    images_dir = args.images_dir or args.from_dir

    face_mod = load_face_detector_module()
    face_detector = face_mod.FaceDetector()
    facenet = FaceNetIdentity()
    facenet.warm_up()
    if facenet.init_ms is not None:
        print(f"[INFO] FaceNet warm-up: {facenet.init_ms:.0f} ms")

    prior = identity_embedding_count(load_enrollments(args.enrollment_db), args.identity)
    display = (args.display_name or args.identity).strip()
    print(
        f"[INFO] Existing embeddings for {args.identity!r} "
        f"(display={display!r}): {prior} (append mode - will not overwrite)"
    )

    def _apply_display_name() -> None:
        db = load_enrollments(args.enrollment_db)
        ids = db.setdefault("identities", {})
        if args.identity not in ids:
            return
        ids[args.identity]["display_name"] = display
        save_enrollments(db, args.enrollment_db)

    _apply_display_name()

    if images_dir is not None:
        enroll_from_images_dir(
            identity=args.identity,
            images_dir=images_dir,
            facenet=facenet,
            face_detector=face_detector,
            enrollment_db=args.enrollment_db,
            skip_near_duplicates=args.skip_near_duplicates,
            duplicate_distance=args.duplicate_distance,
            import_ambiguous_legacy=args.import_ambiguous_legacy,
        )
        _apply_display_name()
        return

    photo_dir = args.photo_dir or (
        DEFAULT_ENROLLMENT_PATH.parent / "enroll_photos" / args.identity
    )
    capture_and_enroll(
        camera_index=args.camera,
        identity=args.identity,
        target_new=args.count,
        out_dir=photo_dir,
        facenet=facenet,
        face_detector=face_detector,
        enrollment_db=args.enrollment_db,
        duplicate_distance=args.duplicate_distance,
        allow_near_duplicates=args.allow_near_duplicates,
    )
    _apply_display_name()


if __name__ == "__main__":
    main()
