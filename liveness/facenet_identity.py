"""FaceNet-128 identity via DeepFace public API + local JSON enrollments.

No untrusted pickle loading. Enrollment store is gitignored under data/.
Match threshold 9.0 mirrors the friend's live_access.py value and is explicitly
UNVALIDATED (DeepFace published Facenet euclidean default is 10.0).
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

MODEL_NAME = "Facenet"
EMBEDDING_DIM = 128
MATCH_THRESHOLD = 9.0
MATCH_THRESHOLD_STATUS = "UNVALIDATED_FRIEND_THRESHOLD"
# Practical near-duplicate flag for enrollment quality (not an auth threshold).
NEAR_DUPLICATE_DISTANCE = 4.0

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_ENROLLMENT_PATH = BASE_DIR / "data" / "enrollments.json"


def euclidean_distance(a: Any, b: Any) -> float:
    return float(
        np.linalg.norm(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))
    )


@dataclass(frozen=True)
class IdentityMatch:
    recognized: bool
    identity: str | None
    best_identity_candidate: str | None
    best_distance: float | None
    distances: dict[str, float]
    threshold: float
    threshold_status: str
    embedding_dim: int
    inference_ms: float
    error: str | None = None
    display_name: str | None = None
    display_name_candidate: str | None = None


def empty_enrollment_db() -> dict[str, Any]:
    return {
        "model_name": MODEL_NAME,
        "embedding_dim": EMBEDDING_DIM,
        "match_threshold": MATCH_THRESHOLD,
        "match_threshold_status": MATCH_THRESHOLD_STATUS,
        "identities": {},
    }


def display_name_of(rec: dict[str, Any], identity_id: str) -> str:
    """Human-readable name for UI; falls back to identity id."""
    name = rec.get("display_name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return identity_id


def rename_identity(
    db: dict[str, Any],
    *,
    old_id: str,
    new_id: str,
    display_name: str | None = None,
) -> dict[str, Any]:
    """
    Rename an identity key in-memory. Preserves embeddings and provenance.
    Refuses to merge into an existing different identity.
    """
    identities = db.setdefault("identities", {})
    if old_id not in identities:
        raise KeyError(f"Identity {old_id!r} not found")
    if new_id != old_id and new_id in identities:
        raise ValueError(
            f"Cannot rename {old_id!r} -> {new_id!r}: target already exists "
            "(refusing to merge embeddings)"
        )
    rec = ensure_identity_record(copy.deepcopy(identities[old_id]))
    n_emb = len(rec.get("embeddings") or [])
    if n_emb == 0:
        raise ValueError(f"Identity {old_id!r} has no embeddings")
    label = (display_name or new_id).strip()
    if not label:
        raise ValueError("display_name must be non-empty")
    rec["display_name"] = label
    if new_id != old_id:
        del identities[old_id]
    identities[new_id] = rec
    return db


def migrate_identity_label(
    *,
    old_id: str,
    new_id: str,
    display_name: str | None = None,
    path: Path | None = None,
) -> dict[str, Any]:
    """Atomic on-disk rename. Returns summary dict."""
    path = path or DEFAULT_ENROLLMENT_PATH
    db = load_enrollments(path)
    before = identity_embedding_count(db, old_id)
    if before == 0 and old_id not in (db.get("identities") or {}):
        raise KeyError(f"Identity {old_id!r} not found in {path}")
    # Collision check before mutate
    identities = db.get("identities") or {}
    if new_id != old_id and new_id in identities:
        raise ValueError(
            f"Conflict: {new_id!r} already enrolled "
            f"({identity_embedding_count(db, new_id)} embeddings). Aborting."
        )
    db = copy.deepcopy(db)
    rename_identity(
        db, old_id=old_id, new_id=new_id, display_name=display_name or new_id
    )
    after = identity_embedding_count(db, new_id)
    if after != before:
        raise RuntimeError(
            f"Embedding count mismatch during rename: {before} -> {after}"
        )
    save_enrollments(db, path)
    return {
        "old_id": old_id,
        "new_id": new_id,
        "display_name": display_name or new_id,
        "embeddings": after,
        "path": str(path),
    }


def load_enrollments(path: Path | None = None) -> dict[str, Any]:
    path = path or DEFAULT_ENROLLMENT_PATH
    if not path.is_file():
        return empty_enrollment_db()
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "identities" not in data or not isinstance(data["identities"], dict):
        raise ValueError(f"Invalid enrollment DB (missing identities): {path}")
    return data


def identity_embedding_count(db: dict[str, Any], identity: str) -> int:
    rec = (db.get("identities") or {}).get(identity) or {}
    return len(rec.get("embeddings") or [])


def file_sha256(path: Path) -> str:
    """Content fingerprint for enrollment provenance (not a biometric)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def normalize_path_key(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").lower()


def ensure_identity_record(rec: dict[str, Any], *, identity_id: str | None = None) -> dict[str, Any]:
    rec.setdefault("embeddings", [])
    rec.setdefault("source_images", [])
    rec.setdefault("image_fingerprints", [])
    if identity_id is not None:
        rec.setdefault("display_name", identity_id)
    elif "display_name" not in rec and identity_id:
        rec["display_name"] = identity_id
    return rec


def backfill_image_fingerprints(db: dict[str, Any], identity: str) -> int:
    """
    Fill missing SHA256 fingerprints from existing source_images when files exist.
    Does not re-embed. Returns number of fingerprints newly written in-memory.
    """
    identities = db.setdefault("identities", {})
    rec = ensure_identity_record(identities.setdefault(identity, {}))
    sources = rec["source_images"]
    fps = rec["image_fingerprints"]
    # Align list lengths carefully: only append missing slots for existing sources
    while len(fps) < len(sources):
        fps.append("")
    filled = 0
    for i, src in enumerate(sources):
        if fps[i]:
            continue
        p = Path(src)
        if not p.is_file():
            continue
        fps[i] = file_sha256(p)
        filled += 1
    return filled


def enrolled_provenance_sets(
    db: dict[str, Any], identity: str
) -> tuple[set[str], set[str]]:
    """Return (fingerprint_set, normalized_source_path_set) for an identity."""
    rec = ensure_identity_record(
        (db.get("identities") or {}).get(identity) or {}
    )
    fps = {fp for fp in (rec.get("image_fingerprints") or []) if fp}
    paths: set[str] = set()
    for src in rec.get("source_images") or []:
        try:
            paths.add(normalize_path_key(Path(src)))
        except OSError:
            paths.add(str(src).replace("\\", "/").lower())
    return fps, paths


def min_distance_to_embeddings(
    embedding: list[float],
    embeddings: list[Any],
) -> float | None:
    if not embeddings:
        return None
    return float(min(euclidean_distance(embedding, e) for e in embeddings))


def is_near_duplicate(
    embedding: list[float],
    existing_embeddings: list[Any],
    *,
    duplicate_distance: float = NEAR_DUPLICATE_DISTANCE,
) -> tuple[bool, float | None]:
    """Flag enrollment near-duplicates; does not change MATCH_THRESHOLD."""
    d = min_distance_to_embeddings(embedding, existing_embeddings)
    if d is None:
        return False, None
    return d < duplicate_distance, d


def save_enrollments(data: dict[str, Any], path: Path | None = None) -> Path:
    """Atomic write: temp file + os.replace. Leaves prior DB intact on failure."""
    path = path or DEFAULT_ENROLLMENT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(data)
    payload.setdefault("model_name", MODEL_NAME)
    payload.setdefault("embedding_dim", EMBEDDING_DIM)
    payload.setdefault("match_threshold", MATCH_THRESHOLD)
    payload.setdefault("match_threshold_status", MATCH_THRESHOLD_STATUS)

    tmp = path.with_suffix(path.suffix + ".tmp")
    bak = path.with_suffix(path.suffix + ".bak")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        if path.is_file():
            shutil.copy2(path, bak)
        os.replace(tmp, path)
    except Exception:
        if tmp.is_file():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise
    return path


class FaceNetIdentity:
    """Lazy FaceNet via DeepFace.represent; models cached by DeepFace."""

    def __init__(self) -> None:
        self._warmed = False
        self._init_ms: float | None = None

    @property
    def init_ms(self) -> float | None:
        return self._init_ms

    def warm_up(self) -> None:
        if self._warmed:
            return
        t0 = time.perf_counter()
        from deepface import DeepFace  # noqa: WPS433

        DeepFace.build_model(MODEL_NAME)
        self._init_ms = (time.perf_counter() - t0) * 1000.0
        self._warmed = True

    def embed_bgr(
        self,
        image_bgr: np.ndarray,
        *,
        enforce_detection: bool = False,
    ) -> tuple[list[float] | None, float, str | None]:
        """Return (embedding, inference_ms, error). Fail closed on exceptions."""
        t0 = time.perf_counter()
        try:
            from deepface import DeepFace  # noqa: WPS433

            self.warm_up()
            result = DeepFace.represent(
                img_path=image_bgr,
                model_name=MODEL_NAME,
                enforce_detection=enforce_detection,
            )
            emb = list(result[0]["embedding"])
            if len(emb) != EMBEDDING_DIM:
                return None, (time.perf_counter() - t0) * 1000.0, (
                    f"unexpected_embedding_dim={len(emb)}"
                )
            return emb, (time.perf_counter() - t0) * 1000.0, None
        except Exception as exc:  # noqa: BLE001
            return None, (time.perf_counter() - t0) * 1000.0, f"{type(exc).__name__}: {exc}"

    def match_crop(
        self,
        face_bgr: np.ndarray,
        enrollments: dict[str, Any],
        *,
        threshold: float = MATCH_THRESHOLD,
    ) -> IdentityMatch:
        emb, ms, err = self.embed_bgr(face_bgr, enforce_detection=False)
        if emb is None:
            return IdentityMatch(
                recognized=False,
                identity=None,
                best_identity_candidate=None,
                best_distance=None,
                distances={},
                threshold=threshold,
                threshold_status=MATCH_THRESHOLD_STATUS,
                embedding_dim=0,
                inference_ms=ms,
                error=err or "embedding_failed",
            )
        match = match_identity(emb, enrollments, threshold=threshold)
        return IdentityMatch(
            recognized=match.recognized,
            identity=match.identity,
            best_identity_candidate=match.best_identity_candidate,
            best_distance=match.best_distance,
            distances=match.distances,
            threshold=match.threshold,
            threshold_status=match.threshold_status,
            embedding_dim=match.embedding_dim,
            inference_ms=ms,
            error=None,
            display_name=match.display_name,
            display_name_candidate=match.display_name_candidate,
        )

    def append_embeddings(
        self,
        identity: str,
        items: list[tuple[list[float], str, str]],
        *,
        enrollments: dict[str, Any] | None = None,
        path: Path | None = None,
    ) -> dict[str, Any]:
        """Append (embedding, source_path, sha256) without clearing prior ones. Atomic save."""
        if not items:
            raise ValueError("No embeddings to append")
        db = enrollments if enrollments is not None else load_enrollments(path)
        db = copy.deepcopy(db)
        backfill_image_fingerprints(db, identity)
        identities = db.setdefault("identities", {})
        rec = ensure_identity_record(
            identities.setdefault(identity, {}), identity_id=identity
        )
        if not (isinstance(rec.get("display_name"), str) and rec["display_name"].strip()):
            rec["display_name"] = identity
        before = len(rec.get("embeddings") or [])
        known_fps, _ = enrolled_provenance_sets(db, identity)
        for emb, source, fingerprint in items:
            if len(emb) != EMBEDDING_DIM:
                raise ValueError(f"Bad embedding dim {len(emb)} from {source}")
            if not fingerprint:
                raise ValueError(f"Missing fingerprint for {source}")
            if fingerprint in known_fps:
                raise ValueError(f"Duplicate fingerprint already enrolled: {source}")
            rec["embeddings"].append(emb)
            rec["source_images"].append(source)
            rec["image_fingerprints"].append(fingerprint)
            known_fps.add(fingerprint)
        save_enrollments(db, path)
        after = len(rec["embeddings"])
        print(
            f"[OK] Appended {after - before} embedding(s) to {identity!r}: "
            f"{before} -> {after} total"
        )
        return db

    def enroll_images(
        self,
        identity: str,
        image_paths: list[Path],
        *,
        enrollments: dict[str, Any] | None = None,
        path: Path | None = None,
        skip_near_duplicates: bool = False,
        duplicate_distance: float = NEAR_DUPLICATE_DISTANCE,
    ) -> dict[str, Any]:
        """Embed full photos (enforce_detection=True) and **append** to JSON DB."""
        db = enrollments if enrollments is not None else load_enrollments(path)
        db = copy.deepcopy(db)
        backfill_image_fingerprints(db, identity)
        known_fps, known_paths = enrolled_provenance_sets(db, identity)
        existing = list(
            ((db.get("identities") or {}).get(identity) or {}).get("embeddings") or []
        )
        pending: list[tuple[list[float], str, str]] = []
        session_embeds: list[list[float]] = list(existing)
        failed: list[str] = []
        skipped_dup: list[str] = []
        skipped_provenanced: list[str] = []

        for img_path in image_paths:
            fp = file_sha256(img_path)
            path_key = normalize_path_key(img_path)
            if fp in known_fps or path_key in known_paths:
                skipped_provenanced.append(img_path.name)
                continue
            result_emb, _, err = self._embed_path(img_path)
            if result_emb is None:
                failed.append(f"{img_path.name}: {err}")
                continue
            near, dist = is_near_duplicate(
                result_emb, session_embeds, duplicate_distance=duplicate_distance
            )
            if near and skip_near_duplicates:
                skipped_dup.append(f"{img_path.name}: dist={dist:.3f}")
                continue
            pending.append((result_emb, str(img_path.resolve()), fp))
            session_embeds.append(result_emb)
            known_fps.add(fp)

        if skipped_provenanced:
            print("[INFO] Skipped already-enrolled (fingerprint/path):")
            for line in skipped_provenanced:
                print(f"  - {line}")
        if failed:
            print("[WARN] Embed failures:")
            for line in failed:
                print(f"  - {line}")
        if skipped_dup:
            print("[INFO] Skipped near-duplicates:")
            for line in skipped_dup:
                print(f"  - {line}")
        if not pending:
            raise RuntimeError(
                "Enrollment produced no new embeddings "
                f"(failed={len(failed)}, skipped_dup={len(skipped_dup)}, "
                f"already_enrolled={len(skipped_provenanced)})"
            )
        return self.append_embeddings(
            identity, pending, enrollments=db, path=path
        )

    def _embed_path(self, img_path: Path) -> tuple[list[float] | None, float, str | None]:
        t0 = time.perf_counter()
        try:
            from deepface import DeepFace  # noqa: WPS433

            self.warm_up()
            result = DeepFace.represent(
                img_path=str(img_path),
                model_name=MODEL_NAME,
                enforce_detection=True,
            )
            emb = list(result[0]["embedding"])
            if len(emb) != EMBEDDING_DIM:
                return None, (time.perf_counter() - t0) * 1000.0, (
                    f"unexpected_embedding_dim={len(emb)}"
                )
            return emb, (time.perf_counter() - t0) * 1000.0, None
        except Exception as exc:  # noqa: BLE001
            return None, (time.perf_counter() - t0) * 1000.0, f"{type(exc).__name__}: {exc}"


def match_identity(
    embedding: list[float],
    enrollments: dict[str, Any],
    *,
    threshold: float = MATCH_THRESHOLD,
) -> IdentityMatch:
    best_id: str | None = None
    best_dist = float("inf")
    all_distances: dict[str, float] = {}
    identities = enrollments.get("identities", {})

    for ident_id, rec in identities.items():
        embeds = rec.get("embeddings") or []
        if not embeds:
            continue
        dists = [euclidean_distance(embedding, emb) for emb in embeds]
        d = float(min(dists))
        all_distances[ident_id] = d
        if d < best_dist:
            best_dist = d
            best_id = ident_id

    recognized = best_id is not None and best_dist < threshold
    disp = None
    disp_cand = None
    if best_id is not None:
        disp_cand = display_name_of(identities.get(best_id) or {}, best_id)
        if recognized:
            disp = disp_cand
    return IdentityMatch(
        recognized=bool(recognized),
        identity=best_id if recognized else None,
        best_identity_candidate=best_id,
        best_distance=None if best_id is None else float(best_dist),
        distances=all_distances,
        threshold=threshold,
        threshold_status=MATCH_THRESHOLD_STATUS,
        embedding_dim=len(embedding),
        inference_ms=0.0,
        error=None,
        display_name=disp,
        display_name_candidate=disp_cand,
    )
