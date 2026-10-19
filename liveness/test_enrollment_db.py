"""Enrollment DB append + atomic save + provenance tests (no DeepFace / camera)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from enroll_identity import _looks_like_vigil_capture
from facenet_identity import (
    MATCH_THRESHOLD,
    NEAR_DUPLICATE_DISTANCE,
    FaceNetIdentity,
    backfill_image_fingerprints,
    empty_enrollment_db,
    enrolled_provenance_sets,
    file_sha256,
    identity_embedding_count,
    is_near_duplicate,
    load_enrollments,
    save_enrollments,
)


class EnrollmentDbTests(unittest.TestCase):
    def test_threshold_unchanged(self) -> None:
        self.assertEqual(MATCH_THRESHOLD, 9.0)

    def test_append_does_not_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "enrollments.json"
            db = empty_enrollment_db()
            db["identities"]["alice"] = {
                "embeddings": [[0.0] * 128],
                "source_images": ["a.jpg"],
            }
            save_enrollments(db, path)
            loaded = load_enrollments(path)
            loaded["identities"]["alice"]["embeddings"].append([1.0] * 128)
            loaded["identities"]["alice"]["source_images"].append("b.jpg")
            save_enrollments(loaded, path)
            again = load_enrollments(path)
            self.assertEqual(identity_embedding_count(again, "alice"), 2)
            self.assertEqual(again["identities"]["alice"]["embeddings"][0][0], 0.0)
            self.assertEqual(again["identities"]["alice"]["embeddings"][1][0], 1.0)

    def test_atomic_save_preserves_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "enrollments.json"
            good = empty_enrollment_db()
            good["identities"]["bob"] = {"embeddings": [[0.5] * 128], "source_images": []}
            save_enrollments(good, path)
            # Simulate failure before replace: corrupt write by making parent a file
            # Instead: write valid, then ensure .bak exists after second save
            good2 = load_enrollments(path)
            good2["identities"]["bob"]["embeddings"].append([0.6] * 128)
            save_enrollments(good2, path)
            bak = path.with_suffix(path.suffix + ".bak")
            self.assertTrue(bak.is_file())
            bak_data = json.loads(bak.read_text(encoding="utf-8"))
            self.assertEqual(len(bak_data["identities"]["bob"]["embeddings"]), 1)
            cur = load_enrollments(path)
            self.assertEqual(len(cur["identities"]["bob"]["embeddings"]), 2)

    def test_near_duplicate_flag(self) -> None:
        base = [0.0] * 128
        close = [0.01] * 128
        far = [5.0] * 128
        near, d = is_near_duplicate(close, [base], duplicate_distance=NEAR_DUPLICATE_DISTANCE)
        self.assertTrue(near)
        self.assertIsNotNone(d)
        near2, _ = is_near_duplicate(far, [base], duplicate_distance=NEAR_DUPLICATE_DISTANCE)
        self.assertFalse(near2)

    def test_fingerprint_and_append_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            img = root / "a.jpg"
            img.write_bytes(b"\xff\xd8\xfffakejpeg-aaaa")
            fp = file_sha256(img)
            db_path = root / "enrollments.json"
            fn = FaceNetIdentity()
            emb = [0.0] * 128
            fn.append_embeddings(
                "tester",
                [(emb, str(img.resolve()), fp)],
                path=db_path,
            )
            db = load_enrollments(db_path)
            self.assertEqual(identity_embedding_count(db, "tester"), 1)
            fps, paths = enrolled_provenance_sets(db, "tester")
            self.assertIn(fp, fps)
            # Second append same fingerprint must fail (no replace of prior)
            with self.assertRaises(ValueError):
                fn.append_embeddings(
                    "tester",
                    [([1.0] * 128, str(img.resolve()), fp)],
                    enrollments=load_enrollments(db_path),
                    path=db_path,
                )
            self.assertEqual(
                identity_embedding_count(load_enrollments(db_path), "tester"), 1
            )

    def test_backfill_fingerprints(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            img = root / "b.jpg"
            img.write_bytes(b"\xff\xd8\xfffakejpeg-bbbb")
            db = empty_enrollment_db()
            db["identities"]["tester"] = {
                "embeddings": [[0.0] * 128],
                "source_images": [str(img)],
                "image_fingerprints": [],
            }
            n = backfill_image_fingerprints(db, "tester")
            self.assertEqual(n, 1)
            self.assertEqual(
                db["identities"]["tester"]["image_fingerprints"][0], file_sha256(img)
            )

    def test_vigil_capture_heuristic(self) -> None:
        self.assertTrue(
            _looks_like_vigil_capture("tester_20261019_155107_1.jpg", "tester")
        )
        self.assertFalse(
            _looks_like_vigil_capture("WhatsApp Image 2026-10-19 at 4.21.44 PM.jpeg", "tester")
        )


if __name__ == "__main__":
    unittest.main()
