"""Parity: vectorized enrollment index matches naive Euclidean matching."""

from __future__ import annotations

import unittest

import numpy as np

from facenet_identity import (
    MATCH_THRESHOLD,
    build_enrollment_index,
    euclidean_distance,
    match_identity,
)


class EnrollmentIndexParityTests(unittest.TestCase):
    def test_vectorized_matches_naive_min_distance(self) -> None:
        rng = np.random.default_rng(42)
        seif = [rng.normal(size=128).astype(np.float64).tolist() for _ in range(30)]
        other = [rng.normal(size=128).astype(np.float64).tolist() for _ in range(5)]
        db = {
            "identities": {
                "Seif": {"embeddings": seif, "display_name": "Seif"},
                "Other": {"embeddings": other, "display_name": "Other"},
            }
        }
        query = (np.asarray(seif[3], dtype=np.float64) + rng.normal(0, 0.01, size=128)).tolist()

        # Naive reference
        best_id = None
        best_d = float("inf")
        for ident_id, rec in db["identities"].items():
            d = min(euclidean_distance(query, e) for e in rec["embeddings"])
            if d < best_d:
                best_d = d
                best_id = ident_id

        index = build_enrollment_index(db)
        indexed = match_identity(query, db, threshold=MATCH_THRESHOLD, enrollment_index=index)
        naive = match_identity(query, db, threshold=MATCH_THRESHOLD, enrollment_index=None)

        self.assertEqual(indexed.best_identity_candidate, best_id)
        self.assertEqual(naive.best_identity_candidate, best_id)
        self.assertEqual(indexed.recognized, naive.recognized)
        self.assertAlmostEqual(indexed.best_distance or 0.0, best_d, places=9)
        self.assertAlmostEqual(naive.best_distance or 0.0, best_d, places=9)
        self.assertAlmostEqual(
            indexed.best_distance or 0.0, naive.best_distance or 0.0, places=9
        )


if __name__ == "__main__":
    unittest.main()
