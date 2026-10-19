"""Unit tests for VIGIL HMAC frame authentication (Stage 1)."""

from __future__ import annotations

import secrets
import unittest
from pathlib import Path

from protocol import (
    VerifyReason,
    create_frame_package,
    generate_session_id,
    load_key,
    parse_frame_package,
    serialize_header,
    FrameVerifier,
    HEADER_SIZE,
    HMAC_SIZE,
    PROTOCOL_VERSION,
)

BASE_DIR = Path(__file__).resolve().parent
KEY_PATH = BASE_DIR / "secrets" / "hmac.key"
TEST_IMAGES = BASE_DIR / "test_images"
ORIGINAL_1 = TEST_IMAGES / "original_1.jpg"
ORIGINAL_2 = TEST_IMAGES / "original_2.jpg"
FAKE_IMAGE = TEST_IMAGES / "fake_image.jpg"


def _require_images(*paths: Path) -> None:
    missing = [p.name for p in paths if not p.is_file()]
    if missing:
        raise unittest.SkipTest(
            f"Missing test image(s) in test_images/: {', '.join(missing)}"
        )


class HmacFrameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not KEY_PATH.is_file():
            raise unittest.SkipTest(
                f"Missing HMAC key at {KEY_PATH}. Run: python keygen.py"
            )
        cls.key = load_key(KEY_PATH)

    def _jpeg_original_1(self) -> bytes:
        _require_images(ORIGINAL_1)
        return ORIGINAL_1.read_bytes()

    def test_01_valid_original_frame(self) -> None:
        """VALID FRAME"""
        jpeg = self._jpeg_original_1()
        session = generate_session_id()
        package = create_frame_package(jpeg, session, 0, self.key)
        verifier = FrameVerifier(self.key, session)
        result = verifier.verify(package)
        self.assertEqual(result.reason, VerifyReason.ACCEPT)

    def test_02_modify_one_jpeg_byte(self) -> None:
        """MODIFIED JPEG"""
        jpeg = self._jpeg_original_1()
        session = generate_session_id()
        package = bytearray(
            create_frame_package(jpeg, session, 0, self.key)
        )
        if len(jpeg) == 0:
            self.skipTest("original_1.jpg is empty; need non-empty JPEG")
        package[HEADER_SIZE] ^= 0x01
        verifier = FrameVerifier(self.key, session)
        result = verifier.verify(bytes(package))
        self.assertEqual(result.reason, VerifyReason.HMAC_MISMATCH)

    def test_03_complete_image_replacement(self) -> None:
        """REPLACED IMAGE"""
        _require_images(FAKE_IMAGE)
        jpeg_orig = self._jpeg_original_1()
        jpeg_fake = FAKE_IMAGE.read_bytes()
        session = generate_session_id()
        signed = create_frame_package(jpeg_orig, session, 0, self.key)
        original_tag = signed[-HMAC_SIZE:]

        new_header = serialize_header(
            PROTOCOL_VERSION, session, 0, len(jpeg_fake)
        )
        tampered = new_header + jpeg_fake + original_tag

        verifier = FrameVerifier(self.key, session)
        result = verifier.verify(tampered)
        self.assertEqual(result.reason, VerifyReason.HMAC_MISMATCH)

    def test_04_counter_tampering(self) -> None:
        """COUNTER TAMPER"""
        jpeg = self._jpeg_original_1()
        session = generate_session_id()
        package = bytearray(
            create_frame_package(jpeg, session, 0, self.key)
        )
        # Frame counter: 8 bytes at header offset 17..24
        package[17] ^= 0xFF
        verifier = FrameVerifier(self.key, session)
        result = verifier.verify(bytes(package))
        self.assertEqual(result.reason, VerifyReason.HMAC_MISMATCH)

    def test_05_session_id_tampering(self) -> None:
        """SESSION TAMPER"""
        jpeg = self._jpeg_original_1()
        session = generate_session_id()
        package = bytearray(
            create_frame_package(jpeg, session, 0, self.key)
        )
        package[1] ^= 0x01
        verifier = FrameVerifier(self.key, session)
        result = verifier.verify(bytes(package))
        self.assertEqual(result.reason, VerifyReason.HMAC_MISMATCH)

    def test_06_fake_random_hmac(self) -> None:
        """FAKE HMAC"""
        jpeg = self._jpeg_original_1()
        session = generate_session_id()
        package = bytearray(
            create_frame_package(jpeg, session, 0, self.key)
        )
        package[-HMAC_SIZE:] = secrets.token_bytes(HMAC_SIZE)
        verifier = FrameVerifier(self.key, session)
        result = verifier.verify(bytes(package))
        self.assertEqual(result.reason, VerifyReason.HMAC_MISMATCH)

    def test_07_unsigned_image_injection(self) -> None:
        """UNSIGNED INJECTION"""
        session = generate_session_id()
        verifier = FrameVerifier(self.key, session)

        too_short = b"\x01" + b"\x00" * 10
        result_short = verifier.verify(too_short)
        self.assertEqual(result_short.reason, VerifyReason.MALFORMED_PACKAGE)

        raw_source = FAKE_IMAGE if FAKE_IMAGE.is_file() else ORIGINAL_1
        if raw_source.is_file():
            result_raw = verifier.verify(raw_source.read_bytes())
            self.assertIn(
                result_raw.reason,
                (
                    VerifyReason.MALFORMED_PACKAGE,
                    VerifyReason.UNSUPPORTED_VERSION,
                ),
            )

    def test_08_duplicate_frame(self) -> None:
        """DUPLICATE REPLAY"""
        jpeg = self._jpeg_original_1()
        session = generate_session_id()
        package = create_frame_package(jpeg, session, 0, self.key)
        verifier = FrameVerifier(self.key, session)

        first = verifier.verify(package)
        self.assertEqual(first.reason, VerifyReason.ACCEPT)

        second = verifier.verify(package)
        self.assertEqual(second.reason, VerifyReason.REPLAY_OR_DUPLICATE)

    def test_09_old_frame_replay(self) -> None:
        """OLD FRAME REPLAY"""
        jpeg = self._jpeg_original_1()
        session = generate_session_id()
        pkg0 = create_frame_package(jpeg, session, 0, self.key)
        pkg1 = create_frame_package(jpeg, session, 1, self.key)
        verifier = FrameVerifier(self.key, session)

        self.assertEqual(verifier.verify(pkg0).reason, VerifyReason.ACCEPT)
        self.assertEqual(verifier.verify(pkg1).reason, VerifyReason.ACCEPT)
        self.assertEqual(
            verifier.verify(pkg0).reason, VerifyReason.REPLAY_OR_DUPLICATE
        )

    def test_10_out_of_order(self) -> None:
        """OUT OF ORDER"""
        jpeg = self._jpeg_original_1()
        session = generate_session_id()
        pkg0 = create_frame_package(jpeg, session, 0, self.key)
        pkg1 = create_frame_package(jpeg, session, 1, self.key)
        pkg3 = create_frame_package(jpeg, session, 3, self.key)
        verifier = FrameVerifier(self.key, session)

        self.assertEqual(verifier.verify(pkg0).reason, VerifyReason.ACCEPT)
        self.assertEqual(verifier.verify(pkg1).reason, VerifyReason.ACCEPT)
        self.assertEqual(verifier.next_expected_counter, 2)

        result = verifier.verify(pkg3)
        self.assertEqual(result.reason, VerifyReason.OUT_OF_ORDER)
        self.assertEqual(verifier.next_expected_counter, 2)

    def test_11_previous_session_replay(self) -> None:
        """OLD SESSION REPLAY"""
        jpeg = self._jpeg_original_1()
        session_a = generate_session_id()
        session_b = generate_session_id()
        while session_b == session_a:
            session_b = generate_session_id()

        pkg_a = create_frame_package(jpeg, session_a, 0, self.key)
        verifier_b = FrameVerifier(self.key, session_b)

        result = verifier_b.verify(pkg_a)
        self.assertEqual(result.reason, VerifyReason.WRONG_SESSION)
        self.assertNotEqual(result.reason, VerifyReason.HMAC_MISMATCH)

    def test_12_truncated_corrupted_package(self) -> None:
        """TRUNCATED PACKAGE"""
        jpeg = self._jpeg_original_1()
        session = generate_session_id()
        package = create_frame_package(jpeg, session, 0, self.key)
        truncated = package[: len(package) // 2]

        verifier = FrameVerifier(self.key, session)
        result = verifier.verify(truncated)
        self.assertEqual(result.reason, VerifyReason.MALFORMED_PACKAGE)

        parsed, reason = parse_frame_package(truncated)
        self.assertIsNone(parsed)
        self.assertEqual(reason, VerifyReason.MALFORMED_PACKAGE)


def _print_summary(result: unittest.TestResult) -> None:
    labels = {
        "test_01_valid_original_frame": "VALID_FRAME",
        "test_02_modify_one_jpeg_byte": "MODIFIED_JPEG",
        "test_03_complete_image_replacement": "REPLACED_IMAGE",
        "test_04_counter_tampering": "COUNTER_TAMPER",
        "test_05_session_id_tampering": "SESSION_TAMPER",
        "test_06_fake_random_hmac": "FAKE_HMAC",
        "test_07_unsigned_image_injection": "UNSIGNED_INJECTION",
        "test_08_duplicate_frame": "DUPLICATE_REPLAY",
        "test_09_old_frame_replay": "OLD_FRAME_REPLAY",
        "test_10_out_of_order": "OUT_OF_ORDER",
        "test_11_previous_session_replay": "OLD_SESSION_REPLAY",
        "test_12_truncated_corrupted_package": "TRUNCATED_PACKAGE",
    }
    skipped_ids = {case.id().split(".")[-1] for case, _ in result.skipped}
    failed_ids = {
        case.id().split(".")[-1] for case, _ in result.failures + result.errors
    }
    print("\n--- Stage 1 test summary ---")
    for test_id, label in labels.items():
        if test_id in skipped_ids:
            print(f"{label:.<28} SKIP")
        elif test_id in failed_ids:
            print(f"{label:.<28} FAIL")
        else:
            print(f"{label:.<28} PASS")


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(__import__(__name__))
    runner = unittest.TextTestRunner(verbosity=2)
    outcome = runner.run(suite)
    _print_summary(outcome)
    if not outcome.wasSuccessful():
        raise SystemExit(1)
