"""Generate and persist the 256-bit HMAC secret key."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import secrets

KEY_SIZE = 32
DEFAULT_KEY_PATH = Path(__file__).resolve().parent / "secrets" / "hmac.key"


def generate_key(key_path: Path, force: bool = False) -> int:
    """
    Write a new 32-byte key to key_path.
    Returns 0 on success, 1 if key exists and was not overwritten.
    """
    key_path = key_path.resolve()
    key_path.parent.mkdir(parents=True, exist_ok=True)

    if key_path.exists() and not force:
        print("[INFO] Key already exists. Not overwriting it.")
        return 1

    if key_path.exists() and force:
        key_path.unlink()

    raw = secrets.token_bytes(KEY_SIZE)
    key_path.write_bytes(raw)

    if key_path.stat().st_size != KEY_SIZE:
        print("[ERROR] Key file size verification failed.", file=sys.stderr)
        return 2

    print(f"[OK] Generated 256-bit HMAC key at {key_path}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate VIGIL HMAC secret key")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing key file",
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_KEY_PATH,
        help="Output path for the key file",
    )
    args = parser.parse_args()
    sys.exit(generate_key(args.path, force=args.force))


if __name__ == "__main__":
    main()
