"""One-shot identity key rename (no re-embedding). Atomic DB update."""

from __future__ import annotations

import argparse
import sys

from facenet_identity import DEFAULT_ENROLLMENT_PATH, migrate_identity_label


def main() -> int:
    parser = argparse.ArgumentParser(description="Rename enrollment identity key")
    parser.add_argument("--from-id", required=True)
    parser.add_argument("--to-id", required=True)
    parser.add_argument("--display-name", default=None)
    parser.add_argument("--enrollment-db", type=str, default=str(DEFAULT_ENROLLMENT_PATH))
    args = parser.parse_args()
    try:
        summary = migrate_identity_label(
            old_id=args.from_id,
            new_id=args.to_id,
            display_name=args.display_name or args.to_id,
            path=__import__("pathlib").Path(args.enrollment_db),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(
        f"[OK] Renamed {summary['old_id']!r} -> {summary['new_id']!r} "
        f"display_name={summary['display_name']!r} "
        f"embeddings={summary['embeddings']} path={summary['path']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
