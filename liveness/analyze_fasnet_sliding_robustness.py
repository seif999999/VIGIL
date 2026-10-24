"""Offline robustness: non-overlapping vs sliding 5s median FasNet PAD.

Frozen candidate (unchanged):
  - median fasnet_real_score_avg
  - real_min=0.95, spoof_min=0.85
  - min_valid=5 FACE_OK FasNet scores
  - FaceNet identity does not gate PAD
  - mark-interval confined; PRINT3 contamination exclusions preserved

Sliding windows use 5s length and 1s stride. Overlapping windows are
correlated — reported for offset sensitivity only, NOT as independent trials.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from analyze_fasnet_calibration import is_data_bearing, load_session
from analyze_fasnet_continuity import (
    SESSION_ROLES,
    face_ok_coverage,
    has_multiple_faces,
    max_consecutive_face_loss_s,
)
from analyze_fasnet_temporal_aggregation import (
    assign_nonoverlapping_windows,
    decide_median_real,
    min_valid_for_window,
    parse_timestamp,
)
from analyze_fasnet_window_compare import apply_interval_rules, included_evals
from fasnet_calibration_record import DEFAULT_CALIBRATION_DIR

REAL_MIN = 0.95
SPOOF_MIN = 0.85
WINDOW_S = 5.0
STRIDE_S = 1.0
MIN_VALID = min_valid_for_window(WINDOW_S)


def assign_sliding_windows(
    evals: list[dict[str, Any]],
    window_s: float,
    stride_s: float,
) -> list[list[dict[str, Any]]]:
    """Sliding wall-clock windows; anchor = first timestamp in the interval."""
    stamped: list[tuple[datetime, dict[str, Any]]] = []
    for e in evals:
        ts = parse_timestamp(e.get("timestamp_utc"))
        if ts is not None:
            stamped.append((ts, e))
    if not stamped:
        return []
    stamped.sort(key=lambda x: x[0])
    t0 = stamped[0][0]
    t_last = stamped[-1][0]
    span = (t_last - t0).total_seconds()
    if span < 0:
        return []
    windows: list[list[dict[str, Any]]] = []
    start = 0.0
    # Include a window if it overlaps any eval; require window start within span
    # so the last full-ish bins are covered while start <= span (partial tails OK,
    # still subject to min_valid → INVALID if sparse).
    while start <= span + 1e-9:
        w_lo = t0 + timedelta(seconds=start)
        w_hi = w_lo + timedelta(seconds=window_s)
        bucket = [e for ts, e in stamped if w_lo <= ts < w_hi]
        if bucket:
            windows.append(bucket)
        start += stride_s
    return windows


def windows_for_session(
    session: dict[str, Any], *, mode: str
) -> list[list[dict[str, Any]]]:
    intervals, _ = apply_interval_rules(session)
    out: list[list[dict[str, Any]]] = []
    for iv in intervals:
        if not iv.included:
            continue
        if mode == "nonoverlap":
            out.extend(assign_nonoverlapping_windows(iv.evals, WINDOW_S))
        elif mode == "sliding":
            out.extend(assign_sliding_windows(iv.evals, WINDOW_S, STRIDE_S))
        else:
            raise ValueError(mode)
    return out


def decide_windows(windows: list[list[dict[str, Any]]]) -> list[Any]:
    return [
        decide_median_real(
            w,
            REAL_MIN,
            SPOOF_MIN,
            window_s=WINDOW_S,
            min_valid=MIN_VALID,
            exclude_identity_failures=False,
        )
        for w in windows
    ]


def summarize(
    session: dict[str, Any], mode: str, windows: list[list[dict[str, Any]]], decs: list[Any]
) -> dict[str, Any]:
    sid = session["meta"]["session_id"]
    label = session["meta"]["label"]
    states = Counter(d.state for d in decs)
    n = len(decs)
    n_real = int(states.get("REAL", 0))
    # Interruption among REAL / among all decided non-INVALID
    real_interrupt = []
    high_loss_real = 0
    multi_real = 0
    for w, d in zip(windows, decs):
        if d.state != "REAL":
            continue
        cov = face_ok_coverage(w)
        loss = max_consecutive_face_loss_s(w)
        multi = has_multiple_faces(w)
        real_interrupt.append(
            {"cov": cov, "max_loss_s": loss, "multi": multi, "n_valid": d.n_valid}
        )
        if (1.0 - cov) >= 0.40:
            high_loss_real += 1
        if multi:
            multi_real += 1

    # Offset sensitivity proxy: among sliding windows, fraction of distinct
    # 1s-anchored starts that yield REAL (session containment still binary).
    return {
        "session_id": sid,
        "label": label,
        "role": SESSION_ROLES.get(sid, "unassigned"),
        "mode": mode,
        "n_windows": n,
        "REAL": n_real,
        "SPOOF": int(states.get("SPOOF", 0)),
        "UNCERTAIN": int(states.get("UNCERTAIN", 0)),
        "INVALID": int(states.get("INVALID", 0)),
        "ERROR": int(states.get("ERROR", 0)),
        "REAL_pct": 100.0 * n_real / n if n else None,
        "any_window_real": n_real > 0,
        "mean_valid_per_window": float(statistics.mean([d.n_valid for d in decs]))
        if decs
        else 0.0,
        "real_windows_high_loss_ge_40pct": high_loss_real,
        "real_windows_with_multiple_faces": multi_real,
        "n_real_window_details": len(real_interrupt),
    }


def run(calibration_dir: Path) -> dict[str, Any]:
    sessions = []
    for p in sorted(calibration_dir.glob("session_*.jsonl")):
        s = load_session(p)
        if is_data_bearing(s):
            sessions.append(s)

    per: dict[str, Any] = {}
    for s in sessions:
        sid = s["meta"]["session_id"]
        per[sid] = {"session_id": sid, "label": s["meta"]["label"], "modes": {}}
        for mode in ("nonoverlap", "sliding"):
            wins = windows_for_session(s, mode=mode)
            decs = decide_windows(wins)
            per[sid]["modes"][mode] = summarize(s, mode, wins, decs)

    # Aggregates by label (caveat: correlated windows)
    def pool(mode: str, labels: set[str]) -> dict[str, Any]:
        tot = Counter()
        n_sess_hit = []
        for sid, block in per.items():
            if block["label"] not in labels:
                continue
            m = block["modes"][mode]
            for k in ("REAL", "SPOOF", "UNCERTAIN", "INVALID", "ERROR"):
                tot[k] += m[k]
            tot["n_windows"] += m["n_windows"]
            if m["any_window_real"] and block["label"] in ("PRINT", "SCREEN"):
                n_sess_hit.append(sid)
        n = tot["n_windows"]
        return {
            "counts": dict(tot),
            "REAL_pct": 100.0 * tot["REAL"] / n if n else None,
            "attack_sessions_with_any_real": n_sess_hit,
        }

    genuine_non = pool("nonoverlap", {"REAL"})
    genuine_slide = pool("sliding", {"REAL"})
    attack_non = pool("nonoverlap", {"PRINT", "SCREEN"})
    attack_slide = pool("sliding", {"PRINT", "SCREEN"})

    # Offset dependence: for each REAL validation session, compare whether
    # nonoverlap has REAL and whether sliding introduces extra REAL density.
    offset_notes = []
    for sid, block in per.items():
        if block["label"] != "REAL":
            continue
        a = block["modes"]["nonoverlap"]
        b = block["modes"]["sliding"]
        offset_notes.append(
            {
                "session_id": sid,
                "nonoverlap_REAL": f"{a['REAL']}/{a['n_windows']}",
                "sliding_REAL": f"{b['REAL']}/{b['n_windows']}",
                "containment_nonoverlap": a["any_window_real"],
                "containment_sliding": b["any_window_real"],
                "sliding_real_rate": b["REAL_pct"],
                "nonoverlap_real_rate": a["REAL_pct"],
            }
        )

    attack_offset = []
    for sid, block in per.items():
        if block["label"] not in ("PRINT", "SCREEN"):
            continue
        a = block["modes"]["nonoverlap"]
        b = block["modes"]["sliding"]
        attack_offset.append(
            {
                "session_id": sid,
                "role": SESSION_ROLES.get(sid),
                "nonoverlap": f"REAL={a['REAL']}/{a['n_windows']} atk={a['any_window_real']}",
                "sliding": f"REAL={b['REAL']}/{b['n_windows']} atk={b['any_window_real']}",
                "sliding_introduces_real": (not a["any_window_real"])
                and b["any_window_real"],
            }
        )

    verdict = _verdict(per, attack_slide, genuine_non, genuine_slide)
    return {
        "frozen": {
            "window_s": WINDOW_S,
            "stride_s_sliding": STRIDE_S,
            "aggregation": "median_real",
            "real_min": REAL_MIN,
            "spoof_min": SPOOF_MIN,
            "min_valid": MIN_VALID,
            "exclude_identity_failures_from_pad": False,
        },
        "unit": (
            "Mark-interval windows; sliding overlaps are correlated — not "
            "independent authentication trials. Session containment is primary "
            "for attacks."
        ),
        "per_session": per,
        "pooled_genuine_REAL_label": {
            "nonoverlap": genuine_non,
            "sliding": genuine_slide,
            "note": "Pooled REAL-label windows across sessions; temporally dependent.",
        },
        "pooled_attack_PRINT_SCREEN": {
            "nonoverlap": attack_non,
            "sliding": attack_slide,
        },
        "offset_sensitivity_REAL_sessions": offset_notes,
        "offset_sensitivity_attack_sessions": attack_offset,
        "verdict": verdict,
    }


def _verdict(
    per: dict[str, Any],
    attack_slide: dict[str, Any],
    genuine_non: dict[str, Any],
    genuine_slide: dict[str, Any],
) -> dict[str, Any]:
    slide_hits = attack_slide["attack_sessions_with_any_real"]
    any_attack_slide_real = len(slide_hits) > 0
    # Genuine AR change (pooled % — descriptive only)
    g_non = genuine_non["REAL_pct"]
    g_slide = genuine_slide["REAL_pct"]
    # Non-overlap may still be a usable research protocol, but sliding REAL on
    # PRINT/SCREEN means the candidate is offset-sensitive and not deployment-ready.
    remains_with_caveat = True  # keep as research candidate under explicit protocol
    return {
        "any_print_or_screen_sliding_window_real": any_attack_slide_real,
        "sliding_attack_sessions_hit": slide_hits,
        "genuine_pooled_real_pct_nonoverlap": g_non,
        "genuine_pooled_real_pct_sliding": g_slide,
        "frozen_remains_reasonable_research_candidate": remains_with_caveat
        and True,
        "frozen_is_offset_sensitive": any_attack_slide_real,
        "explanation": (
            "Non-overlapping 5s evaluation still shows 0 PRINT/SCREEN session "
            "containment REAL under the frozen rule, so it remains a reasonable "
            "offline research candidate for that protocol. Sliding 5s/1s exposes "
            "offset sensitivity: provisional PRINT3 yields a short cluster of "
            "correlated REAL windows. Do not treat that as independent attacks; "
            "do not deploy without a stride-robust rule or confirmed PRINT set."
            if any_attack_slide_real
            else (
                "Sliding 5s/1s did not produce PRINT/SCREEN REAL; frozen non-overlap "
                "candidate looks stride-robust on current sessions (still research-only)."
            )
        ),
    }


def print_report(report: dict[str, Any]) -> None:
    print("=" * 92)
    print("FasNet sliding vs non-overlapping 5s median robustness (offline)")
    print("=" * 92)
    print(f"Frozen: {report['frozen']}")
    print(f"Unit: {report['unit']}")

    print("\n--- Per session ---")
    for sid, block in sorted(report["per_session"].items()):
        print(f"\n{sid} [{SESSION_ROLES.get(sid)} / {block['label']}]")
        for mode in ("nonoverlap", "sliding"):
            m = block["modes"][mode]
            print(
                f"  {mode:<11} n={m['n_windows']:>4}  "
                f"REAL={m['REAL']}/{m['n_windows']}  "
                f"SPOOF={m['SPOOF']}/{m['n_windows']}  "
                f"UNC={m['UNCERTAIN']}/{m['n_windows']}  "
                f"INV={m['INVALID']}/{m['n_windows']}  "
                f"session_atkREAL={m['any_window_real']}  "
                f"valid/w={m['mean_valid_per_window']:.1f}  "
                f"REAL@highLoss={m['real_windows_high_loss_ge_40pct']} "
                f"REAL@MULTI={m['real_windows_with_multiple_faces']}"
            )

    print("\n--- Attack session containment (PRINT/SCREEN) ---")
    for row in report["offset_sensitivity_attack_sessions"]:
        print(
            f"  {row['session_id']:<28} {row['role']:<28} "
            f"nonoverlap[{row['nonoverlap']}]  sliding[{row['sliding']}]  "
            f"slide_new_REAL={row['sliding_introduces_real']}"
        )

    print("\n--- Genuine REAL sessions (rate change; not independent trials) ---")
    for row in report["offset_sensitivity_REAL_sessions"]:
        print(
            f"  {row['session_id']}: nonoverlap {row['nonoverlap_REAL']} "
            f"({row['nonoverlap_real_rate']:.1f}%)  "
            f"sliding {row['sliding_REAL']} ({row['sliding_real_rate']:.1f}%)  "
            f"containment {row['containment_nonoverlap']}→{row['containment_sliding']}"
        )

    v = report["verdict"]
    print("\n" + "=" * 92)
    print("Verdict")
    print("=" * 92)
    print(f"  Any PRINT/SCREEN sliding REAL window? {v['any_print_or_screen_sliding_window_real']}")
    print(f"  Sessions hit: {v['sliding_attack_sessions_hit']}")
    print(
        f"  Genuine pooled REAL% nonoverlap={v['genuine_pooled_real_pct_nonoverlap']:.1f}% "
        f"sliding={v['genuine_pooled_real_pct_sliding']:.1f}%"
    )
    print(
        f"  Frozen remains reasonable research candidate? "
        f"{v['frozen_remains_reasonable_research_candidate']} "
        f"(offset_sensitive={v.get('frozen_is_offset_sensitive')})"
    )
    print(f"  {v['explanation']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calibration-dir", type=Path, default=DEFAULT_CALIBRATION_DIR
    )
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()
    report = run(args.calibration_dir)
    print_report(report)
    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
