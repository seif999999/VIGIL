"""Offline comparison: argmax-gated vs combined-mass FasNet decision rules.

Research only. Does not modify live authentication, Challenge 1, or recordings.
Uses marked JSONL evaluations and the same session-level splits as
analyze_fasnet_calibration.py.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal

from analyze_fasnet_calibration import (
    DEFAULT_CANDIDATES,
    is_data_bearing,
    load_session,
    marked_evals,
    shadow_state_for_eval,
    split_data_bearing_sessions,
)
from fasnet_calibration_record import DEFAULT_CALIBRATION_DIR, VALID_LABELS
from fasnet_decision_policy import RESEARCH_CANDIDATE_REAL_MIN, RESEARCH_CANDIDATE_SPOOF_MIN
from fasnet_liveness import FasNetClass

DecisionState = Literal["REAL", "SPOOF", "UNCERTAIN", "ERROR"]

# Softmax masses should sum ~1 after /2 averaging; tolerate float noise.
_MASS_SUM_TOL = 0.05

COMPARE_CANDIDATES = (0.50, 0.70, 0.80, 0.85, 0.90, 0.95, 0.97, 0.99)


def approach_a_argmax_gated(
    ev: dict[str, Any],
    real_min: float,
    spoof_min: float,
) -> DecisionState:
    """Current shadow semantics: native argmax + winning-class score gates."""
    return shadow_state_for_eval(ev, real_min, spoof_min)  # type: ignore[return-value]


def approach_b_combined_mass(
    ev: dict[str, Any],
    real_min: float,
    spoof_min: float,
    *,
    mass_sum_tol: float = _MASS_SUM_TOL,
) -> DecisionState:
    """
    Binary combined-mass rule (offline research):

      REAL_score  = class_scores_avg[1]  (or fasnet_real_score_avg)
      SPOOF_score = class_scores_avg[0]+[2] (or fasnet_spoof_score_combined)

    - Missing / unusable scores → ERROR
    - Numerically inconsistent masses (|r+s-1| > tol) → UNCERTAIN
    - Both REAL and SPOOF gates pass → UNCERTAIN (no arbitrary tie-break)
    - Else REAL if REAL_score >= real_min
    - Else SPOOF if SPOOF_score >= spoof_min
    - Else UNCERTAIN

    Does not use native argmax. Scores are uncalibrated ensemble masses.
    """
    real_s = ev.get("fasnet_real_score_avg")
    spoof_s = ev.get("fasnet_spoof_score_combined")
    avg = ev.get("fasnet_class_scores_avg")

    if real_s is None and isinstance(avg, (list, tuple)) and len(avg) == 3:
        try:
            real_s = float(avg[1])
        except (TypeError, ValueError):
            real_s = None
    if spoof_s is None and isinstance(avg, (list, tuple)) and len(avg) == 3:
        try:
            spoof_s = float(avg[0]) + float(avg[2])
        except (TypeError, ValueError):
            spoof_s = None

    if real_s is None or spoof_s is None:
        # Match Approach A: no usable FasNet output → ERROR
        if ev.get("fasnet_native_class") is None and ev.get("fasnet_winning_score") is None:
            return "ERROR"
        return "ERROR"

    try:
        r = float(real_s)
        s = float(spoof_s)
    except (TypeError, ValueError):
        return "ERROR"

    if not (math.isfinite(r) and math.isfinite(s)):
        return "ERROR"
    if r < -1e-6 or s < -1e-6 or r > 1.0 + 1e-3 or s > 1.0 + 1e-3:
        return "UNCERTAIN"

    if abs((r + s) - 1.0) > mass_sum_tol:
        # Masses should partition the simplex; refuse confident claims.
        return "UNCERTAIN"

    real_pass = r >= real_min
    spoof_pass = s >= spoof_min
    if real_pass and spoof_pass:
        # Overlapping region — do not pick a winner arbitrarily.
        return "UNCERTAIN"
    if real_pass:
        return "REAL"
    if spoof_pass:
        return "SPOOF"
    return "UNCERTAIN"


def _rate(n: int, d: int) -> dict[str, Any]:
    if d <= 0:
        return {
            "pct": None,
            "n": n,
            "d": d,
            "display": "N/A",
            "note": "denominator is zero",
        }
    pct = 100.0 * n / d
    return {"pct": pct, "n": n, "d": d, "display": f"{pct:.1f}%"}


def _pool_frames(
    sessions: list[dict[str, Any]], label: str
) -> list[dict[str, Any]]:
    return [
        e
        for s in sessions
        if s["meta"]["label"] == label
        for e in marked_evals(s)
    ]


def summarize_rule(
    frames: list[dict[str, Any]],
    *,
    label: str,
    classify,
    real_min: float,
    spoof_min: float,
    n_sessions: int,
) -> dict[str, Any]:
    states = Counter(classify(e, real_min, spoof_min) for e in frames)
    n = len(frames)
    n_err = int(states.get("ERROR", 0))
    n_cls = n - n_err
    n_real = int(states.get("REAL", 0))
    n_spoof = int(states.get("SPOOF", 0))
    n_unc = int(states.get("UNCERTAIN", 0))
    native_real = sum(1 for e in frames if e.get("fasnet_native_class") == "REAL")
    live_accept = sum(1 for e in frames if e.get("auth_decision") == "ACCEPT")

    out: dict[str, Any] = {
        "n_sessions": n_sessions,
        "n_marked_evals": n,
        "n_classifiable": n_cls,
        "counts": {
            "REAL": n_real,
            "SPOOF": n_spoof,
            "UNCERTAIN": n_unc,
            "ERROR": n_err,
        },
        "native_REAL_count": native_real,
        "live_ACCEPT_count": live_accept,
        "REAL_rate_all": _rate(n_real, n),
        "SPOOF_rate_all": _rate(n_spoof, n),
        "UNCERTAIN_rate_all": _rate(n_unc, n),
        "ERROR_rate_all": _rate(n_err, n),
        "REAL_rate_classifiable": _rate(n_real, n_cls),
        "SPOOF_rate_classifiable": _rate(n_spoof, n_cls),
        "UNCERTAIN_rate_classifiable": _rate(n_unc, n_cls),
        "native_REAL_rate_all": _rate(native_real, n),
        "live_ACCEPT_rate_all": _rate(live_accept, n),
    }
    if label == "REAL":
        out["genuine_REAL_accept_shadow"] = _rate(n_real, n)
        out["genuine_REAL_reject_or_uncertain"] = _rate(n - n_real, n)
        out["genuine_UNCERTAIN"] = _rate(n_unc, n)
    if label in ("PRINT", "SCREEN"):
        out["false_REAL_all"] = _rate(n_real, n)
        out["false_REAL_classifiable"] = _rate(n_real, n_cls)
        out["false_ACCEPT_live"] = _rate(live_accept, n)
    return out


def per_session_breakdown(
    sessions: list[dict[str, Any]],
    *,
    classify,
    real_min: float,
    spoof_min: float,
) -> list[dict[str, Any]]:
    rows = []
    for s in sessions:
        if not is_data_bearing(s):
            continue
        frames = marked_evals(s)
        states = Counter(classify(e, real_min, spoof_min) for e in frames)
        rows.append(
            {
                "session_id": s["meta"]["session_id"],
                "label": s["meta"]["label"],
                "n_marked": len(frames),
                "counts": dict(states),
                "live_ACCEPT": sum(
                    1 for e in frames if e.get("auth_decision") == "ACCEPT"
                ),
                "native_REAL": sum(
                    1 for e in frames if e.get("fasnet_native_class") == "REAL"
                ),
            }
        )
    rows.sort(key=lambda r: (r["label"], r["session_id"]))
    return rows


def compare_approaches(
    sessions: list[dict[str, Any]],
    *,
    candidates: tuple[float, ...] = COMPARE_CANDIDATES,
    selection_fraction: float = 0.5,
) -> dict[str, Any]:
    empty = [s for s in sessions if not is_data_bearing(s)]
    bearing = [s for s in sessions if is_data_bearing(s)]
    sel, val, split = split_data_bearing_sessions(
        bearing, selection_fraction=selection_fraction
    )

    # Consistency check sample: when scores present, r+s ≈ 1 and
    # winning matches the native class mass.
    consistency = {"checked": 0, "sum_ok": 0, "win_matches_native_mass": 0}
    for s in bearing:
        for e in marked_evals(s):
            r, sp = e.get("fasnet_real_score_avg"), e.get("fasnet_spoof_score_combined")
            w, nat = e.get("fasnet_winning_score"), e.get("fasnet_native_class")
            if r is None or sp is None:
                continue
            consistency["checked"] += 1
            if abs(float(r) + float(sp) - 1.0) <= _MASS_SUM_TOL:
                consistency["sum_ok"] += 1
            if w is not None and nat == "REAL" and abs(float(w) - float(r)) < 1e-5:
                consistency["win_matches_native_mass"] += 1
            if w is not None and nat == "SPOOF":
                # Winning score is max single non-real class, not the sum.
                avg = e.get("fasnet_class_scores_avg")
                if isinstance(avg, (list, tuple)) and len(avg) == 3:
                    win_expected = max(float(avg[0]), float(avg[2]))
                    if abs(float(w) - win_expected) < 1e-4:
                        consistency["win_matches_native_mass"] += 1

    grids: list[dict[str, Any]] = []
    for thr in candidates:
        for approach, classify in (
            ("A_argmax_gated", approach_a_argmax_gated),
            ("B_combined_mass", approach_b_combined_mass),
        ):
            entry: dict[str, Any] = {
                "approach": approach,
                "thresholds": {"real_min": thr, "spoof_min": thr},
                "selection": {},
                "validation": {},
            }
            for pool_name, pool in (("selection", sel), ("validation", val)):
                by_label = {}
                for label in VALID_LABELS:
                    frames = _pool_frames(pool, label)
                    n_sess = sum(1 for s in pool if s["meta"]["label"] == label)
                    by_label[label] = summarize_rule(
                        frames,
                        label=label,
                        classify=classify,
                        real_min=thr,
                        spoof_min=thr,
                        n_sessions=n_sess,
                    )
                entry[pool_name] = by_label
            grids.append(entry)

    # Asymmetric research pairs (symmetric real/spoof thr often conflict for B when thr<=0.5)
    asymmetric: list[dict[str, Any]] = []
    for real_min in (0.85, 0.90, 0.95, 0.97):
        for spoof_min in (0.85, 0.90, 0.95, 0.97):
            for approach, classify in (
                ("A_argmax_gated", approach_a_argmax_gated),
                ("B_combined_mass", approach_b_combined_mass),
            ):
                row: dict[str, Any] = {
                    "approach": approach,
                    "thresholds": {"real_min": real_min, "spoof_min": spoof_min},
                    "selection": {},
                    "validation": {},
                }
                for pool_name, pool in (("selection", sel), ("validation", val)):
                    by_label = {}
                    for label in VALID_LABELS:
                        frames = _pool_frames(pool, label)
                        n_sess = sum(1 for s in pool if s["meta"]["label"] == label)
                        by_label[label] = summarize_rule(
                            frames,
                            label=label,
                            classify=classify,
                            real_min=real_min,
                            spoof_min=spoof_min,
                            n_sessions=n_sess,
                        )
                    row[pool_name] = by_label
                asymmetric.append(row)

    ref_a = per_session_breakdown(
        bearing,
        classify=approach_a_argmax_gated,
        real_min=RESEARCH_CANDIDATE_REAL_MIN,
        spoof_min=RESEARCH_CANDIDATE_SPOOF_MIN,
    )
    ref_b = per_session_breakdown(
        bearing,
        classify=approach_b_combined_mass,
        real_min=RESEARCH_CANDIDATE_REAL_MIN,
        spoof_min=RESEARCH_CANDIDATE_SPOOF_MIN,
    )

    return {
        "purpose": (
            "Offline comparison of FasNet decision rules A (argmax-gated) vs "
            "B (combined-mass). Research only; no live policy change."
        ),
        "score_semantics": {
            "ensemble": (
                "MiniFASNetV2 + MiniFASNetV1SE softmax sum → prediction[3]; "
                "class_scores_avg[i]=prediction[i]/2; index1=REAL; 0+2=non-real"
            ),
            "winning_score": (
                "prediction[argmax]/2 — confidence of the single winning class "
                "(for SPOOF winners this is max(class0,class2), NOT their sum)"
            ),
            "approach_A": "native argmax + winning_score gates (current shadow)",
            "approach_B": (
                "REAL_mass=avg[1], SPOOF_mass=avg[0]+avg[2]; both-pass→UNCERTAIN; "
                "no argmax; uncalibrated masses"
            ),
            "not_calibrated_probabilities": True,
        },
        "consistency_checks": consistency,
        "n_session_files": len(sessions),
        "n_empty_excluded": len(empty),
        "n_data_bearing": len(bearing),
        "split": split,
        "symmetric_grids": grids,
        "asymmetric_grids": asymmetric,
        "per_session_A_0.85": ref_a,
        "per_session_B_0.85": ref_b,
        "recommendation_placeholder": (
            "See printed report; do not activate either rule without more sessions."
        ),
    }


def _fmt(rate: dict[str, Any] | None) -> str:
    if not rate or rate.get("pct") is None:
        return "N/A"
    return f"{rate['pct']:.1f}%({rate['n']}/{rate['d']})"


def print_comparison(report: dict[str, Any]) -> None:
    print("=== FasNet decision-rule comparison (offline research) ===")
    print(report["purpose"])
    print("Semantics:", report["score_semantics"]["winning_score"])
    print("Approach A:", report["score_semantics"]["approach_A"])
    print("Approach B:", report["score_semantics"]["approach_B"])
    c = report["consistency_checks"]
    print(
        f"Consistency: checked={c['checked']} sum≈1 ok={c['sum_ok']} "
        f"win↔native_mass ok={c['win_matches_native_mass']}"
    )
    print(
        f"Files={report['n_session_files']} data_bearing={report['n_data_bearing']} "
        f"empty_excluded={report['n_empty_excluded']}"
    )
    print("Split:", json.dumps(report["split"]["by_label"], indent=2))

    print("\n-- Per-session @ 0.85/0.85 --")
    print("Approach A:")
    for r in report["per_session_A_0.85"]:
        print(
            f"  {r['session_id']} {r['label']} n={r['n_marked']} "
            f"{r['counts']} liveA={r['live_ACCEPT']} natR={r['native_REAL']}"
        )
    print("Approach B:")
    for r in report["per_session_B_0.85"]:
        print(
            f"  {r['session_id']} {r['label']} n={r['n_marked']} "
            f"{r['counts']} liveA={r['live_ACCEPT']} natR={r['native_REAL']}"
        )

    print("\n-- Symmetric thresholds (selection | validation) --")
    print(
        "thr approach | REAL shR / unc (all) | PRINT fR (all|cls) liveFA | "
        "SCREEN fR (all) liveFA"
    )
    # Group by thr
    by_thr: dict[float, list] = defaultdict(list)
    for g in report["symmetric_grids"]:
        by_thr[g["thresholds"]["real_min"]].append(g)
    for thr in sorted(by_thr):
        for g in by_thr[thr]:
            ap = "A" if g["approach"].startswith("A") else "B"
            for pool in ("selection", "validation"):
                real = g[pool]["REAL"]
                pr = g[pool]["PRINT"]
                sc = g[pool]["SCREEN"]
                print(
                    f"{thr:.2f} {ap} [{pool[:3]}] "
                    f"REAL n={real['n_marked_evals']} "
                    f"R={_fmt(real['REAL_rate_all'])} "
                    f"U={_fmt(real['UNCERTAIN_rate_all'])} "
                    f"E={_fmt(real['ERROR_rate_all'])} "
                    f"liveA={_fmt(real['live_ACCEPT_rate_all'])} | "
                    f"PRINT fR={_fmt(pr.get('false_REAL_all'))}/"
                    f"{_fmt(pr.get('false_REAL_classifiable'))} "
                    f"liveFA={_fmt(pr.get('false_ACCEPT_live'))} "
                    f"S={_fmt(pr['SPOOF_rate_all'])} U={_fmt(pr['UNCERTAIN_rate_all'])} | "
                    f"SCREEN fR={_fmt(sc.get('false_REAL_all'))} "
                    f"liveFA={_fmt(sc.get('false_ACCEPT_live'))} "
                    f"S={_fmt(sc['SPOOF_rate_all'])}"
                )

    print("\n-- Key asymmetric @ selection (0.85–0.97) --")
    for g in report["asymmetric_grids"]:
        rm = g["thresholds"]["real_min"]
        sm = g["thresholds"]["spoof_min"]
        if rm != sm and not (rm >= 0.90 and sm >= 0.85):
            continue
        if g["approach"].startswith("A") and rm == sm:
            continue
        ap = "A" if g["approach"].startswith("A") else "B"
        real = g["selection"]["REAL"]
        pr = g["selection"]["PRINT"]
        sc = g["selection"]["SCREEN"]
        print(
            f"{ap} real>={rm:.2f} spoof>={sm:.2f} SEL: "
            f"genuineR={_fmt(real['REAL_rate_all'])} "
            f"PRINT_fR={_fmt(pr.get('false_REAL_all'))} "
            f"SCREEN_fR={_fmt(sc.get('false_REAL_all'))} "
            f"PRINT_S={_fmt(pr['SPOOF_rate_all'])} PRINT_U={_fmt(pr['UNCERTAIN_rate_all'])}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare FasNet argmax-gated vs combined-mass rules (offline)"
    )
    parser.add_argument("--dir", type=Path, default=DEFAULT_CALIBRATION_DIR)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--selection-fraction", type=float, default=0.5)
    args = parser.parse_args()

    paths = sorted(args.dir.glob("session_*.jsonl")) if args.dir.is_dir() else []
    sessions = [load_session(p) for p in paths]
    report = compare_approaches(
        sessions, selection_fraction=args.selection_fraction
    )
    print_comparison(report)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[OK] Wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
