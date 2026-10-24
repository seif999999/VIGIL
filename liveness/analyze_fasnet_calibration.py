"""Research-only analysis of labeled FasNet calibration JSONL sessions.

Uses ONLY evaluations written during explicitly MARKED intervals (recorder
already gates on M pressed). Empty session files are inventoried but excluded
from selection/validation splits so they cannot distort rates.

Does NOT activate thresholds or claim production calibration.
Consecutive frames within a session are temporally dependent — session-level
splits only; frames are never split across selection and validation.

If too few data-bearing sessions exist for a held-out validation split,
reports insufficiency clearly (N/A when denominators are zero).
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from fasnet_calibration_record import DEFAULT_CALIBRATION_DIR, VALID_LABELS
from fasnet_decision_policy import (
    RESEARCH_CANDIDATE_REAL_MIN,
    RESEARCH_CANDIDATE_SPOOF_MIN,
    FasNetConfidencePolicy,
    FasNetLivenessState,
    FasNetPolicyMode,
    POLICY_STATUS_RESEARCH_CANDIDATE,
    RESEARCH_CANDIDATE_NOTE,
    classify_fasnet_winning_score,
)
from fasnet_liveness import FasNetClass

DEFAULT_CANDIDATES = (0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)

# Shadow DIAGNOSTIC prediction vs live enforced policy (default shadow mode uses
# native argmax for auth decisions; recorded auth_decision reflects live policy).
LIVE_POLICY_NOTE = (
    "Live authentication (default shadow mode) enforces native FasNet argmax "
    "for liveness, not the research shadow confidence gates. "
    "shadow_* metrics are diagnostic only; auth_decision / live_ACCEPT_* reflect "
    "the policy that actually ran during recording."
)


def load_session(path: Path) -> dict[str, Any]:
    meta: dict[str, Any] | None = None
    evals: list[dict[str, Any]] = []
    mark_events: list[dict[str, Any]] = []
    end: dict[str, Any] | None = None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            t = obj.get("type")
            if t == "session_meta":
                meta = obj
            elif t == "eval":
                evals.append(obj)
            elif t == "mark_event":
                mark_events.append(obj)
            elif t == "session_end":
                end = obj
    if meta is None:
        raise ValueError(f"Missing session_meta in {path}")
    return {
        "path": str(path),
        "meta": meta,
        "evals": evals,
        "mark_events": mark_events,
        "end": end,
    }


def marked_evals(session: dict[str, Any]) -> list[dict[str, Any]]:
    """Return only marked-interval evaluations (calibration source of truth)."""
    out: list[dict[str, Any]] = []
    for e in session.get("evals") or []:
        # Recorder always writes marked=True for saved evals; treat missing as
        # marked for forward-compat only when type==eval (never invent unmarked).
        if e.get("type", "eval") != "eval":
            continue
        if e.get("marked") is False:
            continue
        out.append(e)
    return out


def is_data_bearing(session: dict[str, Any]) -> bool:
    return len(marked_evals(session)) > 0


def _rate(numerator: int, denominator: int, *, what: str) -> dict[str, Any]:
    """Percentage with explicit N/A when denominator is zero (never fake 0%)."""
    if denominator <= 0:
        return {
            "pct": None,
            "n": int(numerator),
            "d": int(denominator),
            "display": "N/A",
            "note": f"N/A: {what} denominator is zero (no usable samples).",
        }
    pct = 100.0 * float(numerator) / float(denominator)
    return {
        "pct": pct,
        "n": int(numerator),
        "d": int(denominator),
        "display": f"{pct:.1f}%",
        "note": f"{numerator}/{denominator} ({what})",
    }


def _quantiles(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    vs = sorted(values)

    def q(p: float) -> float:
        if len(vs) == 1:
            return vs[0]
        idx = p * (len(vs) - 1)
        lo = int(math.floor(idx))
        hi = int(math.ceil(idx))
        if lo == hi:
            return vs[lo]
        return vs[lo] * (hi - idx) + vs[hi] * (idx - lo)

    return {
        "n_frames": float(len(vs)),
        "min": vs[0],
        "p25": q(0.25),
        "median": statistics.median(vs),
        "p75": q(0.75),
        "max": vs[-1],
        "mean": statistics.mean(vs),
    }


def shadow_state_for_eval(
    ev: dict[str, Any],
    real_min: float,
    spoof_min: float,
) -> str:
    native = ev.get("fasnet_native_class")
    score = ev.get("fasnet_winning_score")
    if native is None:
        return FasNetLivenessState.ERROR.value
    try:
        cls = FasNetClass(native)
    except ValueError:
        return FasNetLivenessState.ERROR.value
    policy = FasNetConfidencePolicy(
        confident_real_min=real_min,
        confident_spoof_min=spoof_min,
        status=POLICY_STATUS_RESEARCH_CANDIDATE,
        mode=FasNetPolicyMode.SHADOW,
        note=RESEARCH_CANDIDATE_NOTE,
    )
    return classify_fasnet_winning_score(cls, score, policy).value


def _session_summary(s: dict[str, Any]) -> dict[str, Any]:
    evals = marked_evals(s)
    wins = [
        float(e["fasnet_winning_score"])
        for e in evals
        if e.get("fasnet_winning_score") is not None
    ]
    reals = [
        float(e["fasnet_real_score_avg"])
        for e in evals
        if e.get("fasnet_real_score_avg") is not None
    ]
    spoofs = [
        float(e["fasnet_spoof_score_combined"])
        for e in evals
        if e.get("fasnet_spoof_score_combined") is not None
    ]
    native = Counter(e.get("fasnet_native_class") for e in evals)
    shadow085 = Counter(
        shadow_state_for_eval(
            e, RESEARCH_CANDIDATE_REAL_MIN, RESEARCH_CANDIDATE_SPOOF_MIN
        )
        for e in evals
    )
    auth = Counter(e.get("auth_decision") for e in evals)
    return {
        "session_id": s["meta"]["session_id"],
        "label": s["meta"]["label"],
        "path": s.get("path"),
        "data_bearing": len(evals) > 0,
        "n_marked_evals": len(evals),
        "n_raw_eval_records": len(s.get("evals") or []),
        "n_mark_events": len(s.get("mark_events") or []),
        "native_class_frame_counts": {str(k): v for k, v in native.items()},
        "shadow_0.85_frame_counts": dict(shadow085),
        "auth_decision_frame_counts": dict(auth),
        "live_ACCEPT_count": int(auth.get("ACCEPT", 0)),
        "winning_score": _quantiles(wins),
        "real_score_avg": _quantiles(reals),
        "spoof_score_combined": _quantiles(spoofs),
        "note": (
            "Marked-interval evals only; frame counts are temporally dependent "
            "within a session — not independent trials."
        ),
    }


def split_data_bearing_sessions(
    data_bearing: list[dict[str, Any]],
    *,
    selection_fraction: float = 0.5,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Deterministic label-stratified session split (no frame mixing)."""
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in data_bearing:
        by_label[s["meta"]["label"]].append(s)

    split_info: dict[str, Any] = {
        "selection_fraction": selection_fraction,
        "by_label": {},
        "sufficient_for_held_out_validation": True,
        "limitations": [],
        "unit": "data-bearing sessions (zero-eval files excluded from split)",
    }
    selection_sessions: list[dict[str, Any]] = []
    validation_sessions: list[dict[str, Any]] = []

    for label in VALID_LABELS:
        group = sorted(
            by_label.get(label, []),
            key=lambda s: s["meta"]["session_id"],
        )
        n = len(group)
        if n == 0:
            split_info["by_label"][label] = {
                "n_data_bearing_sessions": 0,
                "selection": [],
                "validation": [],
            }
            split_info["sufficient_for_held_out_validation"] = False
            split_info["limitations"].append(
                f"No data-bearing {label} sessions (marked evals > 0)."
            )
            continue
        if n < 2:
            sel, val = group, []
            split_info["sufficient_for_held_out_validation"] = False
            split_info["limitations"].append(
                f"Only {n} data-bearing {label} session(s); need >=2 for "
                "independent selection+validation split."
            )
        else:
            n_sel = max(1, int(math.floor(n * selection_fraction)))
            n_sel = min(n_sel, n - 1)  # leave at least one for validation
            sel, val = group[:n_sel], group[n_sel:]
        split_info["by_label"][label] = {
            "n_data_bearing_sessions": n,
            "selection": [s["meta"]["session_id"] for s in sel],
            "validation": [s["meta"]["session_id"] for s in val],
            "selection_marked_evals": sum(len(marked_evals(s)) for s in sel),
            "validation_marked_evals": sum(len(marked_evals(s)) for s in val),
        }
        selection_sessions.extend(sel)
        validation_sessions.extend(val)

    return selection_sessions, validation_sessions, split_info


def _label_metrics(
    frames: list[dict[str, Any]],
    *,
    label: str,
    real_min: float,
    spoof_min: float,
    n_sessions: int,
) -> dict[str, Any]:
    states = Counter(shadow_state_for_eval(e, real_min, spoof_min) for e in frames)
    n = len(frames)
    n_error = int(states.get("ERROR", 0))
    n_real = int(states.get("REAL", 0))
    n_spoof = int(states.get("SPOOF", 0))
    n_unc = int(states.get("UNCERTAIN", 0))
    n_classifiable = n - n_error  # non-ERROR shadow states

    native_real = sum(1 for e in frames if e.get("fasnet_native_class") == "REAL")
    native_spoof = sum(1 for e in frames if e.get("fasnet_native_class") == "SPOOF")
    native_missing = sum(1 for e in frames if e.get("fasnet_native_class") is None)
    live_accept = sum(1 for e in frames if e.get("auth_decision") == "ACCEPT")
    live_deny = sum(1 for e in frames if e.get("auth_decision") == "DENY")
    live_other = n - live_accept - live_deny

    entry: dict[str, Any] = {
        "n_sessions": n_sessions,
        "n_marked_evals": n,
        "n_classifiable_shadow": n_classifiable,
        "shadow_state_frame_counts": {
            "REAL": n_real,
            "SPOOF": n_spoof,
            "UNCERTAIN": n_unc,
            "ERROR": n_error,
        },
        "native_class_frame_counts": {
            "REAL": native_real,
            "SPOOF": native_spoof,
            "missing_or_null": native_missing,
        },
        "live_auth_decision_counts": {
            "ACCEPT": live_accept,
            "DENY": live_deny,
            "other": live_other,
        },
        "shadow_REAL_rate": _rate(n_real, n, what="all marked evals"),
        "shadow_SPOOF_rate": _rate(n_spoof, n, what="all marked evals"),
        "shadow_UNCERTAIN_rate": _rate(n_unc, n, what="all marked evals"),
        "shadow_ERROR_rate": _rate(n_error, n, what="all marked evals"),
        "shadow_REAL_rate_among_classifiable": _rate(
            n_real, n_classifiable, what="non-ERROR marked evals"
        ),
        "native_REAL_rate": _rate(native_real, n, what="all marked evals"),
        "live_ACCEPT_rate": _rate(live_accept, n, what="all marked evals"),
        "live_policy_note": LIVE_POLICY_NOTE,
    }

    if label == "REAL":
        # Genuine presentation: acceptance / rejection under live policy + shadow.
        entry["genuine_live_ACCEPT_rate"] = _rate(
            live_accept, n, what="REAL-label marked evals"
        )
        entry["genuine_live_reject_rate"] = _rate(
            n - live_accept, n, what="REAL-label marked evals (non-ACCEPT)"
        )
        entry["genuine_shadow_REAL_rate"] = _rate(
            n_real, n, what="REAL-label marked evals"
        )
        entry["genuine_shadow_reject_or_uncertain_rate"] = _rate(
            n - n_real, n, what="REAL-label marked evals (not confident shadow REAL)"
        )
        entry["genuine_native_REAL_rate"] = _rate(
            native_real, n, what="REAL-label marked evals"
        )

    if label in ("PRINT", "SCREEN"):
        entry["false_REAL_shadow_rate"] = _rate(
            n_real,
            n,
            what=f"{label}-label marked evals (confident shadow REAL on attack)",
        )
        entry["false_REAL_shadow_rate_among_classifiable"] = _rate(
            n_real,
            n_classifiable,
            what=f"{label} non-ERROR marked evals",
        )
        entry["false_REAL_native_argmax_rate"] = _rate(
            native_real,
            n,
            what=f"{label}-label marked evals (native argmax REAL on attack)",
        )
        entry["false_ACCEPT_live_policy_rate"] = _rate(
            live_accept,
            n,
            what=f"{label}-label marked evals (live auth ACCEPT on attack)",
        )
        # Back-compat aliases used by older print paths / tests
        entry["false_REAL_shadow_frame_pct"] = entry["false_REAL_shadow_rate"]["pct"]
        entry["false_REAL_native_argmax_frame_pct"] = entry[
            "false_REAL_native_argmax_rate"
        ]["pct"]
        entry["n_frames"] = n

    # Back-compat numeric fields (None when N/A — printers must not treat as 0)
    entry["shadow_REAL_frame_pct"] = entry["shadow_REAL_rate"]["pct"]
    entry["shadow_UNCERTAIN_frame_pct"] = entry["shadow_UNCERTAIN_rate"]["pct"]
    entry["shadow_SPOOF_frame_pct"] = entry["shadow_SPOOF_rate"]["pct"]
    entry["native_REAL_frame_pct"] = entry["native_REAL_rate"]["pct"]
    entry["n_frames"] = n
    return entry


def eval_threshold_set(
    sess_list: list[dict[str, Any]], real_min: float, spoof_min: float
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "by_label": {},
        "real_min": real_min,
        "spoof_min": spoof_min,
        "shadow_vs_live_note": LIVE_POLICY_NOTE,
    }
    for label in VALID_LABELS:
        label_sessions = [s for s in sess_list if s["meta"]["label"] == label]
        frames = [e for s in label_sessions for e in marked_evals(s)]
        out["by_label"][label] = _label_metrics(
            frames,
            label=label,
            real_min=real_min,
            spoof_min=spoof_min,
            n_sessions=len(label_sessions),
        )
    return out


def analyze_sessions(
    sessions: list[dict[str, Any]],
    *,
    candidates: tuple[float, ...] = DEFAULT_CANDIDATES,
    selection_fraction: float = 0.5,
) -> dict[str, Any]:
    empty_sessions = [s for s in sessions if not is_data_bearing(s)]
    data_bearing = [s for s in sessions if is_data_bearing(s)]

    by_label_all: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_label_data: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in sessions:
        by_label_all[s["meta"]["label"]].append(s)
    for s in data_bearing:
        by_label_data[s["meta"]["label"]].append(s)

    session_summaries = [_session_summary(s) for s in sessions]
    # Stable order: data-bearing first by id, then empty
    session_summaries.sort(
        key=lambda r: (0 if r["data_bearing"] else 1, r["session_id"])
    )

    selection_sessions, validation_sessions, split_info = split_data_bearing_sessions(
        data_bearing, selection_fraction=selection_fraction
    )
    if empty_sessions:
        split_info["empty_sessions_excluded_from_split"] = [
            s["meta"]["session_id"] for s in empty_sessions
        ]
        split_info["limitations"].append(
            f"{len(empty_sessions)} session file(s) with zero marked evals "
            "preserved on disk but excluded from selection/validation splits."
        )

    candidate_results = []
    for thr in candidates:
        candidate_results.append(
            {
                "thresholds": {"real_min": thr, "spoof_min": thr},
                "selection": eval_threshold_set(selection_sessions, thr, thr),
                "validation": eval_threshold_set(validation_sessions, thr, thr),
            }
        )

    asymmetric: list[dict[str, Any]] = []
    for real_min in candidates:
        for spoof_min in candidates:
            asymmetric.append(
                {
                    "thresholds": {"real_min": real_min, "spoof_min": spoof_min},
                    "selection": eval_threshold_set(
                        selection_sessions, real_min, spoof_min
                    ),
                    "validation": eval_threshold_set(
                        validation_sessions, real_min, spoof_min
                    ),
                }
            )

    overlap: dict[str, Any] = {}
    for label in VALID_LABELS:
        if label == "REAL":
            vals = [
                float(e["fasnet_winning_score"])
                for s in selection_sessions
                if s["meta"]["label"] == "REAL"
                for e in marked_evals(s)
                if e.get("fasnet_winning_score") is not None
            ]
        else:
            vals = [
                float(e["fasnet_winning_score"])
                for s in selection_sessions
                if s["meta"]["label"] == label
                for e in marked_evals(s)
                if e.get("fasnet_winning_score") is not None
            ]
        overlap[label] = _quantiles(vals)

    total_marked = sum(len(marked_evals(s)) for s in sessions)

    session_level: dict[str, Any] = {"by_label": {}}
    for label in VALID_LABELS:
        rows = []
        for s in data_bearing:
            if s["meta"]["label"] != label:
                continue
            states = [
                shadow_state_for_eval(
                    e, RESEARCH_CANDIDATE_REAL_MIN, RESEARCH_CANDIDATE_SPOOF_MIN
                )
                for e in marked_evals(s)
            ]
            if not states:
                continue
            maj = Counter(states).most_common(1)[0][0]
            auth = Counter(e.get("auth_decision") for e in marked_evals(s))
            rows.append(
                {
                    "session_id": s["meta"]["session_id"],
                    "majority_shadow_0.85": maj,
                    "n_marked_evals": len(states),
                    "live_ACCEPT_count": int(auth.get("ACCEPT", 0)),
                }
            )
        session_level["by_label"][label] = {
            "n_sessions_with_frames": len(rows),
            "sessions": rows,
            "note": (
                "Majority shadow state per session at research-candidate 0.85/0.85 "
                "for description only — not an activated operating point. "
                + LIVE_POLICY_NOTE
            ),
        }

    insufficient = bool(split_info["limitations"]) or not split_info[
        "sufficient_for_held_out_validation"
    ]
    report = {
        "purpose": "FasNet confidence threshold research (NOT production calibration)",
        "disclaimer": (
            "Marked-interval evaluations only. Frame-level rates are descriptive; "
            "consecutive frames are dependent. "
            "winning_score / real_score_avg / spoof_score_combined are uncalibrated "
            "model confidences — NOT calibrated probabilities. "
            "Do not activate thresholds without adequate held-out sessions and review. "
            "0.85/0.85 remain RESEARCH_CANDIDATE_UNVALIDATED and are NOT auto-activated. "
            + LIVE_POLICY_NOTE
        ),
        "n_session_files": len(sessions),
        "n_sessions": len(sessions),  # back-compat
        "n_empty_session_files": len(empty_sessions),
        "n_data_bearing_sessions": len(data_bearing),
        "n_marked_frames_total": total_marked,
        "sessions_per_label_all_files": {k: len(v) for k, v in by_label_all.items()},
        "sessions_per_label_data_bearing": {
            k: len(by_label_data.get(k, [])) for k in VALID_LABELS
        },
        "sessions_per_label": {
            k: len(by_label_data.get(k, [])) for k in VALID_LABELS
        },  # data-bearing (analysis-relevant)
        "split": split_info,
        "per_session": session_summaries,
        "session_level_majority_shadow_0_85": session_level,
        "score_overlap_selection": overlap,
        "candidate_threshold_grids": candidate_results,
        "asymmetric_threshold_grids": asymmetric,
        "research_candidate_reference": {
            "real_min": RESEARCH_CANDIDATE_REAL_MIN,
            "spoof_min": RESEARCH_CANDIDATE_SPOOF_MIN,
            "status": "RESEARCH_CANDIDATE_UNVALIDATED",
            "auto_activated": False,
        },
        "calibration_complete": False,
        "insufficient_independent_validation": insufficient,
        "calibration_complete_reason": (
            "No marked evaluation frames found. Record physical sessions with M "
            "pressed during the test interval."
            if total_marked == 0
            else (
                "Insufficient independent validation evidence: "
                + "; ".join(split_info["limitations"])
                if insufficient
                else (
                    "Analysis produced candidate comparisons only; human review and "
                    "security acceptance still required before any auth activation. "
                    "No production threshold is selected or activated."
                )
            )
        ),
    }
    return report


def _fmt_rate(rate: dict[str, Any] | None) -> str:
    if not rate:
        return "N/A"
    if rate.get("pct") is None:
        return "N/A"
    return f"{rate['pct']:.1f}% ({rate['n']}/{rate['d']})"


def print_report(report: dict[str, Any]) -> None:
    print("=== FasNet calibration analysis (research only) ===")
    print(
        f"session_files={report.get('n_session_files', report['n_sessions'])} "
        f"data_bearing={report.get('n_data_bearing_sessions')} "
        f"empty_excluded_from_split={report.get('n_empty_session_files')} "
        f"marked_evals={report['n_marked_frames_total']}"
    )
    print(
        "data_bearing_per_label="
        f"{report.get('sessions_per_label_data_bearing') or report.get('sessions_per_label')}"
    )
    print(f"calibration_complete={report['calibration_complete']}")
    print(
        f"insufficient_independent_validation="
        f"{report.get('insufficient_independent_validation')}"
    )
    print(f"reason: {report['calibration_complete_reason']}")
    print(f"NOTE: {LIVE_POLICY_NOTE}")
    lim = report["split"].get("limitations") or []
    if lim:
        print("limitations:")
        for x in lim:
            print(f"  - {x}")
    print("\n-- Split (data-bearing sessions only) --")
    for label in VALID_LABELS:
        info = report["split"]["by_label"].get(label, {})
        print(
            f"  {label}: n={info.get('n_data_bearing_sessions', 0)} "
            f"selection={info.get('selection')} "
            f"validation={info.get('validation')}"
        )

    print("\n-- Per session (marked evals only) --")
    for s in report["per_session"]:
        tag = "DATA" if s.get("data_bearing") else "EMPTY"
        print(
            f"  [{tag}] {s['session_id']} label={s['label']} "
            f"marked_evals={s['n_marked_evals']} "
            f"shadow0.85={s['shadow_0.85_frame_counts']} "
            f"native={s['native_class_frame_counts']} "
            f"live_ACCEPT={s.get('live_ACCEPT_count', 0)}"
        )

    print("\n-- Candidate thresholds (selection AND validation) --")
    print(
        "  Rates use marked evals. falseREAL shadow = confident shadow REAL on "
        "attack label. live false ACCEPT = auth_decision ACCEPT under enforced policy."
    )
    for c in report["candidate_threshold_grids"]:
        thr = c["thresholds"]["real_min"]
        for split_name in ("selection", "validation"):
            pool = c[split_name]["by_label"]
            real = pool.get("REAL", {})
            pr = pool.get("PRINT", {})
            sc = pool.get("SCREEN", {})
            print(
                f"  thr={thr:.2f} [{split_name}] "
                f"REAL n_sess={real.get('n_sessions', 0)} "
                f"evals={real.get('n_marked_evals', real.get('n_frames', 0))} "
                f"shadowREAL={_fmt_rate(real.get('shadow_REAL_rate'))} "
                f"liveACCEPT={_fmt_rate(real.get('live_ACCEPT_rate') or real.get('genuine_live_ACCEPT_rate'))} "
                f"| PRINT falseREAL_shadow={_fmt_rate(pr.get('false_REAL_shadow_rate'))} "
                f"falseREAL_native={_fmt_rate(pr.get('false_REAL_native_argmax_rate'))} "
                f"live_falseACCEPT={_fmt_rate(pr.get('false_ACCEPT_live_policy_rate'))} "
                f"| SCREEN falseREAL_shadow={_fmt_rate(sc.get('false_REAL_shadow_rate'))} "
                f"live_falseACCEPT={_fmt_rate(sc.get('false_ACCEPT_live_policy_rate'))}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze labeled FasNet calibration JSONL (research only)"
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=DEFAULT_CALIBRATION_DIR,
        help="Directory containing session_*.jsonl",
    )
    parser.add_argument(
        "--session",
        action="append",
        default=None,
        help="Optional explicit session JSONL path (repeatable)",
    )
    parser.add_argument(
        "--candidates",
        type=str,
        default=",".join(str(x) for x in DEFAULT_CANDIDATES),
        help="Comma-separated REAL/SPOOF min candidates (symmetric)",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write full JSON report",
    )
    parser.add_argument(
        "--selection-fraction",
        type=float,
        default=0.5,
        help="Fraction of data-bearing sessions per label for selection (rest validation)",
    )
    args = parser.parse_args()

    paths: list[Path] = []
    if args.session:
        paths = [Path(p) for p in args.session]
    else:
        if not args.dir.is_dir():
            print(f"[INFO] No calibration directory yet: {args.dir}")
            print("Record sessions with live_friend_auth.py --record-label ...")
            report = analyze_sessions([])
            print_report(report)
            return 0
        paths = sorted(args.dir.glob("session_*.jsonl"))

    sessions = [load_session(p) for p in paths]
    candidates = tuple(float(x.strip()) for x in args.candidates.split(",") if x.strip())
    report = analyze_sessions(
        sessions,
        candidates=candidates,
        selection_fraction=args.selection_fraction,
    )
    print_report(report)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[OK] Wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
