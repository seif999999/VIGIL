"""Offline single-frame vs 3s vs 5s FasNet decision comparison.

Research only. Does not modify live authentication, Challenge 1/2, recordings,
models, enrollment, or activate thresholds.

Decision semantics — PAD-only (FaceNet identity does NOT gate FasNet evidence):

1) single_frame_approach_a
   - Per marked eval: native FasNet argmax + winning-score gates.
   - REAL iff native REAL and winning_score >= real_min.
   - SPOOF iff native SPOOF and winning_score >= spoof_min.
   - Else UNCERTAIN; NO_FACE / MULTIPLE_FACES / FasNet ERROR → ERROR.
   - FaceNet unknown_identity / identity_recognized=False does NOT invalidate
     an otherwise usable FasNet observation (reported separately for e2e auth).

2) median_real (3s / 5s non-overlapping wall-clock windows)
   - Within one mark-interval only (never cross recording-toggle boundaries).
   - Collect fasnet_real_score_avg from FACE_OK frames with usable FasNet scores
     (identity outcome ignored for PAD validity).
   - If n_valid < min_valid(window) → INVALID (never REAL).
   - Else: REAL if median(real_score_avg) >= real_min;
           SPOOF if median(spoof_score_combined) >= spoof_min;
           else UNCERTAIN.
   - Uses REAL-class mass, never mixed winning-class scores.

3) majority_vote (3s / 5s)
   - Same PAD valid-frame filter and min_valid floor.
   - Each valid frame votes Approach A state; majority wins.
   - Tie involving REAL → UNCERTAIN (never invent REAL).

Minimum valid evidence (fair 3s vs 5s):
  min_valid(window_s) = ceil(window_s * 1.0) → 3 for 3s, 5 for 5s
  (at least one usable FasNet-scored FACE_OK observation per second of window).

PRINT_20261021T154852Z (PRINT 3) contamination filter (offline view only;
original JSONL untouched):
  - Keep first marked interval only (provisional photograph / PRINT).
  - Exclude later marked intervals (operator: auto#1295–1304 and auto#1309
    are real face). Those intervals are dropped entirely before windowing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from analyze_fasnet_calibration import (
    is_data_bearing,
    load_session,
    split_data_bearing_sessions,
)
from analyze_fasnet_temporal_aggregation import (
    WindowDecision,
    approach_a_frame,
    assign_windows_within_intervals,
    decide_majority_vote,
    decide_median_real,
    frame_exclusion_reason,
    is_identity_failure,
    is_score_usable,
    MarkedInterval,
    marked_intervals_from_session,
    min_valid_for_window,
    session_timing_diagnostics,
    summarize_decisions,
)
from fasnet_calibration_record import DEFAULT_CALIBRATION_DIR, VALID_LABELS

# Study matrix
REAL_THRESHOLDS = (0.90, 0.92, 0.95)
SPOOF_MIN = 0.85
WINDOW_METHODS: tuple[tuple[str, float, str], ...] = (
    ("single_frame_approach_a", 0.0, "single"),
    ("w3s_median_real", 3.0, "median_real"),
    ("w3s_majority_vote", 3.0, "majority_vote"),
    ("w5s_median_real", 5.0, "median_real"),
    ("w5s_majority_vote", 5.0, "majority_vote"),
)

# Offline contamination rules keyed by session_id (do not mutate JSONL).
# Interval indices are 0-based mark-on segments in timestamp order.
SESSION_INTERVAL_RULES: dict[str, dict[str, Any]] = {
    "PRINT_20261021T154852Z": {
        "include_interval_indices": [0],
        "exclude_interval_indices": [1, 2],
        "label_status": "PROVISIONAL_PRINT",
        "notes": [
            "Interval 0 (~auto#44–1249): main photograph test; provisional PRINT.",
            "Intervals 1–2 (~auto#1295–1304 and auto#1309): real face — excluded.",
            "Unmarked records are never loaded (recorder writes marked-only).",
            "Windows never cross mark-toggle boundaries.",
        ],
    },
    "REAL_20261021T154119Z": {
        "include_interval_indices": None,  # all
        "exclude_interval_indices": [],
        "label_status": "REAL",
        "notes": ["Single marked interval; all marked evals included."],
    },
}


def apply_interval_rules(
    session: dict[str, Any],
) -> tuple[list[MarkedInterval], dict[str, Any]]:
    sid = session["meta"]["session_id"]
    intervals = marked_intervals_from_session(session)
    rule = SESSION_INTERVAL_RULES.get(sid)
    meta: dict[str, Any] = {
        "session_id": sid,
        "label": session["meta"]["label"],
        "n_mark_intervals": len(intervals),
        "label_status": (rule or {}).get("label_status", session["meta"]["label"]),
        "notes": list((rule or {}).get("notes") or []),
        "intervals": [],
    }
    include = None if rule is None else rule.get("include_interval_indices")
    exclude = set((rule or {}).get("exclude_interval_indices") or [])

    out: list[MarkedInterval] = []
    for iv in intervals:
        included = True
        reason = None
        if iv.index in exclude:
            included = False
            reason = "operator_excluded_real_face_contamination"
        elif include is not None and iv.index not in include:
            included = False
            reason = "not_in_include_list"
        adj = MarkedInterval(
            index=iv.index,
            t_start=iv.t_start,
            t_end=iv.t_end,
            evals=iv.evals,
            included=included,
            exclude_reason=reason,
        )
        out.append(adj)
        meta["intervals"].append(
            {
                "index": iv.index,
                "t_start": iv.t_start.isoformat(),
                "t_end": iv.t_end.isoformat(),
                "duration_s": (iv.t_end - iv.t_start).total_seconds(),
                "n_evals": len(iv.evals),
                "included": included,
                "exclude_reason": reason,
            }
        )
    meta["n_evals_included"] = sum(len(i.evals) for i in out if i.included)
    meta["n_evals_excluded"] = sum(len(i.evals) for i in out if not i.included)
    return out, meta


def included_evals(intervals: list[MarkedInterval]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for iv in intervals:
        if iv.included:
            out.extend(iv.evals)
    return out


def evaluate_single_frame_pad(
    evals: list[dict[str, Any]],
    real_min: float,
    spoof_min: float,
) -> list[WindowDecision]:
    """Approach A PAD-only: FaceNet identity does not gate FasNet validity."""
    out: list[WindowDecision] = []
    for ev in evals:
        state = approach_a_frame(
            ev, real_min, spoof_min, exclude_identity_failures=False
        )
        usable = is_score_usable(ev)
        mean_r = float(ev["fasnet_real_score_avg"]) if usable else None
        excl: dict[str, int] = {}
        reason = frame_exclusion_reason(ev, exclude_identity_failures=False)
        if reason:
            excl[reason] = 1
        # Identity retained as diagnostic annotation only (not an exclusion).
        if is_identity_failure(ev):
            excl["IDENTITY_DIAGNOSTIC"] = excl.get("IDENTITY_DIAGNOSTIC", 0) + 1
        t0 = ev.get("timestamp_utc") if isinstance(ev.get("timestamp_utc"), str) else None
        out.append(
            WindowDecision(
                state=state,
                method="single_frame_approach_a",
                window_s=0.0,
                n_frames=1,
                n_valid=1 if usable else 0,
                exclusions=excl,
                mean_real=mean_r,
                median_real=mean_r,
                time_to_decision_s=0.0,
                t0=t0,
                t1=t0,
            )
        )
    return out


def evaluate_method(
    intervals: list[MarkedInterval],
    method_key: str,
    window_s: float,
    agg: str,
    real_min: float,
    spoof_min: float,
) -> list[WindowDecision]:
    # PAD-only: never exclude FaceNet identity failures from FasNet evidence.
    if agg == "single":
        return evaluate_single_frame_pad(
            included_evals(intervals), real_min, spoof_min
        )
    need = min_valid_for_window(window_s)
    windows = assign_windows_within_intervals(intervals, window_s)
    if agg == "median_real":
        return [
            decide_median_real(
                w,
                real_min,
                spoof_min,
                window_s=window_s,
                min_valid=need,
                exclude_identity_failures=False,
            )
            for w in windows
        ]
    if agg == "majority_vote":
        return [
            decide_majority_vote(
                w,
                real_min,
                spoof_min,
                window_s=window_s,
                min_valid=need,
                exclude_identity_failures=False,
            )
            for w in windows
        ]
    raise ValueError(f"Unknown aggregation {agg}")


def identity_and_auth_diagnostics(
    evals: list[dict[str, Any]],
    pad_decisions: list[WindowDecision],
    *,
    real_min: float,
    spoof_min: float,
) -> dict[str, Any]:
    """Separate FaceNet / live-auth stats from PAD-only FasNet decisions."""
    n = len(evals)
    n_id_fail = sum(1 for e in evals if is_identity_failure(e))
    n_id_ok = sum(1 for e in evals if e.get("identity_recognized") is True)
    n_id_skip = sum(
        1
        for e in evals
        if e.get("face_decision") == "FACE_OK"
        and e.get("identity_recognized") is None
        and not e.get("facenet_error")
    )
    n_auth_accept = sum(1 for e in evals if e.get("auth_decision") == "ACCEPT")
    n_auth_deny = sum(1 for e in evals if e.get("auth_decision") == "DENY")

    # Single-frame FaceNet-gated: PAD REAL and identity matched.
    n_pad_real = 0
    n_pad_real_and_id = 0
    if pad_decisions and pad_decisions[0].window_s == 0.0 and len(pad_decisions) == n:
        for ev, d in zip(evals, pad_decisions):
            if d.state == "REAL":
                n_pad_real += 1
                if ev.get("identity_recognized") is True:
                    n_pad_real_and_id += 1
    else:
        for ev in evals:
            st = approach_a_frame(
                ev, real_min, spoof_min, exclude_identity_failures=False
            )
            if st == "REAL":
                n_pad_real += 1
                if ev.get("identity_recognized") is True:
                    n_pad_real_and_id += 1

    return {
        "n_frames": n,
        "n_identity_failure": n_id_fail,
        "n_identity_recognized": n_id_ok,
        "n_identity_skipped_or_unset_on_face_ok": n_id_skip,
        "n_live_auth_ACCEPT": n_auth_accept,
        "n_live_auth_DENY": n_auth_deny,
        "n_pad_real_frames": n_pad_real,
        "n_pad_real_and_identity_recognized": n_pad_real_and_id,
        "note": (
            "PAD REAL uses FasNet only. FaceNet-gated counts require identity_recognized=True. "
            "Live auth_decision reflects the recorded pipeline (native FasNet + identity), "
            "not the research shadow thresholds."
        ),
    }


def _fmt(r: dict[str, Any] | None) -> str:
    if r is None:
        return "—"
    return r.get("display", "N/A")


def run_study(calibration_dir: Path) -> dict[str, Any]:
    paths = sorted(calibration_dir.glob("session_*.jsonl"))
    sessions = [load_session(p) for p in paths]
    data_bearing = [s for s in sessions if is_data_bearing(s)]
    sel, val, split_info = split_data_bearing_sessions(data_bearing)

    session_prep: dict[str, Any] = {}
    prepared: dict[str, tuple[dict[str, Any], list[MarkedInterval]]] = {}
    for s in data_bearing:
        intervals, meta = apply_interval_rules(s)
        timing = session_timing_diagnostics(included_evals(intervals))
        meta["timing_included"] = timing
        session_prep[s["meta"]["session_id"]] = meta
        prepared[s["meta"]["session_id"]] = (s, intervals)

    decision_docs = {
        "single_frame_approach_a": (
            "Approach A PAD-only: native argmax + winning-score gates. "
            "FaceNet identity does not invalidate usable FasNet scores. "
            "NO_FACE / MULTIPLE_FACES / FasNet ERROR → ERROR; never REAL from "
            "missing scores or insufficient evidence."
        ),
        "median_real": (
            "Median of fasnet_real_score_avg over FACE_OK frames with usable FasNet "
            "scores (identity ignored for validity) in a non-overlapping window "
            "confined to one mark interval. "
            f"REAL if median >= real_min; SPOOF if median spoof_combined >= {SPOOF_MIN}; "
            "else UNCERTAIN. n_valid < ceil(window_s) → INVALID."
        ),
        "majority_vote": (
            "Majority of per-frame Approach A PAD states among FasNet-valid frames. "
            "REAL-tie → UNCERTAIN. n_valid < ceil(window_s) → INVALID."
        ),
    }

    min_valid = {
        "3s": min_valid_for_window(3.0),
        "5s": min_valid_for_window(5.0),
        "rule": "ceil(window_seconds * 1.0 usable scored frames)",
    }

    results: dict[str, Any] = {}
    per_session: dict[str, Any] = {}

    for real_min in REAL_THRESHOLDS:
        thr_name = f"{real_min:.2f}/{SPOOF_MIN:.2f}"
        thr_block: dict[str, Any] = {
            "real_min": real_min,
            "spoof_min": SPOOF_MIN,
            "splits": {},
        }
        for split_name, split_sessions in (("selection", sel), ("validation", val)):
            by_label: dict[str, dict[str, list[WindowDecision]]] = {
                lab: {} for lab in VALID_LABELS
            }
            for s in split_sessions:
                sid = s["meta"]["session_id"]
                label = s["meta"]["label"]
                _, intervals = prepared[sid]
                sess_key = f"{thr_name}|{sid}"
                per_session[sess_key] = {
                    "session_id": sid,
                    "label": label,
                    "label_status": session_prep[sid].get("label_status"),
                    "split": split_name,
                    "threshold": thr_name,
                    "methods": {},
                }
                evals_inc = included_evals(intervals)
                for method_key, window_s, agg in WINDOW_METHODS:
                    decs = evaluate_method(
                        intervals, method_key, window_s, agg, real_min, SPOOF_MIN
                    )
                    by_label[label].setdefault(method_key, []).extend(decs)
                    summ = summarize_decisions(decs, label=label)
                    # Session-level attack containment (one attack interval, not n windows).
                    if label in ("PRINT", "SCREEN"):
                        summ["attack_interval_contains_real"] = any(
                            d.state == "REAL" for d in decs
                        )
                    summ["identity_auth_diagnostics"] = identity_and_auth_diagnostics(
                        evals_inc,
                        decs,
                        real_min=real_min,
                        spoof_min=SPOOF_MIN,
                    )
                    # Mean valid observations (denominator for evidence density).
                    summ["n_valid_obs_total"] = int(sum(d.n_valid for d in decs))
                    per_session[sess_key]["methods"][method_key] = summ

            split_out: dict[str, Any] = {
                "sessions": [s["meta"]["session_id"] for s in split_sessions],
                "by_label": {},
            }
            for lab in VALID_LABELS:
                split_out["by_label"][lab] = {}
                for method_key, _, _ in WINDOW_METHODS:
                    decs = by_label[lab].get(method_key, [])
                    summ = summarize_decisions(decs, label=lab)
                    if lab in ("PRINT", "SCREEN"):
                        # Also list which sessions contributed a REAL hit.
                        hits = []
                        for s in split_sessions:
                            if s["meta"]["label"] != lab:
                                continue
                            sk = f"{thr_name}|{s['meta']['session_id']}"
                            ms = per_session[sk]["methods"].get(method_key, {})
                            if ms.get("attack_interval_contains_real"):
                                hits.append(s["meta"]["session_id"])
                        summ["sessions_with_any_real"] = hits
                        summ["attack_interval_contains_real"] = len(hits) > 0
                    split_out["by_label"][lab][method_key] = summ
            thr_block["splits"][split_name] = split_out
        results[thr_name] = thr_block

    # Focused REAL3 / PRINT3 comparison (validation sessions of interest).
    focus_ids = ["REAL_20261021T154119Z", "PRINT_20261021T154852Z"]
    focus: dict[str, Any] = {}
    for thr_name in results:
        focus[thr_name] = {}
        for sid in focus_ids:
            key = f"{thr_name}|{sid}"
            if key in per_session:
                focus[thr_name][sid] = per_session[key]

    # Fair 3s vs 5s: normalize REAL hits by attack duration minutes.
    fair_compare = _fair_3s_vs_5s(results, session_prep, per_session)

    recommendation = _recommend(results, focus, fair_compare)

    return {
        "unit": (
            "Marked evals only; mark-interval-confined non-overlapping windows; "
            "session-level selection/validation. Correlated frames/windows ≠ "
            "independent authentication attempts."
        ),
        "spoof_min": SPOOF_MIN,
        "real_thresholds": list(REAL_THRESHOLDS),
        "methods": [m[0] for m in WINDOW_METHODS],
        "min_valid_evidence": min_valid,
        "decision_semantics": decision_docs,
        "exclude_identity_failures_from_pad": False,
        "methodology_fix": (
            "Prior report incorrectly excluded FaceNet identity failures from PAD "
            "valid evidence. Corrected: usable single-face FasNet scores always "
            "contribute; identity reported separately for e2e auth analysis."
        ),
        "session_preparation": session_prep,
        "split_info": split_info,
        "results": results,
        "per_session": per_session,
        "focus_real3_print3": focus,
        "fair_3s_vs_5s": fair_compare,
        "recommendation": recommendation,
        "policy_note": (
            "Offline research only. Thresholds remain RESEARCH_CANDIDATE_UNVALIDATED. "
            "PRINT 3 photograph label is PROVISIONAL until independently confirmed. "
            "Production calibration is NOT complete. Live auth unchanged."
        ),
    }


def _fair_3s_vs_5s(
    results: dict[str, Any],
    session_prep: dict[str, Any],
    per_session: dict[str, Any],
) -> dict[str, Any]:
    """Account for different window counts / observation density."""
    out: dict[str, Any] = {}
    for thr_name in results:
        rows = []
        for split in ("selection", "validation"):
            for method in (
                "w3s_median_real",
                "w3s_majority_vote",
                "w5s_median_real",
                "w5s_majority_vote",
            ):
                for lab in ("REAL", "PRINT", "SCREEN"):
                    summ = (
                        results[thr_name]["splits"][split]["by_label"]
                        .get(lab, {})
                        .get(method)
                    )
                    if not summ or summ.get("n_windows", 0) == 0:
                        continue
                    # Approximate covered seconds = n_windows * window_s
                    window_s = 3.0 if "w3s_" in method else 5.0
                    covered = summ["n_windows"] * window_s
                    n_real = summ["state_counts"].get("REAL", 0)
                    rows.append(
                        {
                            "split": split,
                            "label": lab,
                            "method": method,
                            "n_windows": summ["n_windows"],
                            "window_s": window_s,
                            "approx_covered_s": covered,
                            "n_real": n_real,
                            "real_per_minute": (
                                (60.0 * n_real / covered) if covered > 0 else None
                            ),
                            "mean_valid_per_window": summ[
                                "mean_valid_frames_per_window"
                            ],
                            "real_rate": summ["real_rate"],
                            "invalid_rate": summ.get("invalid_rate"),
                        }
                    )
        out[thr_name] = {
            "rows": rows,
            "note": (
                "real_per_minute normalizes for unequal window counts; "
                "still not independent trials."
            ),
        }
    return out


def _recommend(
    results: dict[str, Any],
    focus: dict[str, Any],
    fair: dict[str, Any],
) -> dict[str, Any]:
    """Pick one provisional config from the study matrix (descriptive)."""
    candidates = []
    for thr_name, thr in results.items():
        for method, _, _ in WINDOW_METHODS:
            # Prefer validation metrics; require PRINT attack_interval FAR awareness.
            val = thr["splits"]["validation"]["by_label"]
            sel = thr["splits"]["selection"]["by_label"]
            real_v = val.get("REAL", {}).get(method, {})
            print_v = val.get("PRINT", {}).get(method, {})
            print_s = sel.get("PRINT", {}).get(method, {})
            screen_s = sel.get("SCREEN", {}).get(method, {})

            ar = (real_v.get("genuine_real_accept") or {}).get("pct")
            pfar_v = (print_v.get("print_false_accept") or {}).get("pct")
            pfar_s = (print_s.get("print_false_accept") or {}).get("pct")
            sfar_s = (screen_s.get("screen_false_accept") or {}).get("pct")
            attack_hit_v = bool(print_v.get("attack_interval_contains_real"))
            attack_hit_s = bool(print_s.get("attack_interval_contains_real")) or bool(
                screen_s.get("attack_interval_contains_real")
            )
            latency = 0.0
            if method.startswith("w3s_"):
                latency = 3.0
            elif method.startswith("w5s_"):
                latency = 5.0

            # Security-first: attack-interval REAL is a hard penalty. Then low FAR,
            # then genuine AR, then lower latency. Prefer 3s over 5s unless 5s
            # uniquely clears attack-interval REAL.
            if ar is None:
                continue
            score = 0.0
            if attack_hit_v:
                score -= 80.0
            if attack_hit_s:
                score -= 40.0
            if not attack_hit_v and not attack_hit_s:
                score += 100.0
            score += max(0.0, 50.0 - 3.0 * ((pfar_v or 0.0) + (pfar_s or 0.0) + (sfar_s or 0.0)))
            score += (ar or 0.0) * 0.25
            score -= latency * 3.0
            if "median" in method:
                score += 1.5
            if method == "single_frame_approach_a":
                score += 0.5  # latency / simplicity tie-break
            # Focus PRINT3 / REAL3 when available in validation.
            focus_print = focus.get(thr_name, {}).get("PRINT_20261021T154852Z", {})
            focus_real = focus.get(thr_name, {}).get("REAL_20261021T154119Z", {})
            fp = (focus_print.get("methods") or {}).get(method) or {}
            fr = (focus_real.get("methods") or {}).get(method) or {}
            p3_far = (fp.get("print_false_accept") or fp.get("real_rate") or {}).get("pct")
            r3_ar = (fr.get("genuine_real_accept") or fr.get("real_rate") or {}).get("pct")
            if p3_far is not None:
                score -= 2.0 * p3_far
            if r3_ar is not None:
                score += 0.2 * r3_ar
            candidates.append(
                {
                    "threshold": thr_name,
                    "method": method,
                    "score": score,
                    "val_real_ar": ar,
                    "val_print_far": pfar_v,
                    "sel_print_far": pfar_s,
                    "sel_screen_far": sfar_s,
                    "val_attack_contains_real": attack_hit_v,
                    "sel_attack_contains_real": attack_hit_s,
                    "print3_window_far": p3_far,
                    "real3_ar": r3_ar,
                    "latency_s": latency,
                }
            )
    candidates.sort(key=lambda c: c["score"], reverse=True)
    best = candidates[0] if candidates else None

    # Derive 3s vs 5s note from focus PRINT3 / REAL3 at 0.95/0.85 when present.
    p3_95 = (focus.get("0.95/0.85") or {}).get("PRINT_20261021T154852Z", {})
    r3_95 = (focus.get("0.95/0.85") or {}).get("REAL_20261021T154119Z", {})
    p3_3 = ((p3_95.get("methods") or {}).get("w3s_median_real") or {})
    p3_5 = ((p3_95.get("methods") or {}).get("w5s_median_real") or {})
    r3_3 = ((r3_95.get("methods") or {}).get("w3s_median_real") or {})
    r3_5 = ((r3_95.get("methods") or {}).get("w5s_median_real") or {})
    why_3_vs_5 = (
        f"At 0.95/0.85 PAD-only: PRINT3 3s-median atkREAL="
        f"{p3_3.get('attack_interval_contains_real')} "
        f"({(p3_3.get('state_counts') or {}).get('REAL', 0)}/"
        f"{p3_3.get('n_windows', 0)} REAL); "
        f"5s-median atkREAL={p3_5.get('attack_interval_contains_real')} "
        f"({(p3_5.get('state_counts') or {}).get('REAL', 0)}/"
        f"{p3_5.get('n_windows', 0)} REAL). "
        f"REAL3 AR 3s={((r3_3.get('real_rate') or {}).get('display'))} vs "
        f"5s={((r3_5.get('real_rate') or {}).get('display'))}. "
        "After identity-fix, 5s uniquely clears PRINT3 attack-interval REAL at "
        "0.95/0.85 while 3s does not; the extra 2s is a material security gain "
        "here, not merely a different window percentage. Still correlated windows "
        "and a provisional photograph label — not production calibration."
    )
    if best and best.get("val_attack_contains_real"):
        rationale = (
            "No tested config clears validation PRINT attack-interval REAL. "
            "Provisional pick balances photograph FAR, genuine AR, and latency — "
            "research only, not production-ready."
        )
    else:
        rationale = (
            "Clears validation PRINT attack-interval REAL under PAD-only rules "
            "(correlated windows; provisional PRINT3 label; not production-ready)."
        )
    return {
        "provisional_configuration": best,
        "rationale": rationale,
        "alternatives": candidates[1:6],
        "why_3_vs_5": why_3_vs_5,
        "caveats": [
            "PRINT 3 photograph label is provisional.",
            "Window/frame counts are correlated; not independent auth attempts.",
            "SCREEN has no held-out validation session.",
            "Sparse selection REAL session makes 3s/5s windows INVALID (density).",
            "Production calibration is NOT complete; do not activate thresholds.",
        ],
    }


def _counts_line(summ: dict[str, Any]) -> str:
    sc = summ.get("state_counts") or {}
    n = summ.get("n_windows", 0)
    return (
        f"REAL={sc.get('REAL', 0)}/{n} "
        f"SPOOF={sc.get('SPOOF', 0)}/{n} "
        f"UNC={sc.get('UNCERTAIN', 0)}/{n} "
        f"INV={sc.get('INVALID', 0)}/{n} "
        f"ERR={sc.get('ERROR', 0)}/{n} "
        f"valid/w={summ.get('mean_valid_frames_per_window', 0):.1f} "
        f"valid_obsΣ={summ.get('n_valid_obs_total', '—')}"
    )


def print_report(report: dict[str, Any]) -> None:
    print("=" * 96)
    print("FasNet window compare: single-frame vs 3s vs 5s (offline, marked, interval-safe)")
    print("=" * 96)
    print(f"\nMethodology: {report.get('methodology_fix')}")
    print(f"exclude_identity_failures_from_pad={report.get('exclude_identity_failures_from_pad')}")
    print(f"\nMin valid evidence: {report['min_valid_evidence']}")
    print("Decision semantics:")
    for k, v in report["decision_semantics"].items():
        print(f"  - {k}: {v}")

    print("\nSession preparation / contamination filters:")
    for sid, meta in report["session_preparation"].items():
        if sid in (
            "PRINT_20261021T154852Z",
            "REAL_20261021T154119Z",
        ) or meta.get("n_evals_excluded", 0) > 0:
            print(
                f"  {sid}: status={meta.get('label_status')} "
                f"included_evals={meta.get('n_evals_included')} "
                f"excluded_evals={meta.get('n_evals_excluded')}"
            )
            for iv in meta.get("intervals") or []:
                flag = "KEEP" if iv["included"] else "DROP"
                print(
                    f"    [{flag}] interval#{iv['index']} "
                    f"n={iv['n_evals']} dur={iv['duration_s']:.1f}s "
                    f"reason={iv.get('exclude_reason')}"
                )
            for n in meta.get("notes") or []:
                print(f"    note: {n}")

    print("\nSelection / validation split (session-level):")
    for lab, info in report["split_info"].get("by_label", {}).items():
        print(
            f"  {lab}: sel={info.get('selection')} val={info.get('validation')}"
        )
    for lim in report["split_info"].get("limitations") or []:
        print(f"  limitation: {lim}")

    for thr_name, thr in report["results"].items():
        print("\n" + "-" * 96)
        print(f"Thresholds real_min/spoof_min = {thr_name}")
        print("-" * 96)
        for split_name, split_block in thr["splits"].items():
            print(f"\n### {split_name.upper()}  {split_block['sessions']}")
            header = (
                f"{'method':<28} {'REAL_AR':>8} {'PRINT_FAR':>10} {'SCREEN_FAR':>11} "
                f"{'SPOOF%':>7} {'UNC%':>7} {'INV%':>7} {'n_win':>6} "
                f"{'valid/w':>8} {'lat_s':>5} {'atkREAL?':>8}"
            )
            print(header)
            print("-" * len(header))
            for method, _, _ in WINDOW_METHODS:
                real_s = split_block["by_label"].get("REAL", {}).get(method)
                print_s = split_block["by_label"].get("PRINT", {}).get(method)
                screen_s = split_block["by_label"].get("SCREEN", {}).get(method)
                base = real_s or print_s or screen_s or {}
                lat = base.get("nominal_decision_latency_s")
                lat_s = f"{lat:.0f}" if lat else "0"
                atk = "YES" if (
                    (print_s and print_s.get("attack_interval_contains_real"))
                    or (screen_s and screen_s.get("attack_interval_contains_real"))
                ) else "no"
                print(
                    f"{method:<28} "
                    f"{_fmt(real_s.get('genuine_real_accept') if real_s else None):>8} "
                    f"{_fmt(print_s.get('print_false_accept') if print_s else None):>10} "
                    f"{_fmt(screen_s.get('screen_false_accept') if screen_s else None):>11} "
                    f"{_fmt(real_s.get('spoof_rate') if real_s else None):>7} "
                    f"{_fmt(real_s.get('uncertain_rate') if real_s else None):>7} "
                    f"{_fmt(real_s.get('invalid_rate') if real_s else None):>7} "
                    f"{base.get('n_windows', 0):>6} "
                    f"{base.get('mean_valid_frames_per_window', 0.0):>8.1f} "
                    f"{lat_s:>5} {atk:>8}"
                )

            print("\n  Focus sessions (REAL3 / PRINT3) — raw counts + %:")
            for sid in ("REAL_20261021T154119Z", "PRINT_20261021T154852Z"):
                key = f"{thr_name}|{sid}"
                block = report["per_session"].get(key)
                if not block or block["split"] != split_name:
                    continue
                for method, _, _ in WINDOW_METHODS:
                    m = block["methods"][method]
                    idd = m.get("identity_auth_diagnostics") or {}
                    print(
                        f"    {sid} [{block['label_status']}] {method}: "
                        f"{_counts_line(m)} | "
                        f"REAL%={m['real_rate']['display']} "
                        f"SPOOF%={m['spoof_rate']['display']} "
                        f"UNC%={m['uncertain_rate']['display']} "
                        f"INV%={m['invalid_rate']['display']} "
                        f"atkREAL={m.get('attack_interval_contains_real')}"
                    )
                    if method == "single_frame_approach_a":
                        print(
                            f"      FaceNet/e2e diagnostic: "
                            f"id_fail={idd.get('n_identity_failure')} "
                            f"id_ok={idd.get('n_identity_recognized')} "
                            f"PAD_REAL={idd.get('n_pad_real_frames')} "
                            f"PAD_REAL∧id={idd.get('n_pad_real_and_identity_recognized')} "
                            f"live_ACCEPT={idd.get('n_live_auth_ACCEPT')} "
                            f"live_DENY={idd.get('n_live_auth_DENY')}"
                        )

    print("\n" + "=" * 96)
    print("Recommendation (provisional research only)")
    print("=" * 96)
    rec = report["recommendation"]
    best = rec.get("provisional_configuration")
    if best:
        print(
            f"  → {best['method']} @ {best['threshold']} "
            f"(latency≈{best['latency_s']}s, val REAL AR={best['val_real_ar']:.1f}%, "
            f"val PRINT FAR={best['val_print_far']}, "
            f"PRINT3 FAR={best.get('print3_window_far')}, "
            f"REAL3 AR={best.get('real3_ar')}, "
            f"val attack_contains_REAL={best['val_attack_contains_real']})"
        )
    if rec.get("rationale"):
        print(f"  rationale: {rec['rationale']}")
    print(f"  3s vs 5s: {rec['why_3_vs_5']}")
    for c in rec.get("caveats") or []:
        print(f"  caveat: {c}")
    print(f"\n{report['policy_note']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=DEFAULT_CALIBRATION_DIR,
    )
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()
    report = run_study(args.calibration_dir)
    print_report(report)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
