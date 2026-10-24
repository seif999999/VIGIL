"""Offline temporal aggregation of FasNet Approach A confidence decisions.

Research only. Does not modify live authentication, Challenge 1/2, recordings,
models, enrollment, or activate thresholds.

Uses marked JSONL evaluations from data-bearing sessions and the same
session-level selection/validation split as analyze_fasnet_calibration.py.

Windows are non-overlapping wall-clock bins from timestamp_utc. Per-frame
Approach A (native argmax + winning-score gates) remains the baseline.
Aggregations that average confidence use fasnet_real_score_avg (REAL class
mass), never winning-class scores from mixed classes.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal

from analyze_fasnet_calibration import (
    is_data_bearing,
    load_session,
    marked_evals,
    shadow_state_for_eval,
    split_data_bearing_sessions,
)
from fasnet_calibration_record import DEFAULT_CALIBRATION_DIR, VALID_LABELS

DecisionState = Literal["REAL", "SPOOF", "UNCERTAIN", "ERROR", "INVALID"]

# Research threshold configurations requested for this study.
THRESHOLD_CONFIGS: tuple[tuple[str, float, float], ...] = (
    ("0.85/0.85", 0.85, 0.85),
    ("0.90/0.90", 0.90, 0.90),
    ("0.95/0.85", 0.95, 0.85),
)

WINDOW_SECONDS: tuple[float, ...] = (3.0, 5.0, 10.0)

# Minimum FACE_OK frames with usable REAL-class scores before a window may
# emit REAL (never allow sparse / empty windows to accept).
MIN_VALID_FOR_REAL = 1

# Duration-proportional floor used by the 3s/5s comparison study:
# at least one usable scored observation per second of window.
MIN_VALID_PER_SECOND = 1.0

# Conservative: require this fraction of valid frames to be Approach A REAL,
# and mean REAL-class mass >= real_min. Otherwise never REAL.
CONSERVATIVE_REAL_FRACTION = 1.0


def min_valid_for_window(window_s: float) -> int:
    """Minimum usable scored frames required before a window may emit REAL."""
    if window_s <= 0:
        return MIN_VALID_FOR_REAL
    return max(MIN_VALID_FOR_REAL, int(math.ceil(window_s * MIN_VALID_PER_SECOND)))


def parse_timestamp(value: Any) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def is_score_usable(ev: dict[str, Any]) -> bool:
    """FACE_OK with REAL-class mass present and no FasNet error."""
    if ev.get("face_decision") != "FACE_OK":
        return False
    if ev.get("fasnet_error"):
        return False
    if ev.get("fasnet_native_class") in (None, "ERROR"):
        return False
    if ev.get("fasnet_real_score_avg") is None:
        return False
    try:
        float(ev["fasnet_real_score_avg"])
    except (TypeError, ValueError):
        return False
    return True


def is_identity_failure(ev: dict[str, Any]) -> bool:
    """Explicit FaceNet failure / negative match (not merely FaceNet skipped)."""
    if ev.get("facenet_error"):
        return True
    if ev.get("face_decision") == "FACE_OK" and ev.get("identity_recognized") is False:
        return True
    return False


def frame_exclusion_reason(
    ev: dict[str, Any],
    *,
    exclude_identity_failures: bool = False,
) -> str | None:
    """Return why a frame cannot contribute REAL-class scores, else None."""
    fd = ev.get("face_decision")
    if fd == "NO_FACE":
        return "NO_FACE"
    if fd == "MULTIPLE_FACES":
        return "MULTIPLE_FACES"
    if fd is not None and fd != "FACE_OK":
        return f"FACE_{fd}"
    if ev.get("fasnet_error"):
        return "FASNET_ERROR"
    if exclude_identity_failures and is_identity_failure(ev):
        return "IDENTITY_FAILURE"
    if ev.get("fasnet_native_class") in (None, "ERROR"):
        return "MISSING_OR_ERROR_CLASS"
    if ev.get("fasnet_real_score_avg") is None:
        return "MISSING_REAL_SCORE"
    return None


def approach_a_frame(
    ev: dict[str, Any],
    real_min: float,
    spoof_min: float,
    *,
    exclude_identity_failures: bool = False,
) -> DecisionState:
    reason = frame_exclusion_reason(
        ev, exclude_identity_failures=exclude_identity_failures
    )
    if reason in (
        "NO_FACE",
        "MULTIPLE_FACES",
        "FASNET_ERROR",
        "MISSING_OR_ERROR_CLASS",
        "IDENTITY_FAILURE",
    ):
        return "ERROR" if reason != "IDENTITY_FAILURE" else "INVALID"
    if reason == "MISSING_REAL_SCORE" and ev.get("fasnet_winning_score") is None:
        return "ERROR"
    state = shadow_state_for_eval(ev, real_min, spoof_min)
    # Tracking / identity uncertainty must never yield confident REAL.
    if (
        exclude_identity_failures
        and state == "REAL"
        and is_identity_failure(ev)
    ):
        return "INVALID"
    return state  # type: ignore[return-value]


def _decide_from_real_mass(
    real_mass: float,
    spoof_mass: float | None,
    real_min: float,
    spoof_min: float,
) -> DecisionState:
    """Map aggregated REAL-class mass (+ optional spoof mass) to a state.

    REAL only when real_mass >= real_min. SPOOF when spoof mass clears spoof_min
    (or, if spoof mass missing, when 1 - real_mass >= spoof_min). Else UNCERTAIN.
    """
    if real_mass >= real_min:
        return "REAL"
    s = spoof_mass if spoof_mass is not None else (1.0 - real_mass)
    if s >= spoof_min:
        return "SPOOF"
    return "UNCERTAIN"


@dataclass(frozen=True)
class WindowDecision:
    state: DecisionState
    method: str
    window_s: float
    n_frames: int
    n_valid: int
    exclusions: dict[str, int]
    mean_real: float | None
    median_real: float | None
    time_to_decision_s: float | None
    t0: str | None
    t1: str | None


def _stats(vals: list[float]) -> tuple[float | None, float | None]:
    if not vals:
        return None, None
    return float(statistics.mean(vals)), float(statistics.median(vals))


def decide_mean_real(
    frames: list[dict[str, Any]],
    real_min: float,
    spoof_min: float,
    *,
    window_s: float,
    min_valid: int | None = None,
    exclude_identity_failures: bool = False,
) -> WindowDecision:
    return _aggregate_real_mass(
        frames,
        real_min,
        spoof_min,
        window_s=window_s,
        method="mean_real",
        use_median=False,
        min_valid=min_valid,
        exclude_identity_failures=exclude_identity_failures,
    )


def decide_median_real(
    frames: list[dict[str, Any]],
    real_min: float,
    spoof_min: float,
    *,
    window_s: float,
    min_valid: int | None = None,
    exclude_identity_failures: bool = False,
) -> WindowDecision:
    return _aggregate_real_mass(
        frames,
        real_min,
        spoof_min,
        window_s=window_s,
        method="median_real",
        use_median=True,
        min_valid=min_valid,
        exclude_identity_failures=exclude_identity_failures,
    )


def _insufficient_state(
    frames: list[dict[str, Any]],
    exclusions: Counter[str],
    n_valid: int,
) -> DecisionState:
    """Map insufficient evidence to INVALID/ERROR — never REAL."""
    if not frames:
        return "ERROR"
    if n_valid == 0 and (
        exclusions.get("FASNET_ERROR", 0) + exclusions.get("MISSING_OR_ERROR_CLASS", 0)
        == sum(exclusions.values())
        and sum(exclusions.values()) > 0
    ):
        return "ERROR"
    return "INVALID"


def _aggregate_real_mass(
    frames: list[dict[str, Any]],
    real_min: float,
    spoof_min: float,
    *,
    window_s: float,
    method: str,
    use_median: bool,
    min_valid: int | None = None,
    exclude_identity_failures: bool = False,
) -> WindowDecision:
    need = MIN_VALID_FOR_REAL if min_valid is None else max(1, int(min_valid))
    exclusions: Counter[str] = Counter()
    reals: list[float] = []
    spoofs: list[float] = []
    for ev in frames:
        reason = frame_exclusion_reason(
            ev, exclude_identity_failures=exclude_identity_failures
        )
        if reason:
            exclusions[reason] += 1
            continue
        reals.append(float(ev["fasnet_real_score_avg"]))
        if ev.get("fasnet_spoof_score_combined") is not None:
            spoofs.append(float(ev["fasnet_spoof_score_combined"]))

    mean_r, med_r = _stats(reals)
    mean_s, med_s = _stats(spoofs)
    n_valid = len(reals)
    t0, t1, ttd = _window_timing(frames)

    if n_valid < need:
        return WindowDecision(
            state=_insufficient_state(frames, exclusions, n_valid),
            method=method,
            window_s=window_s,
            n_frames=len(frames),
            n_valid=n_valid,
            exclusions=dict(exclusions),
            mean_real=mean_r,
            median_real=med_r,
            time_to_decision_s=ttd,
            t0=t0,
            t1=t1,
        )

    mass = med_r if use_median else mean_r
    spoof_mass = med_s if use_median else mean_s
    assert mass is not None
    state = _decide_from_real_mass(mass, spoof_mass, real_min, spoof_min)
    return WindowDecision(
        state=state,
        method=method,
        window_s=window_s,
        n_frames=len(frames),
        n_valid=n_valid,
        exclusions=dict(exclusions),
        mean_real=mean_r,
        median_real=med_r,
        time_to_decision_s=ttd,
        t0=t0,
        t1=t1,
    )


def decide_majority_vote(
    frames: list[dict[str, Any]],
    real_min: float,
    spoof_min: float,
    *,
    window_s: float,
    min_valid: int | None = None,
    exclude_identity_failures: bool = False,
) -> WindowDecision:
    need = MIN_VALID_FOR_REAL if min_valid is None else max(1, int(min_valid))
    exclusions: Counter[str] = Counter()
    votes: list[DecisionState] = []
    reals: list[float] = []
    for ev in frames:
        reason = frame_exclusion_reason(
            ev, exclude_identity_failures=exclude_identity_failures
        )
        if reason:
            exclusions[reason] += 1
            # Explicit non-contributors: do not vote REAL.
            continue
        st = approach_a_frame(
            ev,
            real_min,
            spoof_min,
            exclude_identity_failures=exclude_identity_failures,
        )
        if st in ("ERROR", "INVALID"):
            exclusions[f"APPROACH_A_{st}"] += 1
            continue
        votes.append(st)
        reals.append(float(ev["fasnet_real_score_avg"]))

    mean_r, med_r = _stats(reals)
    t0, t1, ttd = _window_timing(frames)
    n_valid = len(votes)

    if n_valid < need:
        return WindowDecision(
            state=_insufficient_state(frames, exclusions, n_valid),
            method="majority_vote",
            window_s=window_s,
            n_frames=len(frames),
            n_valid=n_valid,
            exclusions=dict(exclusions),
            mean_real=mean_r,
            median_real=med_r,
            time_to_decision_s=ttd,
            t0=t0,
            t1=t1,
        )

    counts = Counter(votes)
    # Prefer REAL only if it strictly ties-or-wins AND meets min support.
    # Tie between REAL and non-REAL → UNCERTAIN (never invent REAL).
    best = counts.most_common()
    top_state, top_n = best[0]
    tied = [s for s, n in best if n == top_n]
    if len(tied) > 1:
        if "REAL" in tied:
            state = "UNCERTAIN"
        elif "SPOOF" in tied:
            state = "SPOOF" if "UNCERTAIN" not in tied else "UNCERTAIN"
        else:
            state = "UNCERTAIN"
    else:
        state = top_state

    return WindowDecision(
        state=state,
        method="majority_vote",
        window_s=window_s,
        n_frames=len(frames),
        n_valid=n_valid,
        exclusions=dict(exclusions),
        mean_real=mean_r,
        median_real=med_r,
        time_to_decision_s=ttd,
        t0=t0,
        t1=t1,
    )


def decide_conservative_sustained(
    frames: list[dict[str, Any]],
    real_min: float,
    spoof_min: float,
    *,
    window_s: float,
    required_fraction: float = CONSERVATIVE_REAL_FRACTION,
    min_valid: int | None = None,
    exclude_identity_failures: bool = False,
) -> WindowDecision:
    """Require sustained Approach A REAL evidence across the window.

    REAL only if:
      - n_valid >= min_valid
      - fraction of Approach A REAL among valid frames >= required_fraction
      - mean fasnet_real_score_avg >= real_min
    Else SPOOF if any Approach A SPOOF (and no REAL path), else UNCERTAIN.
    Insufficient valid frames → INVALID/ERROR, never REAL.
    """
    need = MIN_VALID_FOR_REAL if min_valid is None else max(1, int(min_valid))
    exclusions: Counter[str] = Counter()
    votes: list[DecisionState] = []
    reals: list[float] = []
    for ev in frames:
        reason = frame_exclusion_reason(
            ev, exclude_identity_failures=exclude_identity_failures
        )
        if reason:
            exclusions[reason] += 1
            continue
        st = approach_a_frame(
            ev,
            real_min,
            spoof_min,
            exclude_identity_failures=exclude_identity_failures,
        )
        if st in ("ERROR", "INVALID"):
            exclusions[f"APPROACH_A_{st}"] += 1
            continue
        votes.append(st)
        reals.append(float(ev["fasnet_real_score_avg"]))

    mean_r, med_r = _stats(reals)
    t0, t1, ttd = _window_timing(frames)
    n_valid = len(votes)

    if n_valid < need:
        return WindowDecision(
            state=_insufficient_state(frames, exclusions, n_valid),
            method="conservative_sustained",
            window_s=window_s,
            n_frames=len(frames),
            n_valid=n_valid,
            exclusions=dict(exclusions),
            mean_real=mean_r,
            median_real=med_r,
            time_to_decision_s=ttd,
            t0=t0,
            t1=t1,
        )

    n_real = sum(1 for v in votes if v == "REAL")
    n_spoof = sum(1 for v in votes if v == "SPOOF")
    frac = n_real / float(n_valid)
    if (
        frac >= required_fraction
        and mean_r is not None
        and mean_r >= real_min
        and n_valid >= MIN_VALID_FOR_REAL
    ):
        state = "REAL"
    elif n_spoof > 0 and n_real == 0:
        # Sustained non-REAL with at least one confident SPOOF.
        state = "SPOOF"
    elif n_spoof >= n_real and n_spoof > 0:
        state = "SPOOF"
    else:
        state = "UNCERTAIN"

    return WindowDecision(
        state=state,
        method="conservative_sustained",
        window_s=window_s,
        n_frames=len(frames),
        n_valid=n_valid,
        exclusions=dict(exclusions),
        mean_real=mean_r,
        median_real=med_r,
        time_to_decision_s=ttd,
        t0=t0,
        t1=t1,
    )


def _window_timing(
    frames: list[dict[str, Any]],
) -> tuple[str | None, str | None, float | None]:
    times: list[datetime] = []
    for ev in frames:
        ts = parse_timestamp(ev.get("timestamp_utc"))
        if ts is not None:
            times.append(ts)
    if not times:
        return None, None, None
    t0, t1 = min(times), max(times)
    return (
        t0.isoformat(),
        t1.isoformat(),
        (t1 - t0).total_seconds(),
    )


WINDOW_METHODS: dict[str, Callable[..., WindowDecision]] = {
    "mean_real": decide_mean_real,
    "median_real": decide_median_real,
    "majority_vote": decide_majority_vote,
    "conservative_sustained": decide_conservative_sustained,
}


def assign_nonoverlapping_windows(
    evals: list[dict[str, Any]],
    window_s: float,
) -> list[list[dict[str, Any]]]:
    """Bin marked evals into non-overlapping wall-clock windows.

    Anchor is the first timestamped eval. Frames without timestamps are dropped
    from windowing (reported as a limitation, not invented).
    """
    stamped: list[tuple[datetime, dict[str, Any]]] = []
    for ev in evals:
        ts = parse_timestamp(ev.get("timestamp_utc"))
        if ts is None:
            continue
        stamped.append((ts, ev))
    if not stamped:
        return []
    stamped.sort(key=lambda x: x[0])
    t_anchor = stamped[0][0]
    buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ts, ev in stamped:
        idx = int((ts - t_anchor).total_seconds() // window_s)
        buckets[idx].append(ev)
    return [buckets[i] for i in sorted(buckets)]


@dataclass(frozen=True)
class MarkedInterval:
    """One continuous MARKED=on segment between recording toggles."""

    index: int
    t_start: datetime
    t_end: datetime
    evals: list[dict[str, Any]]
    included: bool
    exclude_reason: str | None


def marked_intervals_from_session(
    session: dict[str, Any],
) -> list[MarkedInterval]:
    """Split marked evals by mark_event toggles (never cross boundaries)."""
    marks = sorted(
        (m for m in (session.get("mark_events") or []) if m.get("type") == "mark_event"),
        key=lambda m: m.get("timestamp_utc") or "",
    )
    evals = marked_evals(session)
    if not marks:
        # Forward-compat: treat all marked evals as one interval.
        times = [parse_timestamp(e.get("timestamp_utc")) for e in evals]
        times = [t for t in times if t is not None]
        if not times:
            return []
        return [
            MarkedInterval(
                index=0,
                t_start=min(times),
                t_end=max(times),
                evals=list(evals),
                included=True,
                exclude_reason=None,
            )
        ]

    ranges: list[tuple[datetime, datetime]] = []
    cur: datetime | None = None
    for m in marks:
        ts = parse_timestamp(m.get("timestamp_utc"))
        if ts is None:
            continue
        if m.get("marked") is True:
            cur = ts
        elif m.get("marked") is False and cur is not None:
            ranges.append((cur, ts))
            cur = None

    intervals: list[MarkedInterval] = []
    for i, (a, b) in enumerate(ranges):
        subset = []
        for e in evals:
            ts = parse_timestamp(e.get("timestamp_utc"))
            if ts is None:
                continue
            if a <= ts < b:
                subset.append(e)
        intervals.append(
            MarkedInterval(
                index=i,
                t_start=a,
                t_end=b,
                evals=subset,
                included=True,
                exclude_reason=None,
            )
        )
    return intervals


def assign_windows_within_intervals(
    intervals: list[MarkedInterval],
    window_s: float,
) -> list[list[dict[str, Any]]]:
    """Non-overlapping windows per included interval; never cross toggles."""
    out: list[list[dict[str, Any]]] = []
    for iv in intervals:
        if not iv.included:
            continue
        out.extend(assign_nonoverlapping_windows(iv.evals, window_s))
    return out


def evaluate_windows_on_intervals(
    intervals: list[MarkedInterval],
    window_s: float,
    method: str,
    real_min: float,
    spoof_min: float,
    *,
    min_valid: int | None = None,
    exclude_identity_failures: bool = False,
) -> list[WindowDecision]:
    fn = WINDOW_METHODS[method]
    need = min_valid if min_valid is not None else min_valid_for_window(window_s)
    windows = assign_windows_within_intervals(intervals, window_s)
    return [
        fn(
            w,
            real_min,
            spoof_min,
            window_s=window_s,
            min_valid=need,
            exclude_identity_failures=exclude_identity_failures,
        )
        for w in windows
    ]


def session_timing_diagnostics(evals: list[dict[str, Any]]) -> dict[str, Any]:
    times: list[datetime] = []
    missing_ts = 0
    for ev in evals:
        ts = parse_timestamp(ev.get("timestamp_utc"))
        if ts is None:
            missing_ts += 1
        else:
            times.append(ts)
    times.sort()
    gaps: list[float] = []
    for i in range(1, len(times)):
        gaps.append((times[i] - times[i - 1]).total_seconds())
    face = Counter(e.get("face_decision") for e in evals)
    return {
        "n_marked": len(evals),
        "n_with_timestamp": len(times),
        "n_missing_timestamp": missing_ts,
        "span_s": (times[-1] - times[0]).total_seconds() if len(times) > 1 else 0.0,
        "median_gap_s": float(statistics.median(gaps)) if gaps else None,
        "mean_gap_s": float(statistics.mean(gaps)) if gaps else None,
        "max_gap_s": float(max(gaps)) if gaps else None,
        "approx_hz": (1.0 / statistics.median(gaps)) if gaps and statistics.median(gaps) > 0 else None,
        "face_decision_counts": {str(k): v for k, v in face.items()},
        "n_score_usable": sum(1 for e in evals if is_score_usable(e)),
    }


def evaluate_single_frame_session(
    evals: list[dict[str, Any]],
    real_min: float,
    spoof_min: float,
) -> list[WindowDecision]:
    """One 'window' per marked eval (Approach A baseline)."""
    out: list[WindowDecision] = []
    for ev in evals:
        state = approach_a_frame(ev, real_min, spoof_min)
        usable = is_score_usable(ev)
        mean_r = float(ev["fasnet_real_score_avg"]) if usable else None
        excl: dict[str, int] = {}
        reason = frame_exclusion_reason(ev)
        if reason:
            excl[reason] = 1
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


def evaluate_windows_session(
    evals: list[dict[str, Any]],
    window_s: float,
    method: str,
    real_min: float,
    spoof_min: float,
) -> list[WindowDecision]:
    fn = WINDOW_METHODS[method]
    windows = assign_nonoverlapping_windows(evals, window_s)
    return [fn(w, real_min, spoof_min, window_s=window_s) for w in windows]


def _rate(n: int, d: int) -> dict[str, Any]:
    if d <= 0:
        return {"pct": None, "n": n, "d": d, "display": "N/A"}
    pct = 100.0 * n / d
    return {"pct": pct, "n": n, "d": d, "display": f"{pct:.1f}%"}


def summarize_decisions(
    decisions: list[WindowDecision],
    *,
    label: str,
) -> dict[str, Any]:
    states = Counter(d.state for d in decisions)
    n = len(decisions)
    n_real = int(states.get("REAL", 0))
    n_spoof = int(states.get("SPOOF", 0))
    n_unc = int(states.get("UNCERTAIN", 0))
    n_err = int(states.get("ERROR", 0))
    n_inv = int(states.get("INVALID", 0))
    valid_frames = [d.n_valid for d in decisions]
    ttds = [d.time_to_decision_s for d in decisions if d.time_to_decision_s is not None]
    # Latency proxy for windows: nominal window duration (decision after window closes).
    nominal_latency = None
    if decisions and decisions[0].window_s > 0:
        nominal_latency = float(decisions[0].window_s)

    any_spoof_window_real = False
    if label in ("PRINT", "SCREEN"):
        any_spoof_window_real = n_real > 0

    genuine_accept = _rate(n_real, n) if label == "REAL" else None
    print_far = _rate(n_real, n) if label == "PRINT" else None
    screen_far = _rate(n_real, n) if label == "SCREEN" else None

    excl_tot: Counter[str] = Counter()
    for d in decisions:
        excl_tot.update(d.exclusions)

    return {
        "label": label,
        "n_windows": n,
        "state_counts": dict(states),
        "genuine_real_accept": genuine_accept,
        "print_false_accept": print_far,
        "screen_false_accept": screen_far,
        "uncertain_rate": _rate(n_unc, n),
        "spoof_rate": _rate(n_spoof, n),
        "error_rate": _rate(n_err, n),
        "invalid_rate": _rate(n_inv, n),
        "real_rate": _rate(n_real, n),
        "n_invalid_windows": n_inv,
        "mean_valid_frames_per_window": float(statistics.mean(valid_frames)) if valid_frames else 0.0,
        "median_valid_frames_per_window": float(statistics.median(valid_frames)) if valid_frames else 0.0,
        "mean_time_to_decision_s": float(statistics.mean(ttds)) if ttds else None,
        "nominal_decision_latency_s": nominal_latency,
        "exclusion_frame_totals": dict(excl_tot),
        "any_spoof_window_receives_real": any_spoof_window_real if label in ("PRINT", "SCREEN") else None,
        "attack_interval_contains_real": any_spoof_window_real if label in ("PRINT", "SCREEN") else None,
        "note": (
            "Window counts are temporally dependent within a session — "
            "not independent attack trials or subjects."
        ),
    }


def feasibility_report(sessions: list[dict[str, Any]]) -> dict[str, Any]:
    per = []
    limitations: list[str] = []
    for s in sessions:
        ev = marked_evals(s)
        diag = session_timing_diagnostics(ev)
        diag["session_id"] = s["meta"]["session_id"]
        diag["label"] = s["meta"]["label"]
        per.append(diag)
        if diag["n_missing_timestamp"]:
            limitations.append(
                f"{s['meta']['session_id']}: {diag['n_missing_timestamp']} marked evals lack timestamp_utc."
            )
        if diag["approx_hz"] is not None and diag["approx_hz"] < 1.0:
            limitations.append(
                f"{s['meta']['session_id']}: sparse sampling (~{diag['approx_hz']:.2f} Hz); "
                "3s windows may contain only 1–2 scored frames."
            )
        if diag["max_gap_s"] is not None and diag["max_gap_s"] > 2.0:
            limitations.append(
                f"{s['meta']['session_id']}: max inter-eval gap {diag['max_gap_s']:.2f}s "
                "(NO_FACE / pause / slow inference) — wall-clock bins are not continuous face tracks."
            )

    limitations.extend(
        [
            "No face-track / identity continuity IDs in JSONL — windows group by wall clock only.",
            "Sampling rate differs across sessions (~0.6 Hz early REAL vs ~5 Hz later REAL vs ~20 Hz attack).",
            "Non-overlapping windows within one session are correlated; do not treat as independent trials.",
            "NO_FACE / MULTIPLE_FACES frames occupy wall-clock time but contribute no REAL-class scores.",
            "Approach A REAL still requires native argmax REAL + winning-score gate; mean/median methods "
            "use REAL-class mass and can diverge from Approach A on the same frames.",
        ]
    )
    return {
        "supported": all(p["n_with_timestamp"] > 0 for p in per),
        "n_data_bearing_sessions": len(sessions),
        "per_session": per,
        "limitations": limitations,
    }


def run_analysis(calibration_dir: Path) -> dict[str, Any]:
    paths = sorted(calibration_dir.glob("session_*.jsonl"))
    sessions = [load_session(p) for p in paths]
    data_bearing = [s for s in sessions if is_data_bearing(s)]
    sel, val, split_info = split_data_bearing_sessions(data_bearing)

    feasibility = feasibility_report(data_bearing)

    methods = ["single_frame_approach_a"] + [
        f"w{int(w)}s_{m}" for w in WINDOW_SECONDS for m in WINDOW_METHODS
    ]

    # results[threshold_name][split][label][method] = summary
    results: dict[str, Any] = {}
    per_session: dict[str, Any] = {}

    for thr_name, real_min, spoof_min in THRESHOLD_CONFIGS:
        thr_block: dict[str, Any] = {"real_min": real_min, "spoof_min": spoof_min, "splits": {}}
        for split_name, split_sessions in (("selection", sel), ("validation", val)):
            by_label_methods: dict[str, dict[str, Any]] = {
                lab: {} for lab in VALID_LABELS
            }
            for s in split_sessions:
                label = s["meta"]["label"]
                sid = s["meta"]["session_id"]
                evals = marked_evals(s)
                sess_key = f"{thr_name}|{sid}"
                per_session.setdefault(sess_key, {
                    "session_id": sid,
                    "label": label,
                    "split": split_name,
                    "threshold": thr_name,
                    "methods": {},
                })

                # Single-frame baseline
                sf = evaluate_single_frame_session(evals, real_min, spoof_min)
                by_label_methods[label].setdefault("single_frame_approach_a", []).extend(sf)
                per_session[sess_key]["methods"]["single_frame_approach_a"] = summarize_decisions(
                    sf, label=label
                )

                for w in WINDOW_SECONDS:
                    for m_name, m_fn in WINDOW_METHODS.items():
                        key = f"w{int(w)}s_{m_name}"
                        decs = evaluate_windows_session(
                            evals, w, m_name, real_min, spoof_min
                        )
                        by_label_methods[label].setdefault(key, []).extend(decs)
                        per_session[sess_key]["methods"][key] = summarize_decisions(
                            decs, label=label
                        )

            split_out: dict[str, Any] = {"by_label": {}, "sessions": [
                s["meta"]["session_id"] for s in split_sessions
            ]}
            for lab in VALID_LABELS:
                split_out["by_label"][lab] = {
                    method: summarize_decisions(decs, label=lab)
                    for method, decs in by_label_methods[lab].items()
                }
            thr_block["splits"][split_name] = split_out
        results[thr_name] = thr_block

    # Security improvement assessment (descriptive, not statistical claim)
    assessment = _security_assessment(results)

    return {
        "unit": (
            "Marked [REC] evals only; non-overlapping wall-clock windows; "
            "session-level selection/validation. Correlated windows ≠ independent trials."
        ),
        "thresholds": list(THRESHOLD_CONFIGS),
        "window_seconds": list(WINDOW_SECONDS),
        "methods": methods,
        "min_valid_for_real": MIN_VALID_FOR_REAL,
        "conservative_real_fraction": CONSERVATIVE_REAL_FRACTION,
        "feasibility": feasibility,
        "split_info": split_info,
        "results": results,
        "per_session": per_session,
        "security_assessment": assessment,
        "policy_note": (
            "Offline research only. Thresholds remain RESEARCH_CANDIDATE_UNVALIDATED. "
            "Live authentication / Challenge 1 / Challenge 2 unchanged."
        ),
    }


def _security_assessment(results: dict[str, Any]) -> dict[str, Any]:
    """Compare single-frame vs windows on FAR and genuine accept (descriptive)."""
    notes: list[str] = []
    spoof_real_hits: list[str] = []
    recommendations: list[str] = []

    for thr_name, thr_block in results.items():
        for split_name, split_block in thr_block["splits"].items():
            by_label = split_block["by_label"]
            for method in sorted({m for lab in by_label.values() for m in lab}):
                for attack in ("PRINT", "SCREEN"):
                    summ = by_label.get(attack, {}).get(method)
                    if not summ:
                        continue
                    if summ.get("any_spoof_window_receives_real"):
                        spoof_real_hits.append(
                            f"{thr_name}/{split_name}/{attack}/{method}: "
                            f"{summ['real_rate']['display']} windows REAL"
                        )

    if spoof_real_hits:
        notes.append(
            "At least one spoof-labeled window/frame receives REAL under some method/threshold."
        )
    else:
        notes.append(
            "No evaluated method assigned REAL to any PRINT/SCREEN window under tested gates "
            "(may still be UNCERTAIN rather than SPOOF)."
        )

    # Prefer methods that never REAL spoof windows and retain some genuine REAL.
    notes.append(
        "Temporal bins smooth score jitter; security gain requires eliminating false REAL "
        "without relying on correlated multi-window counts as independent evidence."
    )
    recommendations.append(
        "If single-frame and window methods show similar spoof REAL rates, prefer single-frame "
        "or conservative_sustained for lower latency; longer windows mainly delay decisions."
    )
    return {
        "spoof_real_hits": spoof_real_hits,
        "notes": notes,
        "recommendations": recommendations,
    }


def _fmt_rate(r: dict[str, Any] | None) -> str:
    if r is None:
        return "—"
    return r.get("display", "N/A")


def print_comparison_tables(report: dict[str, Any]) -> None:
    print("=" * 88)
    print("FasNet temporal aggregation (offline, marked evals, Approach A baseline)")
    print("=" * 88)
    feas = report["feasibility"]
    print(f"\nTimestamp feasibility: supported={feas['supported']}")
    print(f"Data-bearing sessions: {feas['n_data_bearing_sessions']}")
    for p in feas["per_session"]:
        print(
            f"  {p['session_id']}: n={p['n_marked']} span={p['span_s']:.1f}s "
            f"med_gap={p['median_gap_s']}s ~{p['approx_hz']} Hz "
            f"usable_scores={p['n_score_usable']} face={p['face_decision_counts']}"
        )
    print("\nLimitations:")
    for lim in feas["limitations"]:
        print(f"  - {lim}")

    split = report["split_info"]
    print("\nSession split:")
    for lab, info in split.get("by_label", {}).items():
        print(
            f"  {lab}: sel={info.get('selection')} val={info.get('validation')} "
            f"(marked sel={info.get('selection_marked_evals')} val={info.get('validation_marked_evals')})"
        )
    for lim in split.get("limitations") or []:
        print(f"  split-limitation: {lim}")

    for thr_name, thr_block in report["results"].items():
        print("\n" + "-" * 88)
        print(f"Thresholds {thr_name} (real_min={thr_block['real_min']}, spoof_min={thr_block['spoof_min']})")
        print("-" * 88)
        for split_name, split_block in thr_block["splits"].items():
            print(f"\n### {split_name.upper()}  sessions={split_block['sessions']}")
            methods = sorted({m for lab in split_block["by_label"].values() for m in lab})
            # Preferred column order
            prefer = ["single_frame_approach_a"] + [
                f"w{int(w)}s_{m}"
                for w in WINDOW_SECONDS
                for m in (
                    "mean_real",
                    "median_real",
                    "majority_vote",
                    "conservative_sustained",
                )
            ]
            methods = [m for m in prefer if m in methods] + [
                m for m in methods if m not in prefer
            ]

            header = (
                f"{'method':<36} {'REAL_AR':>8} {'PRINT_FAR':>10} {'SCREEN_FAR':>11} "
                f"{'UNC%':>7} {'SPF%':>7} {'n_win':>6} {'valid/w':>8} {'ttd_s':>7} {'spoofREAL?':>10}"
            )
            print(header)
            print("-" * len(header))
            for method in methods:
                real_s = split_block["by_label"].get("REAL", {}).get(method)
                print_s = split_block["by_label"].get("PRINT", {}).get(method)
                screen_s = split_block["by_label"].get("SCREEN", {}).get(method)
                # Use REAL summary for timing/valid when present, else first available
                base = real_s or print_s or screen_s or {}
                n_win = base.get("n_windows", 0)
                valid_w = base.get("mean_valid_frames_per_window", 0.0)
                ttd = base.get("mean_time_to_decision_s")
                ttd_s = f"{ttd:.2f}" if ttd is not None else "N/A"
                spoof_hit = "YES" if (
                    (print_s and print_s.get("any_spoof_window_receives_real"))
                    or (screen_s and screen_s.get("any_spoof_window_receives_real"))
                ) else "no"
                # Aggregate UNCERTAIN/SPOOF across labels for display: show REAL-label rates
                unc = _fmt_rate(real_s.get("uncertain_rate") if real_s else None)
                spf = _fmt_rate(real_s.get("spoof_rate") if real_s else None)
                # For attack FAR use attack summaries; for UNCERTAIN show combined note via REAL row
                # Better: show unc/spf as PRINT+SCREEN pooled? Task asks rates per method —
                # show REAL AR from REAL, FARs from attacks, and unc/spf from REAL sessions
                # plus mention attack unc in spoofREAL column.
                print(
                    f"{method:<36} "
                    f"{_fmt_rate(real_s.get('genuine_real_accept') if real_s else None):>8} "
                    f"{_fmt_rate(print_s.get('print_false_accept') if print_s else None):>10} "
                    f"{_fmt_rate(screen_s.get('screen_false_accept') if screen_s else None):>11} "
                    f"{unc:>7} {spf:>7} {n_win:>6} {valid_w:>8.1f} {ttd_s:>7} {spoof_hit:>10}"
                )

            # Per-session breakdown (compact)
            print("\n  Per-session REAL rates (state REAL / n_windows):")
            for s in split_block["sessions"]:
                for key, block in report["per_session"].items():
                    if block["session_id"] != s or block["threshold"] != thr_name:
                        continue
                    for method in ("single_frame_approach_a", "w3s_conservative_sustained",
                                   "w5s_conservative_sustained", "w10s_conservative_sustained",
                                   "w5s_mean_real", "w5s_majority_vote"):
                        msum = block["methods"].get(method)
                        if not msum:
                            continue
                        rr = msum["real_rate"]["display"]
                        print(
                            f"    {s} [{block['label']}] {method}: REAL={rr} "
                            f"UNC={msum['uncertain_rate']['display']} "
                            f"SPOOF={msum['spoof_rate']['display']} "
                            f"n={msum['n_windows']} valid/w={msum['mean_valid_frames_per_window']:.1f}"
                        )

    print("\n" + "=" * 88)
    print("Security assessment")
    print("=" * 88)
    for hit in report["security_assessment"]["spoof_real_hits"][:40]:
        print(f"  HIT: {hit}")
    if not report["security_assessment"]["spoof_real_hits"]:
        print("  (no spoof→REAL hits listed)")
    for n in report["security_assessment"]["notes"]:
        print(f"  note: {n}")
    for r in report["security_assessment"]["recommendations"]:
        print(f"  rec: {r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=DEFAULT_CALIBRATION_DIR,
        help="Directory of session_*.jsonl recordings",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write full JSON report",
    )
    args = parser.parse_args()
    report = run_analysis(args.calibration_dir)
    print_comparison_tables(report)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
