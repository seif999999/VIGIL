"""Frozen FasNet temporal PAD for LOCAL RESEARCH end-to-end testing.

Reuses offline ``decide_median_real`` (5s / median REAL-class mass / 0.95 / 0.85 /
min_valid=5). Does not change default live authentication or activate production
thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from analyze_fasnet_temporal_aggregation import (
    WindowDecision,
    decide_median_real,
    min_valid_for_window,
)

# Frozen research PAD (offline continuity / sliding studies).
FROZEN_PAD_WINDOW_S = 5.0
FROZEN_PAD_REAL_MIN = 0.95
FROZEN_PAD_SPOOF_MIN = 0.85
FROZEN_PAD_MIN_VALID = min_valid_for_window(FROZEN_PAD_WINDOW_S)  # 5
FROZEN_PAD_METHOD = "median_real"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def eval_from_observation(
    *,
    face_decision: str,
    fasnet_native_class: str | None,
    fasnet_real_score_avg: float | None,
    fasnet_spoof_score_combined: float | None,
    fasnet_error: str | None,
    timestamp_utc: str | None = None,
) -> dict[str, Any]:
    """Shape compatible with offline ``decide_median_real`` / ``is_score_usable``."""
    return {
        "timestamp_utc": timestamp_utc or utc_now_iso(),
        "face_decision": face_decision,
        "fasnet_native_class": fasnet_native_class,
        "fasnet_real_score_avg": fasnet_real_score_avg,
        "fasnet_spoof_score_combined": fasnet_spoof_score_combined,
        "fasnet_error": fasnet_error,
        # Identity must not gate PAD (exclude_identity_failures=False default).
        "identity_recognized": None,
        "facenet_error": None,
    }


@dataclass
class FrozenPadWindowBuffer:
    """Non-overlapping wall-clock windows of FasNet observations."""

    window_s: float = FROZEN_PAD_WINDOW_S
    real_min: float = FROZEN_PAD_REAL_MIN
    spoof_min: float = FROZEN_PAD_SPOOF_MIN
    min_valid: int = FROZEN_PAD_MIN_VALID
    frames: list[dict[str, Any]] = field(default_factory=list)
    window_t0_perf: float | None = None

    def reset(self) -> None:
        self.frames.clear()
        self.window_t0_perf = None

    def add(
        self,
        *,
        perf_t: float,
        eval_row: dict[str, Any],
    ) -> WindowDecision | None:
        """Append one observation. Return a decision when the window closes."""
        if self.window_t0_perf is None:
            self.window_t0_perf = perf_t
        self.frames.append(eval_row)
        elapsed = perf_t - self.window_t0_perf
        if elapsed < self.window_s:
            return None
        decision = self.decide_current()
        self.reset()
        return decision

    def decide_current(self) -> WindowDecision:
        return decide_median_real(
            list(self.frames),
            self.real_min,
            self.spoof_min,
            window_s=self.window_s,
            min_valid=self.min_valid,
            exclude_identity_failures=False,
        )

    @property
    def progress_s(self) -> float:
        if self.window_t0_perf is None:
            return 0.0
        return 0.0  # caller should pass now; kept for HUD via elapsed()

    def elapsed_s(self, perf_now: float) -> float:
        if self.window_t0_perf is None:
            return 0.0
        return max(0.0, perf_now - self.window_t0_perf)

    def n_frames(self) -> int:
        return len(self.frames)
