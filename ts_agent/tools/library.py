"""Tool library — history reconditioning, foundation forecast stub, future enforcement."""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# History reconditioning ops (Πhist)
# ---------------------------------------------------------------------------

def apply_hist_ops(series: list[float], ops: list[dict]) -> list[float]:
    arr = np.array(series, dtype=float)
    for op in ops:
        name = op.get("op", "")
        p = op.get("params", {})
        if name == "clip_outliers":
            lo = np.percentile(arr, p.get("lo_pct", 1))
            hi = np.percentile(arr, p.get("hi_pct", 99))
            arr = np.clip(arr, lo, hi)
        elif name == "impute_missing":
            nans = np.isnan(arr)
            if nans.any() and (~nans).any():
                arr[nans] = np.interp(
                    np.where(nans)[0], np.where(~nans)[0], arr[~nans]
                )
        elif name == "scale":
            arr = arr * p.get("factor", 1.0) + p.get("offset", 0.0)
        elif name == "apply_event_effect":
            idx = int(p.get("index", -1))
            if 0 <= idx < len(arr):
                arr[idx] *= p.get("multiplier", 1.0)
        elif name == "inject_pattern":
            pattern = np.array(p.get("pattern", []))
            start = int(p.get("start_index", 0))
            end = min(start + len(pattern), len(arr))
            arr[start:end] = pattern[: end - start]
    return arr.tolist()


# ---------------------------------------------------------------------------
# Foundation forecast (Πfm)  — stub; replace with real model API call
# ---------------------------------------------------------------------------

def run_foundation_forecast(series: list[float], spec: dict) -> list[float]:
    """
    Placeholder: seasonal-naive baseline.
    Replace this function body with an actual Moirai / Chronos / Lag-LLaMA call.
    """
    horizon = int(spec.get("horizon", 12))
    season = int(spec.get("season", max(1, len(series) // 4)))
    return [series[-(season - i % season)] for i in range(horizon)]


# ---------------------------------------------------------------------------
# Future enforcement ops (Πfuture)
# ---------------------------------------------------------------------------

def apply_future_ops(forecast: list[float], ops: list[dict]) -> list[float]:
    arr = np.array(forecast, dtype=float)
    for op in ops:
        name = op.get("op", "")
        p = op.get("params", {})
        if name == "cap_at":
            arr = np.minimum(arr, p.get("value", np.inf))
        elif name == "floor_at":
            arr = np.maximum(arr, p.get("value", -np.inf))
        elif name == "apply_trend":
            arr = arr + np.arange(len(arr)) * p.get("slope", 0.0)
        elif name == "enforce_event_spike":
            idx = int(p.get("index", 0))
            if 0 <= idx < len(arr):
                arr[idx] *= p.get("multiplier", 1.5)
        elif name == "enforce_event_dip":
            idx = int(p.get("index", 0))
            if 0 <= idx < len(arr):
                arr[idx] *= p.get("multiplier", 0.5)
    return arr.tolist()
