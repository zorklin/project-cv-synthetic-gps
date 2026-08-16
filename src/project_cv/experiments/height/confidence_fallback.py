"""Causal confidence and fallback policy for barometer bias correction.

The underlying estimator is left unchanged.  This safety layer decides how
much of its proposed correction may reach the graph using only information
available online: accepted sparse updates, innovation size, recent bias
consistency, update age and correction magnitude.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class ConfidencePolicyConfig:
    name: str
    enabled: bool = True
    min_consistent_updates: int = 3
    consistency_window_updates: int = 5
    max_abs_innovation_m: float = 5.0
    max_bias_mad_m: float = 2.0
    max_abs_correction_m: float = 2.5
    max_update_age_sec: float = 20.0
    confidence_rise_tau_sec: float = 10.0
    confidence_fall_tau_sec: float = 3.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must not be empty")
        if self.min_consistent_updates < 1:
            raise ValueError("min_consistent_updates must be at least one")
        if self.consistency_window_updates < self.min_consistent_updates:
            raise ValueError(
                "consistency_window_updates must cover min_consistent_updates"
            )
        for name in (
            "max_abs_innovation_m",
            "max_bias_mad_m",
            "max_abs_correction_m",
            "max_update_age_sec",
            "confidence_rise_tau_sec",
            "confidence_fall_tau_sec",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and greater than zero")


POLICY_CANDIDATES = (
    ConfidencePolicyConfig(name="fallback_only", enabled=False),
    ConfidencePolicyConfig(
        name="strict",
        min_consistent_updates=4,
        consistency_window_updates=5,
        max_abs_innovation_m=3.0,
        max_bias_mad_m=1.5,
        max_abs_correction_m=2.0,
        confidence_rise_tau_sec=15.0,
    ),
    ConfidencePolicyConfig(
        name="conservative",
        min_consistent_updates=3,
        consistency_window_updates=5,
        max_abs_innovation_m=5.0,
        max_bias_mad_m=2.0,
        max_abs_correction_m=2.5,
        confidence_rise_tau_sec=10.0,
    ),
    ConfidencePolicyConfig(
        name="balanced",
        min_consistent_updates=3,
        consistency_window_updates=4,
        max_abs_innovation_m=5.0,
        max_bias_mad_m=3.0,
        max_abs_correction_m=3.0,
        confidence_rise_tau_sec=7.5,
    ),
    ConfidencePolicyConfig(
        name="responsive",
        min_consistent_updates=2,
        consistency_window_updates=3,
        max_abs_innovation_m=8.0,
        max_bias_mad_m=4.0,
        max_abs_correction_m=4.0,
        confidence_rise_tau_sec=5.0,
    ),
)


def _approach(current: float, target: float, delta_time: float, tau: float) -> float:
    if delta_time <= 0.0:
        return current
    weight = 1.0 - math.exp(-delta_time / tau)
    return current + weight * (target - current)


def causal_confidence_series(
    sample_time: np.ndarray,
    update_trace: pd.DataFrame,
    *,
    config: ConfidencePolicyConfig,
) -> np.ndarray:
    """Return a causal confidence in ``[0, 1]`` for every sample time."""

    times = np.asarray(sample_time, dtype=float)
    if times.ndim != 1 or np.any(~np.isfinite(times)) or np.any(np.diff(times) < 0):
        raise ValueError("sample_time must be a finite non-decreasing vector")
    if not config.enabled:
        return np.zeros(len(times), dtype=float)
    required = {"tr", "accepted", "innovation_m", "measurement_bias_m"}
    missing = required.difference(update_trace.columns)
    if missing:
        raise ValueError(f"update_trace is missing columns: {sorted(missing)}")

    trace = update_trace.sort_values("tr", kind="stable").reset_index(drop=True)
    event_time = trace["tr"].to_numpy(dtype=float)
    accepted = trace["accepted"].astype(bool).to_numpy()
    innovation = trace["innovation_m"].to_numpy(dtype=float)
    measurement_bias = trace["measurement_bias_m"].to_numpy(dtype=float)
    confidence = np.zeros(len(times), dtype=float)
    event_index = 0
    current = 0.0
    target = 0.0
    previous_time = float(times[0]) if len(times) else 0.0
    last_trusted_time = -math.inf
    consistent_bias: list[float] = []

    for sample_index, current_time in enumerate(times):
        while event_index < len(event_time) and event_time[event_index] <= current_time:
            next_time = float(event_time[event_index])
            tau = (
                config.confidence_rise_tau_sec
                if target > current
                else config.confidence_fall_tau_sec
            )
            current = _approach(current, target, next_time - previous_time, tau)
            previous_time = next_time

            event_is_trustworthy = bool(accepted[event_index]) and (
                abs(float(innovation[event_index])) <= config.max_abs_innovation_m
            )
            if event_is_trustworthy:
                consistent_bias.append(float(measurement_bias[event_index]))
                consistent_bias = consistent_bias[-config.consistency_window_updates :]
                recent = np.asarray(consistent_bias, dtype=float)
                center = float(np.median(recent))
                mad = float(np.median(np.abs(recent - center)))
                enough = len(recent) >= config.min_consistent_updates
                stable = enough and mad <= config.max_bias_mad_m
                target = 1.0 if stable else 0.0
                if stable:
                    last_trusted_time = next_time
            else:
                consistent_bias.clear()
                target = 0.0
            event_index += 1

        if current_time - last_trusted_time > config.max_update_age_sec:
            target = 0.0
        tau = (
            config.confidence_rise_tau_sec
            if target > current
            else config.confidence_fall_tau_sec
        )
        current = _approach(current, target, float(current_time) - previous_time, tau)
        previous_time = float(current_time)
        confidence[sample_index] = float(np.clip(current, 0.0, 1.0))
    return confidence


def apply_confidence_to_correction(
    sample_time: np.ndarray,
    proposed_correction_m: np.ndarray,
    update_trace: pd.DataFrame,
    *,
    config: ConfidencePolicyConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    proposed = np.asarray(proposed_correction_m, dtype=float)
    if proposed.shape != np.asarray(sample_time).shape:
        raise ValueError("proposed correction and sample time must have equal shape")
    confidence = causal_confidence_series(sample_time, update_trace, config=config)
    limited = np.clip(
        proposed,
        -config.max_abs_correction_m,
        config.max_abs_correction_m,
    )
    applied = confidence * limited
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "causal": True,
        "fallback": "zero correction (raw-start-reset barometer)",
        "config": asdict(config),
        "mean_confidence": float(np.mean(confidence)) if len(confidence) else 0.0,
        "active_fraction": float(np.mean(confidence > 0.05)) if len(confidence) else 0.0,
        "mean_abs_applied_correction_m": (
            float(np.mean(np.abs(applied))) if len(applied) else 0.0
        ),
        "max_abs_applied_correction_m": (
            float(np.max(np.abs(applied))) if len(applied) else 0.0
        ),
    }
    return applied, metadata
