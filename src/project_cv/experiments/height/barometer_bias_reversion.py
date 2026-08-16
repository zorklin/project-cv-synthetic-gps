"""Causal speed-aware barometer correction with mean-reverting states.

The original random-walk estimator can preserve an obsolete correction late
in a flight.  This variant uses an Ornstein-Uhlenbeck-style prediction: slow
bias and quadratic speed gain decay toward zero between sparse Starlink
updates, while process noise still allows both states to move when supported
by new measurements.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np
import pandas as pd

from project_cv.experiments.height.barometer_bias import (
    BarometerBiasResult,
    RobustBiasSpeedConfig,
    _causal_horizontal_speed,
    _prepare_inputs,
)


@dataclass(frozen=True, slots=True)
class RevertingBiasSpeedConfig(RobustBiasSpeedConfig):
    """Tuned causal model with independently reverting bias and speed gain."""

    bias_random_walk_m_per_sqrt_sec: float = 0.2
    bias_mean_reversion_tau_sec: float = 60.0
    speed_gain_mean_reversion_tau_sec: float = 240.0

    def __post_init__(self) -> None:
        RobustBiasSpeedConfig.__post_init__(self)
        for name in (
            "bias_mean_reversion_tau_sec",
            "speed_gain_mean_reversion_tau_sec",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and greater than zero")


def apply_causal_reverting_bias_speed(
    barometer: pd.DataFrame,
    starlink: pd.DataFrame,
    vision_velocity: pd.DataFrame,
    *,
    config: RevertingBiasSpeedConfig | None = None,
) -> BarometerBiasResult:
    """Correct height using robust Starlink updates and mean-reverting states."""

    selected = config or RevertingBiasSpeedConfig()
    baro, sparse = _prepare_inputs(barometer, starlink)
    vision_time, vision_speed = _causal_horizontal_speed(
        vision_velocity,
        config=selected,
    )
    baro_time = baro["tr"].to_numpy(dtype=float)
    baro_height = baro["u_graph"].to_numpy(dtype=float)
    sparse_time = sparse["tr_corr"].to_numpy(dtype=float)
    sparse_height = sparse["u_corr"].to_numpy(dtype=float)

    state = np.zeros(2, dtype=float)
    covariance = np.diag(
        [selected.initial_bias_sigma_m**2, selected.initial_speed_gain_sigma_m**2]
    )
    process_variance_per_sec = np.array(
        [
            selected.bias_random_walk_m_per_sqrt_sec**2,
            selected.speed_gain_random_walk_m_per_sqrt_sec**2,
        ],
        dtype=float,
    )
    measurement_variance = selected.starlink_sigma_m**2
    state_time = float(baro_time[0])
    sparse_index = int(np.searchsorted(sparse_time, baro_time[0], side="left"))
    correction_series = np.empty(len(baro), dtype=float)
    bias_series = np.empty(len(baro), dtype=float)
    gain_series = np.empty(len(baro), dtype=float)
    speed_series = np.empty(len(baro), dtype=float)
    trace_rows: list[dict[str, float | bool]] = []

    def predict_to(target_time: float) -> None:
        nonlocal covariance, state, state_time
        delta_time = max(0.0, float(target_time) - state_time)
        if delta_time <= 0.0:
            return
        decay = np.array(
            [
                math.exp(-delta_time / selected.bias_mean_reversion_tau_sec),
                math.exp(
                    -delta_time / selected.speed_gain_mean_reversion_tau_sec
                ),
            ],
            dtype=float,
        )
        transition = np.diag(decay)
        state *= decay
        covariance = (
            transition @ covariance @ transition
            + np.diag(process_variance_per_sec * delta_time)
        )
        state_time = float(target_time)

    for baro_index, current_time in enumerate(baro_time):
        while sparse_index < len(sparse_time) and sparse_time[sparse_index] <= current_time:
            event_time = float(sparse_time[sparse_index])
            predict_to(event_time)
            source_index = int(
                np.searchsorted(baro_time, event_time, side="right") - 1
            )
            velocity_index = int(
                np.searchsorted(vision_time, event_time, side="right") - 1
            )
            if velocity_index < 0:
                # No future velocity is borrowed for an early Starlink fix.
                sparse_index += 1
                continue

            speed_mps = float(vision_speed[velocity_index])
            normalized_speed_squared = (
                speed_mps / selected.speed_scale_mps
            ) ** 2
            observation = np.array(
                [1.0, normalized_speed_squared],
                dtype=float,
            )
            measurement_bias = float(
                baro_height[source_index] - sparse_height[sparse_index]
            )
            innovation = measurement_bias - float(observation @ state)
            accepted = abs(innovation) <= selected.innovation_gate_m
            update = np.zeros(2, dtype=float)
            if accepted:
                clipped_innovation = float(
                    np.clip(innovation, -selected.huber_clip_m, selected.huber_clip_m)
                )
                kalman_gain = covariance @ observation / float(
                    observation @ covariance @ observation + measurement_variance
                )
                update = kalman_gain * clipped_innovation
                state += update
                state[0] = float(
                    np.clip(state[0], selected.min_bias_m, selected.max_bias_m)
                )
                state[1] = float(
                    np.clip(
                        state[1],
                        selected.min_speed_gain_m,
                        selected.max_speed_gain_m,
                    )
                )
                covariance = (
                    np.eye(2, dtype=float) - np.outer(kalman_gain, observation)
                ) @ covariance
                covariance = 0.5 * (covariance + covariance.T)

            trace_rows.append(
                {
                    "tr": event_time,
                    "barometer_u_m": float(baro_height[source_index]),
                    "starlink_u_m": float(sparse_height[sparse_index]),
                    "horizontal_speed_mps": speed_mps,
                    "normalized_speed_squared": normalized_speed_squared,
                    "measurement_bias_m": measurement_bias,
                    "innovation_m": innovation,
                    "accepted": accepted,
                    "bias_update_m": float(update[0]),
                    "speed_gain_update_m": float(update[1]),
                    "estimated_bias_m": float(state[0]),
                    "estimated_speed_gain_m": float(state[1]),
                }
            )
            sparse_index += 1

        predict_to(float(current_time))
        velocity_index = int(
            np.searchsorted(vision_time, current_time, side="right") - 1
        )
        current_speed = 0.0 if velocity_index < 0 else float(vision_speed[velocity_index])
        normalized_speed_squared = (
            current_speed / selected.speed_scale_mps
        ) ** 2
        correction_m = float(state[0] + state[1] * normalized_speed_squared)
        bias_series[baro_index] = state[0]
        gain_series[baro_index] = state[1]
        speed_series[baro_index] = current_speed
        correction_series[baro_index] = correction_m

    result_frame = baro.copy()
    result_frame["u_graph_uncorrected"] = baro_height
    result_frame["estimated_bias_m"] = bias_series
    result_frame["estimated_speed_gain_m"] = gain_series
    result_frame["horizontal_speed_proxy_mps"] = speed_series
    result_frame["estimated_total_error_m"] = correction_series
    result_frame["u_graph"] = baro_height - correction_series
    trace = pd.DataFrame(trace_rows)
    accepted_count = int(trace["accepted"].sum()) if len(trace) else 0
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "estimator": "robust_mean_reverting_bias_plus_quadratic_speed",
        "causal": True,
        "barometer_sign_convention": (
            "measurement_bias_m = barometer_u_m - starlink_u_m; "
            "corrected_u_m = barometer_u_m - estimated_error_m"
        ),
        "state_updates_use": "sparse_corrected_starlink_altitude_only",
        "dense_gps_used": False,
        "robust_update": (
            "hard innovation gate followed by Huber-clipped innovation"
        ),
        "state_prediction": (
            "independent exponential mean reversion plus random-walk process noise"
        ),
        "config": asdict(selected),
        "processed_starlink_updates": int(len(trace)),
        "accepted_starlink_updates": accepted_count,
        "rejected_starlink_updates": int(len(trace) - accepted_count),
        "speed_source": "causally smoothed optical-flow FRD horizontal speed",
        "speed_model": (
            "gain * (horizontal_speed_mps / speed_scale_mps)^2"
        ),
    }
    return BarometerBiasResult(frame=result_frame, trace=trace, metadata=metadata)
