"""Causal barometer-bias estimators driven by sparse Starlink altitude.

The estimators in this module modify only the barometer measurement supplied
to the fusion graph.  Dense GPS is intentionally absent from every API here:
it may be used later to score an experiment, but never to construct the
corrected graph input.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class RobustBiasConfig:
    """Configuration for a one-state, slowly varying height bias."""

    innovation_gate_m: float = 6.0
    huber_clip_m: float = 3.0
    time_constant_sec: float = 30.0
    nominal_first_update_dt_sec: float = 10.0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and greater than zero")


@dataclass(frozen=True, slots=True)
class RobustBiasSpeedConfig:
    """Configuration for slow bias plus a dynamic-pressure speed term.

    The second state multiplies ``(horizontal_speed / speed_scale)^2``.  This
    follows the expected quadratic dependence of dynamic pressure on speed.
    Both states are updated only when a robustly accepted Starlink altitude is
    available.
    """

    innovation_gate_m: float = 10.0
    huber_clip_m: float = 3.0
    starlink_sigma_m: float = 1.5
    bias_random_walk_m_per_sqrt_sec: float = 0.1
    speed_gain_random_walk_m_per_sqrt_sec: float = 0.02
    initial_bias_sigma_m: float = 4.0
    initial_speed_gain_sigma_m: float = 4.0
    nominal_first_update_dt_sec: float = 10.0
    speed_scale_mps: float = 10.0
    speed_median_window_samples: int = 25
    speed_mean_window_samples: int = 25
    min_bias_m: float = -8.0
    max_bias_m: float = 8.0
    min_speed_gain_m: float = 0.0
    max_speed_gain_m: float = 12.0

    def __post_init__(self) -> None:
        positive = (
            "innovation_gate_m",
            "huber_clip_m",
            "starlink_sigma_m",
            "initial_bias_sigma_m",
            "initial_speed_gain_sigma_m",
            "nominal_first_update_dt_sec",
            "speed_scale_mps",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and greater than zero")
        for name in (
            "bias_random_walk_m_per_sqrt_sec",
            "speed_gain_random_walk_m_per_sqrt_sec",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.speed_median_window_samples < 1 or self.speed_mean_window_samples < 1:
            raise ValueError("speed smoothing windows must be at least one sample")
        if self.max_bias_m <= self.min_bias_m:
            raise ValueError("max_bias_m must exceed min_bias_m")
        if self.max_speed_gain_m <= self.min_speed_gain_m:
            raise ValueError("max_speed_gain_m must exceed min_speed_gain_m")


@dataclass(slots=True)
class BarometerBiasResult:
    """Corrected graph input, sparse update trace and provenance metadata."""

    frame: pd.DataFrame
    trace: pd.DataFrame
    metadata: dict[str, Any]


def _finite_unique_series(
    frame: pd.DataFrame,
    *,
    time_column: str,
    value_columns: tuple[str, ...],
    label: str,
) -> pd.DataFrame:
    required = {time_column, *value_columns}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")
    columns = [time_column, *value_columns]
    result = frame.loc[:, columns].copy()
    for column in columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    finite = np.ones(len(result), dtype=bool)
    for column in columns:
        finite &= np.isfinite(result[column].to_numpy(dtype=float))
    result = result.loc[finite]
    result = result.sort_values(time_column, kind="mergesort")
    result = result.drop_duplicates(subset=[time_column], keep="first")
    result = result.reset_index(drop=True)
    if not len(result):
        raise ValueError(f"{label} has no finite samples")
    times = result[time_column].to_numpy(dtype=float)
    if len(times) > 1 and not np.all(np.diff(times) > 0.0):
        raise ValueError(f"{label} timestamps are not strictly increasing")
    return result


def _prepare_inputs(
    barometer: pd.DataFrame,
    starlink: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    baro = _finite_unique_series(
        barometer,
        time_column="tr",
        value_columns=("u_graph",),
        label="barometer graph input",
    )
    sparse = _finite_unique_series(
        starlink,
        time_column="tr_corr",
        value_columns=("u_corr",),
        label="corrected Starlink",
    )
    return baro, sparse


def _result_metadata(
    *,
    estimator: str,
    config: RobustBiasConfig | RobustBiasSpeedConfig,
    trace: pd.DataFrame,
) -> dict[str, Any]:
    accepted = int(trace["accepted"].sum()) if len(trace) else 0
    return {
        "schema_version": 1,
        "estimator": estimator,
        "causal": True,
        "barometer_sign_convention": (
            "measurement_bias_m = barometer_u_m - starlink_u_m; "
            "corrected_u_m = barometer_u_m - estimated_error_m"
        ),
        "state_updates_use": "sparse_corrected_starlink_altitude_only",
        "dense_gps_used": False,
        "robust_update": "hard innovation gate followed by Huber-clipped innovation",
        "config": asdict(config),
        "processed_starlink_updates": int(len(trace)),
        "accepted_starlink_updates": accepted,
        "rejected_starlink_updates": int(len(trace) - accepted),
    }


def apply_causal_robust_bias(
    barometer: pd.DataFrame,
    starlink: pd.DataFrame,
    *,
    config: RobustBiasConfig | None = None,
) -> BarometerBiasResult:
    """Subtract a causal one-state EWMA bias estimated from sparse Starlink."""

    selected = config or RobustBiasConfig()
    baro, sparse = _prepare_inputs(barometer, starlink)
    baro_time = baro["tr"].to_numpy(dtype=float)
    baro_height = baro["u_graph"].to_numpy(dtype=float)
    sparse_time = sparse["tr_corr"].to_numpy(dtype=float)
    sparse_height = sparse["u_corr"].to_numpy(dtype=float)

    bias_m = 0.0
    last_event_time: float | None = None
    sparse_index = int(np.searchsorted(sparse_time, baro_time[0], side="left"))
    bias_series = np.empty(len(baro), dtype=float)
    trace_rows: list[dict[str, float | bool]] = []

    for baro_index, current_time in enumerate(baro_time):
        while sparse_index < len(sparse_time) and sparse_time[sparse_index] <= current_time:
            source_index = int(
                np.searchsorted(baro_time, sparse_time[sparse_index], side="right") - 1
            )
            measurement_bias = float(
                baro_height[source_index] - sparse_height[sparse_index]
            )
            innovation = measurement_bias - bias_m
            accepted = abs(innovation) <= selected.innovation_gate_m
            delta_time = (
                selected.nominal_first_update_dt_sec
                if last_event_time is None
                else max(0.0, float(sparse_time[sparse_index] - last_event_time))
            )
            gain = 1.0 - math.exp(-delta_time / selected.time_constant_sec)
            update_m = 0.0
            if accepted:
                update_m = gain * float(
                    np.clip(innovation, -selected.huber_clip_m, selected.huber_clip_m)
                )
                bias_m += update_m
            trace_rows.append(
                {
                    "tr": float(sparse_time[sparse_index]),
                    "barometer_u_m": float(baro_height[source_index]),
                    "starlink_u_m": float(sparse_height[sparse_index]),
                    "measurement_bias_m": measurement_bias,
                    "innovation_m": innovation,
                    "accepted": accepted,
                    "gain": gain,
                    "update_m": update_m,
                    "estimated_bias_m": bias_m,
                }
            )
            last_event_time = float(sparse_time[sparse_index])
            sparse_index += 1
        bias_series[baro_index] = bias_m

    result_frame = baro.copy()
    result_frame["u_graph_uncorrected"] = baro_height
    result_frame["estimated_bias_m"] = bias_series
    result_frame["u_graph"] = baro_height - bias_series
    trace = pd.DataFrame(trace_rows)
    return BarometerBiasResult(
        frame=result_frame,
        trace=trace,
        metadata=_result_metadata(estimator="robust_slow_bias", config=selected, trace=trace),
    )


def _causal_horizontal_speed(
    vision_velocity: pd.DataFrame,
    *,
    config: RobustBiasSpeedConfig,
) -> tuple[np.ndarray, np.ndarray]:
    vision = _finite_unique_series(
        vision_velocity,
        time_column="tr_corr",
        value_columns=("vx", "vy"),
        label="vision velocity",
    )
    raw_speed = np.hypot(
        vision["vx"].to_numpy(dtype=float),
        vision["vy"].to_numpy(dtype=float),
    )
    smoothed_speed = (
        pd.Series(raw_speed, dtype="float64")
        .rolling(
            window=int(config.speed_median_window_samples),
            center=False,
            min_periods=1,
        )
        .median()
        .rolling(
            window=int(config.speed_mean_window_samples),
            center=False,
            min_periods=1,
        )
        .mean()
        .to_numpy(dtype=float)
    )
    return vision["tr_corr"].to_numpy(dtype=float), smoothed_speed


def apply_causal_robust_bias_speed(
    barometer: pd.DataFrame,
    starlink: pd.DataFrame,
    vision_velocity: pd.DataFrame,
    *,
    config: RobustBiasSpeedConfig | None = None,
) -> BarometerBiasResult:
    """Estimate slow bias and a quadratic speed-dependent pressure error.

    The state is ``[bias_m, gain_m_per_normalized_speed_squared]``.  A small
    Kalman filter predicts both states as random walks.  Sparse Starlink
    updates are hard-gated and Huber-clipped before they can change the state.
    No future velocity or Starlink sample is used.
    """

    selected = config or RobustBiasSpeedConfig()
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
    measurement_variance = selected.starlink_sigma_m**2
    last_event_time: float | None = None
    sparse_index = int(np.searchsorted(sparse_time, baro_time[0], side="left"))
    correction_series = np.empty(len(baro), dtype=float)
    bias_series = np.empty(len(baro), dtype=float)
    gain_series = np.empty(len(baro), dtype=float)
    speed_series = np.empty(len(baro), dtype=float)
    trace_rows: list[dict[str, float | bool]] = []

    for baro_index, current_time in enumerate(baro_time):
        while sparse_index < len(sparse_time) and sparse_time[sparse_index] <= current_time:
            source_index = int(
                np.searchsorted(baro_time, sparse_time[sparse_index], side="right") - 1
            )
            velocity_index = int(
                np.searchsorted(vision_time, sparse_time[sparse_index], side="right") - 1
            )
            # A Starlink fix before the first velocity sample cannot update the
            # speed-aware model without peeking into the future.
            if velocity_index < 0:
                sparse_index += 1
                continue

            speed_mps = float(vision_speed[velocity_index])
            speed_feature = (speed_mps / selected.speed_scale_mps) ** 2
            observation = np.array([1.0, speed_feature], dtype=float)
            measurement_bias = float(
                baro_height[source_index] - sparse_height[sparse_index]
            )
            delta_time = (
                selected.nominal_first_update_dt_sec
                if last_event_time is None
                else max(0.0, float(sparse_time[sparse_index] - last_event_time))
            )
            covariance += np.diag(
                [
                    selected.bias_random_walk_m_per_sqrt_sec**2 * delta_time,
                    selected.speed_gain_random_walk_m_per_sqrt_sec**2 * delta_time,
                ]
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
                        state[1], selected.min_speed_gain_m, selected.max_speed_gain_m
                    )
                )
                covariance = (
                    np.eye(2, dtype=float) - np.outer(kalman_gain, observation)
                ) @ covariance
                covariance = 0.5 * (covariance + covariance.T)

            trace_rows.append(
                {
                    "tr": float(sparse_time[sparse_index]),
                    "barometer_u_m": float(baro_height[source_index]),
                    "starlink_u_m": float(sparse_height[sparse_index]),
                    "horizontal_speed_mps": speed_mps,
                    "normalized_speed_squared": speed_feature,
                    "measurement_bias_m": measurement_bias,
                    "innovation_m": innovation,
                    "accepted": accepted,
                    "bias_update_m": float(update[0]),
                    "speed_gain_update_m": float(update[1]),
                    "estimated_bias_m": float(state[0]),
                    "estimated_speed_gain_m": float(state[1]),
                }
            )
            last_event_time = float(sparse_time[sparse_index])
            sparse_index += 1

        velocity_index = int(
            np.searchsorted(vision_time, current_time, side="right") - 1
        )
        current_speed = 0.0 if velocity_index < 0 else float(vision_speed[velocity_index])
        speed_feature = (current_speed / selected.speed_scale_mps) ** 2
        correction_m = float(state[0] + state[1] * speed_feature)
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
    metadata = _result_metadata(
        estimator="robust_slow_bias_plus_quadratic_speed",
        config=selected,
        trace=trace,
    )
    metadata["speed_source"] = "causally smoothed optical-flow FRD horizontal speed"
    metadata["speed_model"] = "gain * (horizontal_speed_mps / speed_scale_mps)^2"
    return BarometerBiasResult(frame=result_frame, trace=trace, metadata=metadata)
