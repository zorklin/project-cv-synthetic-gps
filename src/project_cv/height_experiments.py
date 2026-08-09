"""Reusable, causal barometer-height experiments and diagnostics.

The functions in this module deliberately stop before GTSAM.  They construct
the vertical measurements that *would* be supplied to a fusion graph and
compare those measurements with dense GPS only as an offline diagnostic.
Dense GPS is never added to a graph or used inside the causal filters here.

Two alignment modes are provided:

``legacy_filtered_at_start``
    Reproduces the current notebook behaviour: filter the complete barometer
    history, compare the lagged filtered value at the active start with
    ``p0_u_m``, and add that difference to every filtered value.

``corrected_raw_at_start_filter_reset``
    Align the latest causal *raw* barometer sample to ``p0_u_m`` first, then
    restart the causal filter at that sample.  This avoids converting filter
    lag during take-off into a permanent altitude offset.

The offline linear-interpolation metric is explicitly non-causal.  The second
metric uses the latest output sample at or before each reference timestamp and
therefore represents causal zero-order-hold publication semantics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd


AlignmentMode = Literal[
    "legacy_filtered_at_start",
    "corrected_raw_at_start_filter_reset",
]
MatchingMode = Literal[
    "linear_interpolation_offline_noncausal",
    "previous_output_sample_causal_zoh",
]

LEGACY_ALIGNMENT_MODE: AlignmentMode = "legacy_filtered_at_start"
CORRECTED_ALIGNMENT_MODE: AlignmentMode = (
    "corrected_raw_at_start_filter_reset"
)

OFFLINE_INTERPOLATION_MODE: MatchingMode = (
    "linear_interpolation_offline_noncausal"
)
CAUSAL_PREVIOUS_MODE: MatchingMode = "previous_output_sample_causal_zoh"

DENSE_GPS_REFERENCE_ROLE = (
    "diagnostic_only_dense_gps_reference_not_used_by_graph_input"
)

FloatArray = npt.NDArray[np.float64]
NumericSeries = Sequence[float] | pd.Series | FloatArray


@dataclass(frozen=True, slots=True)
class CausalHeightFilterConfig:
    """Sample windows and time constant for the causal height filter."""

    median_window_samples: int = 11
    mean_window_samples: int = 11
    iir_tau_sec: float = 0.8

    def __post_init__(self) -> None:
        if self.median_window_samples < 1:
            raise ValueError("median_window_samples must be at least 1")
        if self.mean_window_samples < 1:
            raise ValueError("mean_window_samples must be at least 1")
        if not math.isfinite(self.iir_tau_sec) or self.iir_tau_sec <= 0.0:
            raise ValueError("iir_tau_sec must be finite and greater than zero")


@dataclass(frozen=True, slots=True)
class EvaluationWindow:
    """Inclusive diagnostic interval in the shared relative timebase."""

    name: str
    start_sec: float | None = None
    end_sec: float | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("EvaluationWindow.name must not be empty")
        if self.start_sec is not None and not math.isfinite(self.start_sec):
            raise ValueError("EvaluationWindow.start_sec must be finite")
        if self.end_sec is not None and not math.isfinite(self.end_sec):
            raise ValueError("EvaluationWindow.end_sec must be finite")
        if (
            self.start_sec is not None
            and self.end_sec is not None
            and self.end_sec < self.start_sec
        ):
            raise ValueError("EvaluationWindow.end_sec precedes start_sec")


@dataclass(slots=True)
class HeightGraphInput:
    """A causal barometer-derived graph-input series and its provenance."""

    mode: AlignmentMode
    frame: pd.DataFrame
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        required = {"tr", "u_raw", "u_filter_base", "u_smooth", "u_graph"}
        missing = sorted(required - set(self.frame.columns))
        if missing:
            raise ValueError(f"HeightGraphInput.frame is missing columns: {missing}")
        if not len(self.frame):
            raise ValueError("HeightGraphInput.frame must not be empty")


@dataclass(slots=True)
class HeightComparison:
    """Both graph-input variants and dense-GPS diagnostic results."""

    graph_inputs: dict[str, HeightGraphInput]
    alignments: pd.DataFrame
    metrics: pd.DataFrame
    summary: dict[str, Any]


def _as_float_array(values: NumericSeries, *, name: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return array


def _filter_config(
    config: CausalHeightFilterConfig | None,
) -> CausalHeightFilterConfig:
    return config if config is not None else CausalHeightFilterConfig()


def _require_columns(
    frame: pd.DataFrame,
    columns: set[str],
    *,
    label: str,
) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def _prepare_series(
    frame: pd.DataFrame,
    *,
    time_column: str,
    value_column: str,
    label: str,
) -> pd.DataFrame:
    """Return a finite, stable-sorted, unique two-column numeric series."""

    _require_columns(frame, {time_column, value_column}, label=label)
    prepared = frame.loc[:, [time_column, value_column]].copy()
    prepared[time_column] = pd.to_numeric(prepared[time_column], errors="coerce")
    prepared[value_column] = pd.to_numeric(prepared[value_column], errors="coerce")

    finite = np.isfinite(prepared[time_column].to_numpy(dtype=float))
    finite &= np.isfinite(prepared[value_column].to_numpy(dtype=float))
    prepared = prepared.loc[finite]
    prepared = prepared.sort_values(time_column, kind="mergesort")
    # The fusion notebook keeps the first message when timestamps collide.
    prepared = prepared.drop_duplicates(subset=[time_column], keep="first")
    prepared = prepared.reset_index(drop=True)

    if not len(prepared):
        raise ValueError(f"{label} has no finite samples")

    time_sec = prepared[time_column].to_numpy(dtype=float)
    if len(time_sec) > 1 and not np.all(np.diff(time_sec) > 0.0):
        raise ValueError(f"{label} timestamps are not strictly increasing")
    return prepared


def causal_rolling_median_mean(
    values: NumericSeries,
    *,
    median_window_samples: int = 11,
    mean_window_samples: int = 11,
) -> FloatArray:
    """Apply trailing rolling median followed by trailing rolling mean.

    Both windows use ``center=False`` and ``min_periods=1``.  Every output
    therefore depends only on the current and earlier values.
    """

    if median_window_samples < 1 or mean_window_samples < 1:
        raise ValueError("rolling windows must be at least 1 sample")
    value_array = _as_float_array(values, name="values")
    if not len(value_array):
        return np.empty(0, dtype=np.float64)

    return (
        pd.Series(value_array, dtype="float64")
        .rolling(
            window=int(median_window_samples),
            center=False,
            min_periods=1,
        )
        .median()
        .rolling(
            window=int(mean_window_samples),
            center=False,
            min_periods=1,
        )
        .mean()
        .to_numpy(dtype=np.float64)
    )


def causal_iir_by_time(
    time_sec: NumericSeries,
    values: NumericSeries,
    *,
    tau_sec: float,
    initial_value: float | None = None,
) -> FloatArray:
    """Apply a causal first-order IIR low-pass using actual sample times.

    The update is ``y += (1-exp(-dt/tau)) * (x-y)``.  A non-finite input is
    handled by holding the latest finite output; no future sample is used.
    """

    time_array = _as_float_array(time_sec, name="time_sec")
    value_array = _as_float_array(values, name="values")
    if len(time_array) != len(value_array):
        raise ValueError("time_sec and values must have the same length")
    if not math.isfinite(tau_sec) or tau_sec <= 0.0:
        raise ValueError("tau_sec must be finite and greater than zero")
    if initial_value is not None and not math.isfinite(initial_value):
        raise ValueError("initial_value must be finite when supplied")
    if not len(value_array):
        return np.empty(0, dtype=np.float64)
    if not np.all(np.isfinite(time_array)):
        raise ValueError("time_sec contains non-finite values")
    if len(time_array) > 1 and not np.all(np.diff(time_array) > 0.0):
        raise ValueError("time_sec must be strictly increasing")

    output = np.full_like(value_array, np.nan, dtype=np.float64)
    finite_indices = np.flatnonzero(np.isfinite(value_array))
    if not len(finite_indices):
        return output

    first = int(finite_indices[0])
    first_output = (
        float(value_array[first])
        if initial_value is None
        else float(initial_value)
    )
    # Before the first finite observation there is no causal state to hold.
    # Keep that prefix as NaN instead of back-filling from a future sample.
    output[first] = first_output
    last_value = first_output
    last_time = float(time_array[first])

    for index in range(first + 1, len(value_array)):
        current_time = float(time_array[index])
        delta_time = max(0.0, current_time - last_time)
        alpha = 1.0 - math.exp(-delta_time / float(tau_sec))
        current_value = float(value_array[index])
        if math.isfinite(current_value):
            last_value += alpha * (current_value - last_value)
        output[index] = last_value
        last_time = current_time
    return output


def causal_height_filter(
    time_sec: NumericSeries,
    raw_height_m: NumericSeries,
    *,
    config: CausalHeightFilterConfig | None = None,
    initial_value: float | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Return ``(rolling_base, iir_output)`` for the configured causal filter."""

    selected = _filter_config(config)
    time_array = _as_float_array(time_sec, name="time_sec")
    raw_array = _as_float_array(raw_height_m, name="raw_height_m")
    if len(time_array) != len(raw_array):
        raise ValueError("time_sec and raw_height_m must have the same length")

    rolling_base = causal_rolling_median_mean(
        raw_array,
        median_window_samples=selected.median_window_samples,
        mean_window_samples=selected.mean_window_samples,
    )
    filtered = causal_iir_by_time(
        time_array,
        rolling_base,
        tau_sec=selected.iir_tau_sec,
        initial_value=initial_value,
    )
    return rolling_base, filtered


def causal_index_at_or_before(time_sec: NumericSeries, query_time_sec: float) -> int:
    """Index of the latest sample at or before ``query_time_sec``."""

    time_array = _as_float_array(time_sec, name="time_sec")
    if not len(time_array):
        raise ValueError("time_sec must not be empty")
    if not math.isfinite(query_time_sec):
        raise ValueError("query_time_sec must be finite")
    if not np.all(np.isfinite(time_array)):
        raise ValueError("time_sec contains non-finite values")
    if len(time_array) > 1 and not np.all(np.diff(time_array) > 0.0):
        raise ValueError("time_sec must be strictly increasing")

    index = int(np.searchsorted(time_array, float(query_time_sec), side="right") - 1)
    if index < 0:
        raise ValueError("no causal sample exists at or before query_time_sec")
    return index


def build_legacy_graph_input(
    barometer: pd.DataFrame,
    *,
    active_start_sec: float,
    p0_u_m: float,
    time_column: str = "tr",
    raw_height_column: str = "u_corr",
    p0_source: str = "unspecified",
    config: CausalHeightFilterConfig | None = None,
) -> HeightGraphInput:
    """Reproduce filtered-at-start alignment from the baseline notebook."""

    if not math.isfinite(active_start_sec) or not math.isfinite(p0_u_m):
        raise ValueError("active_start_sec and p0_u_m must be finite")
    selected = _filter_config(config)
    prepared = _prepare_series(
        barometer,
        time_column=time_column,
        value_column=raw_height_column,
        label="barometer",
    )
    time_sec = prepared[time_column].to_numpy(dtype=np.float64)
    raw_height = prepared[raw_height_column].to_numpy(dtype=np.float64)
    rolling_base, filtered = causal_height_filter(
        time_sec,
        raw_height,
        config=selected,
    )
    anchor_index = causal_index_at_or_before(time_sec, active_start_sec)
    filtered_at_start = float(filtered[anchor_index])
    alignment_offset = float(p0_u_m) - filtered_at_start

    output = pd.DataFrame(
        {
            "tr": time_sec,
            "u_raw": raw_height,
            "u_aligned_raw": raw_height + alignment_offset,
            "u_filter_base": rolling_base,
            "u_smooth": filtered,
            "u_graph": filtered + alignment_offset,
            "active": time_sec >= float(active_start_sec),
            "alignment_mode": LEGACY_ALIGNMENT_MODE,
        }
    )
    metadata: dict[str, Any] = {
        "alignment_mode": LEGACY_ALIGNMENT_MODE,
        "causal": True,
        "filter_history": "complete_pre_start_history",
        "alignment_source": "filtered_value_at_or_before_active_start",
        "active_start_sec": float(active_start_sec),
        "p0_u_m": float(p0_u_m),
        "p0_source": str(p0_source),
        "anchor_index": int(anchor_index),
        "anchor_time_sec": float(time_sec[anchor_index]),
        "anchor_raw_u_m": float(raw_height[anchor_index]),
        "anchor_filtered_u_m": filtered_at_start,
        "alignment_offset_m": alignment_offset,
        "input_rows": int(len(output)),
        "filter": asdict(selected),
        "dense_gps_dataframe_read_by_builder": False,
        "p0_source_is_external_to_module": True,
    }
    return HeightGraphInput(
        mode=LEGACY_ALIGNMENT_MODE,
        frame=output,
        metadata=metadata,
    )


def build_corrected_graph_input(
    barometer: pd.DataFrame,
    *,
    active_start_sec: float,
    p0_u_m: float,
    time_column: str = "tr",
    raw_height_column: str = "u_corr",
    p0_source: str = "unspecified",
    config: CausalHeightFilterConfig | None = None,
) -> HeightGraphInput:
    """Align raw height first and restart the causal filter at active start.

    The latest raw sample at or before ``active_start_sec`` is the anchor.
    Earlier samples are intentionally excluded from this variant, so filter
    state accumulated during the pre-flight/take-off transition cannot create
    a permanent alignment offset.
    """

    if not math.isfinite(active_start_sec) or not math.isfinite(p0_u_m):
        raise ValueError("active_start_sec and p0_u_m must be finite")
    selected = _filter_config(config)
    prepared = _prepare_series(
        barometer,
        time_column=time_column,
        value_column=raw_height_column,
        label="barometer",
    )
    all_time = prepared[time_column].to_numpy(dtype=np.float64)
    all_raw = prepared[raw_height_column].to_numpy(dtype=np.float64)
    anchor_index = causal_index_at_or_before(all_time, active_start_sec)
    anchor_raw = float(all_raw[anchor_index])
    alignment_offset = float(p0_u_m) - anchor_raw

    # Include the causal anchor itself; it may be slightly before the first
    # graph-node time and is therefore available to that node.
    time_sec = all_time[anchor_index:].copy()
    raw_height = all_raw[anchor_index:].copy()
    aligned_raw = raw_height + alignment_offset
    rolling_base, filtered = causal_height_filter(
        time_sec,
        aligned_raw,
        config=selected,
        initial_value=float(p0_u_m),
    )

    output = pd.DataFrame(
        {
            "tr": time_sec,
            "u_raw": raw_height,
            "u_aligned_raw": aligned_raw,
            "u_filter_base": rolling_base,
            "u_smooth": filtered,
            "u_graph": filtered,
            "active": time_sec >= float(active_start_sec),
            "alignment_mode": CORRECTED_ALIGNMENT_MODE,
        }
    )
    metadata: dict[str, Any] = {
        "alignment_mode": CORRECTED_ALIGNMENT_MODE,
        "causal": True,
        "filter_history": "reset_at_latest_raw_sample_before_active_start",
        "alignment_source": "raw_value_at_or_before_active_start",
        "active_start_sec": float(active_start_sec),
        "p0_u_m": float(p0_u_m),
        "p0_source": str(p0_source),
        "anchor_index_in_full_input": int(anchor_index),
        "anchor_time_sec": float(time_sec[0]),
        "anchor_raw_u_m": anchor_raw,
        "anchor_filtered_u_m": float(filtered[0]),
        "alignment_offset_m": alignment_offset,
        "discarded_pre_anchor_rows": int(anchor_index),
        "input_rows": int(len(output)),
        "filter": asdict(selected),
        "dense_gps_dataframe_read_by_builder": False,
        "p0_source_is_external_to_module": True,
    }
    return HeightGraphInput(
        mode=CORRECTED_ALIGNMENT_MODE,
        frame=output,
        metadata=metadata,
    )


def build_alignment_variants(
    barometer: pd.DataFrame,
    *,
    active_start_sec: float,
    p0_u_m: float,
    time_column: str = "tr",
    raw_height_column: str = "u_corr",
    p0_source: str = "unspecified",
    config: CausalHeightFilterConfig | None = None,
) -> dict[str, HeightGraphInput]:
    """Build legacy and corrected graph inputs from the same raw samples."""

    kwargs: dict[str, Any] = {
        "active_start_sec": active_start_sec,
        "p0_u_m": p0_u_m,
        "time_column": time_column,
        "raw_height_column": raw_height_column,
        "p0_source": p0_source,
        "config": config,
    }
    legacy = build_legacy_graph_input(barometer, **kwargs)
    corrected = build_corrected_graph_input(barometer, **kwargs)
    return {
        legacy.mode: legacy,
        corrected.mode: corrected,
    }


def align_height_to_dense_gps(
    samples: pd.DataFrame,
    dense_gps: pd.DataFrame,
    *,
    series_name: str,
    matching_mode: MatchingMode,
    sample_time_column: str = "tr",
    sample_height_column: str = "u_graph",
    reference_time_column: str = "tr",
    reference_height_column: str = "u",
    evaluation_start_sec: float | None = None,
    evaluation_end_sec: float | None = None,
) -> pd.DataFrame:
    """Align one height series to dense GPS for diagnostics only."""

    source = _prepare_series(
        samples,
        time_column=sample_time_column,
        value_column=sample_height_column,
        label=f"height series {series_name}",
    )
    reference = _prepare_series(
        dense_gps,
        time_column=reference_time_column,
        value_column=reference_height_column,
        label="dense GPS diagnostic reference",
    )
    if len(source) < 2 or len(reference) < 2:
        raise ValueError("height series and dense GPS need at least two samples")

    source_time = source[sample_time_column].to_numpy(dtype=np.float64)
    source_height = source[sample_height_column].to_numpy(dtype=np.float64)
    reference_time = reference[reference_time_column].to_numpy(dtype=np.float64)
    reference_height = reference[reference_height_column].to_numpy(dtype=np.float64)

    overlap_start = max(float(source_time[0]), float(reference_time[0]))
    overlap_end = min(float(source_time[-1]), float(reference_time[-1]))
    if evaluation_start_sec is not None:
        overlap_start = max(overlap_start, float(evaluation_start_sec))
    if evaluation_end_sec is not None:
        overlap_end = min(overlap_end, float(evaluation_end_sec))
    if overlap_end < overlap_start:
        raise ValueError(f"{series_name} and dense GPS have no requested overlap")

    use_reference = (
        (reference_time >= overlap_start) & (reference_time <= overlap_end)
    )
    query_time = reference_time[use_reference]
    query_reference = reference_height[use_reference]
    if len(query_time) < 2:
        raise ValueError(f"{series_name} has fewer than two reference points")

    previous_index = np.searchsorted(source_time, query_time, side="right") - 1
    valid_previous = previous_index >= 0
    query_time = query_time[valid_previous]
    query_reference = query_reference[valid_previous]
    previous_index = previous_index[valid_previous]

    if matching_mode == OFFLINE_INTERPOLATION_MODE:
        estimate = np.interp(query_time, source_time, source_height)
        sample_age_sec = np.full(len(query_time), np.nan, dtype=np.float64)
    elif matching_mode == CAUSAL_PREVIOUS_MODE:
        estimate = source_height[previous_index]
        sample_age_sec = query_time - source_time[previous_index]
    else:  # Defensive runtime check for callers without static typing.
        raise ValueError(f"Unknown matching_mode: {matching_mode}")

    aligned = pd.DataFrame(
        {
            "tr": query_time,
            "ref_u": query_reference,
            "est_u": estimate,
            "sample_age_sec": sample_age_sec,
            "series": str(series_name),
            "matching_mode": matching_mode,
            "reference_role": DENSE_GPS_REFERENCE_ROLE,
        }
    )
    aligned["err_u"] = aligned["est_u"] - aligned["ref_u"]
    aligned["abs_err_u"] = aligned["err_u"].abs()
    return aligned


def summarize_height_alignment(
    aligned: pd.DataFrame,
    *,
    window: EvaluationWindow | None = None,
) -> dict[str, Any]:
    """Summarize one already-aligned diagnostic table."""

    _require_columns(
        aligned,
        {
            "tr",
            "err_u",
            "abs_err_u",
            "sample_age_sec",
            "series",
            "matching_mode",
            "reference_role",
        },
        label="aligned height diagnostics",
    )
    selected_window = window or EvaluationWindow(name="all_overlap")
    use = np.ones(len(aligned), dtype=bool)
    time_sec = aligned["tr"].to_numpy(dtype=float)
    if selected_window.start_sec is not None:
        use &= time_sec >= float(selected_window.start_sec)
    if selected_window.end_sec is not None:
        use &= time_sec <= float(selected_window.end_sec)
    selected = aligned.loc[use]
    if not len(selected):
        raise ValueError(f"evaluation window {selected_window.name!r} is empty")

    error = selected["err_u"].to_numpy(dtype=float)
    ages = selected["sample_age_sec"].to_numpy(dtype=float)
    ages = ages[np.isfinite(ages)]
    return {
        "scope": selected_window.name,
        "window_start_sec": selected_window.start_sec,
        "window_end_sec": selected_window.end_sec,
        "series": str(selected["series"].iloc[0]),
        "matching_mode": str(selected["matching_mode"].iloc[0]),
        "reference_role": str(selected["reference_role"].iloc[0]),
        "n": int(len(selected)),
        "bias_u_m": float(np.mean(error)),
        "rmse_u_m": float(np.sqrt(np.mean(error**2))),
        "mae_u_m": float(np.mean(np.abs(error))),
        "p95_abs_u_m": float(np.percentile(np.abs(error), 95)),
        "max_abs_u_m": float(np.max(np.abs(error))),
        "p95_sample_age_sec": (
            float(np.percentile(ages, 95)) if len(ages) else None
        ),
        "max_sample_age_sec": float(np.max(ages)) if len(ages) else None,
    }


def compare_alignment_variants(
    graph_inputs: Mapping[str, HeightGraphInput],
    dense_gps: pd.DataFrame,
    *,
    windows: Sequence[EvaluationWindow] | None = None,
    reference_time_column: str = "tr",
    reference_height_column: str = "u",
) -> HeightComparison:
    """Evaluate all graph inputs on identical dense-GPS diagnostic support."""

    if not graph_inputs:
        raise ValueError("graph_inputs must not be empty")
    selected_windows = tuple(windows or (EvaluationWindow("all_overlap"),))
    if not selected_windows:
        raise ValueError("windows must not be empty")

    # Force common support so variants cannot obtain incomparable metrics by
    # silently using different numbers of dense-GPS samples.
    common_start = -math.inf
    common_end = math.inf
    for name, graph_input in graph_inputs.items():
        prepared = _prepare_series(
            graph_input.frame,
            time_column="tr",
            value_column="u_graph",
            label=f"graph input {name}",
        )
        common_start = max(common_start, float(prepared["tr"].iloc[0]))
        common_end = min(common_end, float(prepared["tr"].iloc[-1]))
    if common_end < common_start:
        raise ValueError("graph inputs have no common time support")

    alignments: list[pd.DataFrame] = []
    metrics: list[dict[str, Any]] = []
    for name, graph_input in graph_inputs.items():
        for matching_mode in (
            OFFLINE_INTERPOLATION_MODE,
            CAUSAL_PREVIOUS_MODE,
        ):
            aligned = align_height_to_dense_gps(
                graph_input.frame,
                dense_gps,
                series_name=str(name),
                matching_mode=matching_mode,
                reference_time_column=reference_time_column,
                reference_height_column=reference_height_column,
                evaluation_start_sec=common_start,
                evaluation_end_sec=common_end,
            )
            alignments.append(aligned)
            for window in selected_windows:
                metrics.append(summarize_height_alignment(aligned, window=window))

    alignment_table = pd.concat(alignments, ignore_index=True)
    metrics_table = pd.DataFrame(metrics)
    graph_input_metadata = {
        name: graph_input.metadata for name, graph_input in graph_inputs.items()
    }
    summary: dict[str, Any] = {
        "schema_version": 1,
        "contract": {
            "gtsam_rerun": False,
            "graph_algorithm_modified": False,
            "graph_inputs_are_causal": True,
            "dense_gps_role": DENSE_GPS_REFERENCE_ROLE,
            "dense_gps_fused": False,
            "dense_gps_used_only_after_graph_inputs_were_constructed": True,
            "offline_interpolation_is_noncausal_diagnostic": True,
        },
        "common_support_start_sec": float(common_start),
        "common_support_end_sec": float(common_end),
        "evaluation_windows": [asdict(item) for item in selected_windows],
        "graph_inputs": graph_input_metadata,
    }
    return HeightComparison(
        graph_inputs=dict(graph_inputs),
        alignments=alignment_table,
        metrics=metrics_table,
        summary=summary,
    )


def build_and_compare_alignment_variants(
    barometer: pd.DataFrame,
    dense_gps: pd.DataFrame,
    *,
    active_start_sec: float,
    p0_u_m: float,
    active_end_sec: float | None = None,
    time_column: str = "tr",
    raw_height_column: str = "u_corr",
    reference_time_column: str = "tr",
    reference_height_column: str = "u",
    p0_source: str = "unspecified",
    config: CausalHeightFilterConfig | None = None,
    windows: Sequence[EvaluationWindow] | None = None,
) -> HeightComparison:
    """Convenience entry point for the focused legacy/corrected comparison."""

    variants = build_alignment_variants(
        barometer,
        active_start_sec=active_start_sec,
        p0_u_m=p0_u_m,
        time_column=time_column,
        raw_height_column=raw_height_column,
        p0_source=p0_source,
        config=config,
    )
    selected_windows = windows
    if selected_windows is None:
        selected_windows = (
            EvaluationWindow(
                "active_full",
                start_sec=float(active_start_sec),
                end_sec=(
                    None if active_end_sec is None else float(active_end_sec)
                ),
            ),
        )
    return compare_alignment_variants(
        variants,
        dense_gps,
        windows=selected_windows,
        reference_time_column=reference_time_column,
        reference_height_column=reference_height_column,
    )


def _records_for_json(frame: pd.DataFrame) -> list[dict[str, Any]]:
    # pandas converts NaN to JSON null, unlike json.dumps(..., allow_nan=False).
    return json.loads(frame.to_json(orient="records"))


def _validate_file_prefix(prefix: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", prefix):
        raise ValueError(
            "prefix must contain only letters, digits, dot, underscore or dash"
        )
    if ".." in prefix:
        raise ValueError("prefix must not contain '..'")
    return prefix


def save_height_comparison(
    comparison: HeightComparison,
    output_dir: str | Path,
    *,
    prefix: str = "height_alignment",
) -> dict[str, Path]:
    """Save graph inputs, aligned samples, metrics and one JSON manifest."""

    selected_prefix = _validate_file_prefix(prefix)
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if not destination.is_dir():
        raise NotADirectoryError(destination)

    saved: dict[str, Path] = {}
    for name, graph_input in comparison.graph_inputs.items():
        safe_name = _validate_file_prefix(str(name))
        path = destination / f"{selected_prefix}_{safe_name}_graph_input.csv"
        graph_input.frame.to_csv(path, index=False)
        saved[f"graph_input_{name}"] = path

    alignments_path = destination / f"{selected_prefix}_aligned_samples.csv"
    metrics_path = destination / f"{selected_prefix}_metrics.csv"
    summary_path = destination / f"{selected_prefix}_summary.json"
    comparison.alignments.to_csv(alignments_path, index=False)
    comparison.metrics.to_csv(metrics_path, index=False)
    saved["aligned_samples"] = alignments_path
    saved["metrics"] = metrics_path

    payload = dict(comparison.summary)
    payload["metrics"] = _records_for_json(comparison.metrics)
    payload["generated_files"] = {
        key: path.name for key, path in {**saved, "summary": summary_path}.items()
    }
    summary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    saved["summary"] = summary_path
    return saved


__all__ = [
    "CAUSAL_PREVIOUS_MODE",
    "CORRECTED_ALIGNMENT_MODE",
    "DENSE_GPS_REFERENCE_ROLE",
    "LEGACY_ALIGNMENT_MODE",
    "OFFLINE_INTERPOLATION_MODE",
    "AlignmentMode",
    "CausalHeightFilterConfig",
    "EvaluationWindow",
    "HeightComparison",
    "HeightGraphInput",
    "MatchingMode",
    "align_height_to_dense_gps",
    "build_alignment_variants",
    "build_and_compare_alignment_variants",
    "build_corrected_graph_input",
    "build_legacy_graph_input",
    "causal_height_filter",
    "causal_iir_by_time",
    "causal_index_at_or_before",
    "causal_rolling_median_mean",
    "compare_alignment_variants",
    "save_height_comparison",
    "summarize_height_alignment",
]
