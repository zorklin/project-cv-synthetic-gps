"""Reproducible static-pressure versus horizontal-speed diagnostics.

Dense GPS is used only as an offline reference for explaining barometer error;
none of the outputs from this module are fused by the navigation graph.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from project_cv.experiments.height.height_experiments import (
    CAUSAL_PREVIOUS_MODE,
    align_height_to_dense_gps,
    summarize_height_alignment,
)


def extract_static_pressure(
    bag_directory: str | Path,
    *,
    t0_epoch_sec: float,
    topic: str = "/mavros/imu/static_pressure",
) -> pd.DataFrame:
    """Extract FluidPressure messages using MCAP bag time."""

    from rosbags.highlevel import AnyReader

    bag_path = Path(bag_directory).expanduser().resolve()
    if not bag_path.is_dir():
        raise FileNotFoundError(f"ROS bag directory does not exist: {bag_path}")
    rows: list[dict[str, float]] = []
    with AnyReader([bag_path]) as reader:
        connections = [item for item in reader.connections if item.topic == topic]
        if not connections:
            raise RuntimeError(f"Topic {topic!r} is absent from {bag_path}")
        for connection, timestamp_ns, rawdata in reader.messages(connections=connections):
            message = reader.deserialize(rawdata, connection.msgtype)
            rows.append(
                {
                    "bag_t": float(timestamp_ns) / 1e9,
                    "tr": float(timestamp_ns) / 1e9 - float(t0_epoch_sec),
                    "pressure_pa": float(message.fluid_pressure),
                    "variance": float(message.variance),
                }
            )
    result = pd.DataFrame(rows)
    if not len(result):
        raise RuntimeError(f"Topic {topic!r} contains no messages")
    return result.sort_values("tr", kind="mergesort").reset_index(drop=True)


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    finite = np.isfinite(left) & np.isfinite(right)
    if int(finite.sum()) < 3:
        return math.nan
    return float(np.corrcoef(left[finite], right[finite])[0, 1])


def _gps_horizontal_speed(dense_gps: pd.DataFrame) -> np.ndarray:
    time_sec = dense_gps["tr"].to_numpy(dtype=float)
    east = (
        dense_gps["e"].rolling(11, center=True, min_periods=1).median().to_numpy(float)
    )
    north = (
        dense_gps["n"].rolling(11, center=True, min_periods=1).median().to_numpy(float)
    )
    return np.hypot(np.gradient(east, time_sec), np.gradient(north, time_sec))


def analyze_static_pressure_speed(
    static_pressure: pd.DataFrame,
    barometer_graph_input: pd.DataFrame,
    dense_gps: pd.DataFrame,
    vision_velocity: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Quantify the pressure/height residual relationship with horizontal speed."""

    pressure = static_pressure.sort_values("tr", kind="mergesort").drop_duplicates(
        "tr", keep="first"
    )
    barometer = barometer_graph_input.sort_values(
        "tr", kind="mergesort"
    ).drop_duplicates("tr", keep="first")
    gps = dense_gps.sort_values("tr", kind="mergesort").drop_duplicates(
        "tr", keep="first"
    )
    vision = vision_velocity.sort_values(
        "tr_corr", kind="mergesort"
    ).drop_duplicates("tr_corr", keep="first")
    for label, frame, columns in (
        ("static pressure", pressure, {"tr", "pressure_pa"}),
        ("barometer graph input", barometer, {"tr", "u_graph"}),
        ("dense GPS", gps, {"tr", "e", "n", "u"}),
        ("vision velocity", vision, {"tr_corr", "vx", "vy"}),
    ):
        missing = sorted(columns - set(frame.columns))
        if missing:
            raise ValueError(f"{label} is missing columns: {missing}")

    aligned = align_height_to_dense_gps(
        barometer,
        gps,
        series_name="raw_start_reset_barometer",
        matching_mode=CAUSAL_PREVIOUS_MODE,
        sample_height_column="u_graph",
    )
    gps_speed = _gps_horizontal_speed(gps)
    aligned["gps_horizontal_speed_mps"] = np.interp(
        aligned["tr"], gps["tr"], gps_speed
    )
    vision_speed = np.hypot(
        vision["vx"].to_numpy(dtype=float),
        vision["vy"].to_numpy(dtype=float),
    )
    aligned["vision_horizontal_speed_mps"] = np.interp(
        aligned["tr"],
        vision["tr_corr"],
        vision_speed,
        left=np.nan,
        right=np.nan,
    )

    overlap_start = max(float(pressure["tr"].min()), float(gps["tr"].min()))
    overlap_end = min(float(pressure["tr"].max()), float(gps["tr"].max()))
    pressure_overlap = pressure.loc[
        (pressure["tr"] >= overlap_start) & (pressure["tr"] <= overlap_end)
    ].copy()
    pressure_time = pressure_overlap["tr"].to_numpy(dtype=float)
    pressure_pa = pressure_overlap["pressure_pa"].to_numpy(dtype=float)
    gps_u_at_pressure = np.interp(pressure_time, gps["tr"], gps["u"])
    gps_speed_at_pressure = np.interp(pressure_time, gps["tr"], gps_speed)
    pressure_fit = np.polyfit(gps_u_at_pressure, pressure_pa, 1)
    pressure_residual = pressure_pa - np.polyval(pressure_fit, gps_u_at_pressure)
    pressure_overlap["gps_u_m"] = gps_u_at_pressure
    pressure_overlap["gps_horizontal_speed_mps"] = gps_speed_at_pressure
    pressure_overlap["pressure_residual_pa"] = pressure_residual

    height_error = aligned["err_u"].to_numpy(dtype=float)
    gps_speed_for_error = aligned["gps_horizontal_speed_mps"].to_numpy(dtype=float)
    vision_speed_for_error = aligned["vision_horizontal_speed_mps"].to_numpy(dtype=float)
    speed_cut = float(np.nanmedian(gps_speed_for_error))
    slow = gps_speed_for_error <= speed_cut
    fast = gps_speed_for_error > speed_cut
    height_metrics = summarize_height_alignment(aligned)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "contract": {
            "diagnostic_only": True,
            "dense_gps_fused": False,
            "dense_gps_role": "offline_reference_only",
            "static_pressure_timebase": "MCAP bag timestamp minus calibration t0",
        },
        "sample_counts": {
            "static_pressure": int(len(pressure)),
            "pressure_gps_overlap": int(len(pressure_overlap)),
            "height_error_alignment": int(len(aligned)),
        },
        "pressure_range_pa": [
            float(pressure["pressure_pa"].min()),
            float(pressure["pressure_pa"].max()),
        ],
        "pressure_altitude_linear_fit": {
            "slope_pa_per_m": float(pressure_fit[0]),
            "intercept_pa": float(pressure_fit[1]),
        },
        "correlations": {
            "pressure_residual_vs_gps_horizontal_speed": _correlation(
                pressure_residual, gps_speed_at_pressure
            ),
            "barometer_height_error_vs_gps_horizontal_speed": _correlation(
                height_error, gps_speed_for_error
            ),
            "barometer_height_error_vs_vision_horizontal_speed": _correlation(
                height_error, vision_speed_for_error
            ),
        },
        "barometer_height_error": height_metrics,
        "speed_split": {
            "median_gps_horizontal_speed_mps": speed_cut,
            "slow_bias_m": float(np.mean(height_error[slow])),
            "slow_rmse_m": float(np.sqrt(np.mean(height_error[slow] ** 2))),
            "fast_bias_m": float(np.mean(height_error[fast])),
            "fast_rmse_m": float(np.sqrt(np.mean(height_error[fast] ** 2))),
        },
    }
    return summary, pressure_overlap


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag-dir", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--barometer-graph-input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    artifacts = os.environ.get("PROJECT_CV_ARTIFACTS")
    derived_bag = os.environ.get("PROJECT_CV_DERIVED_BAG")
    work = args.work_dir or (Path(artifacts) / "fusion_v1/work" if artifacts else None)
    bag = args.bag_dir or (Path(derived_bag) if derived_bag else None)
    if work is None or bag is None:
        raise RuntimeError("Provide paths or source PROJECT_CV_ARTIFACTS/PROJECT_CV_DERIVED_BAG")
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output}")
    calibration = json.loads((work / "calibration.json").read_text(encoding="utf-8"))
    pressure = extract_static_pressure(bag, t0_epoch_sec=float(calibration["t0_epoch_sec"]))
    summary, aligned_pressure = analyze_static_pressure_speed(
        pressure,
        pd.read_csv(args.barometer_graph_input),
        pd.read_csv(work / "gps_ref_enu.csv"),
        pd.read_csv(work / "vision_velocity_raw_frd.csv"),
    )
    output.mkdir(parents=True, exist_ok=True)
    pressure.to_csv(output / "static_pressure.csv", index=False)
    aligned_pressure.to_csv(output / "static_pressure_gps_diagnostic.csv", index=False)
    (output / "static_pressure_speed_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
