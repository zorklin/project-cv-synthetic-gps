"""Explain temporal-validation wins and regressions at sensor-input level."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


def _causal_reference(reference: pd.DataFrame, query_time: np.ndarray) -> np.ndarray:
    ref_time = reference["tr"].to_numpy(dtype=float)
    ref_height = reference["u"].to_numpy(dtype=float)
    indices = np.searchsorted(ref_time, query_time, side="right") - 1
    result = np.full(len(query_time), np.nan, dtype=float)
    valid = indices >= 0
    result[valid] = ref_height[indices[valid]]
    return result


def _error_stats(estimate: np.ndarray, reference: np.ndarray) -> dict[str, float | int]:
    valid = np.isfinite(estimate) & np.isfinite(reference)
    error = estimate[valid] - reference[valid]
    if not len(error):
        return {"n": 0, "mean_m": math.nan, "rmse_m": math.nan, "mae_m": math.nan}
    return {
        "n": int(len(error)),
        "mean_m": float(np.mean(error)),
        "rmse_m": float(np.sqrt(np.mean(error**2))),
        "mae_m": float(np.mean(np.abs(error))),
    }


def diagnose_temporal_sensors(
    work_dir: str | Path,
    validation_root: str | Path,
) -> list[dict[str, object]]:
    work = Path(work_dir).expanduser().resolve()
    root = Path(validation_root).expanduser().resolve()
    summary_path = root / "temporal_validation_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    gps = pd.read_csv(work / "gps_ref_enu.csv").sort_values("tr")
    starlink = pd.read_csv(work / "starlink_corrected_enu.csv").sort_values("tr_corr")
    rows: list[dict[str, object]] = []

    for validation_row in summary["rows"]:
        name = str(validation_row["window"])
        start = float(validation_row["actual_t_start"])
        end = float(validation_row["actual_t_end"])
        baseline = pd.read_csv(root / name / "raw_start_reset" / "gtsam" / "baro_graph_input.csv")
        candidate = pd.read_csv(root / name / "final_candidate" / "gtsam" / "baro_graph_input.csv")
        baseline = baseline.loc[(baseline["tr"] >= start) & (baseline["tr"] <= end)].copy()
        candidate = candidate.loc[(candidate["tr"] >= start) & (candidate["tr"] <= end)].copy()
        joined = baseline[["tr", "u_graph"]].merge(
            candidate[["tr", "u_graph"]],
            on="tr",
            how="inner",
            suffixes=("_baseline", "_candidate"),
            validate="one_to_one",
        )
        baro_reference = _causal_reference(gps, joined["tr"].to_numpy(dtype=float))
        base_stats = _error_stats(joined["u_graph_baseline"].to_numpy(dtype=float), baro_reference)
        candidate_stats = _error_stats(joined["u_graph_candidate"].to_numpy(dtype=float), baro_reference)
        sparse = starlink.loc[
            (starlink["tr_corr"] >= start) & (starlink["tr_corr"] <= end)
        ].copy()
        sparse_reference = _causal_reference(gps, sparse["tr_corr"].to_numpy(dtype=float))
        sparse_stats = _error_stats(sparse["u_corr"].to_numpy(dtype=float), sparse_reference)
        correction = (
            joined["u_graph_baseline"].to_numpy(dtype=float)
            - joined["u_graph_candidate"].to_numpy(dtype=float)
        )
        rows.append(
            {
                "window": name,
                "role": validation_row["role"],
                "graph_raw_improvement_percent": validation_row["raw_improvement_percent"],
                "baseline_baro_mean_error_m": base_stats["mean_m"],
                "baseline_baro_rmse_m": base_stats["rmse_m"],
                "candidate_baro_mean_error_m": candidate_stats["mean_m"],
                "candidate_baro_rmse_m": candidate_stats["rmse_m"],
                "starlink_height_mean_error_m": sparse_stats["mean_m"],
                "starlink_height_rmse_m": sparse_stats["rmse_m"],
                "starlink_height_samples": sparse_stats["n"],
                "mean_applied_correction_m": float(np.mean(correction)),
                "max_abs_applied_correction_m": float(np.max(np.abs(correction))),
            }
        )

    json_path = root / "temporal_sensor_diagnostics.json"
    csv_path = root / "temporal_sensor_diagnostics.csv"
    if json_path.exists() or csv_path.exists():
        raise FileExistsError("Temporal sensor diagnostics already exist")
    json_path.write_text(json.dumps({"schema_version": 1, "rows": rows}, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--validation-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    rows = diagnose_temporal_sensors(args.work_dir, args.validation_root)
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
