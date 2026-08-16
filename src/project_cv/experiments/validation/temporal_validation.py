"""Blocked temporal validation for the frozen baseline and final candidate.

This is deliberately not a random train/test split.  Each window is replayed
chronologically with all causal state reset and with the parameters kept fixed.
Dense GPS is used by the frozen notebook for start initialization and scoring,
but is never inserted into the fusion graph as a measurement factor.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Final, Sequence

from project_cv.experiments.runners.windowed_fusion_runner import WINDOW_MODELS, run_window_experiment


@dataclass(frozen=True)
class WindowSpec:
    name: str
    offset_sec: float
    duration_sec: float
    role: str


LONG_WINDOWS: Final = tuple(
    WindowSpec(f"long_o{offset:03d}_d180", float(offset), 180.0, "primary")
    for offset in (0, 180, 360, 540, 720, 900)
)
SHORT_WINDOWS: Final = (
    WindowSpec("short_o030_d040", 30.0, 40.0, "startup_stress"),
    WindowSpec("short_o300_d040", 300.0, 40.0, "startup_stress"),
    WindowSpec("short_o600_d040", 600.0, 40.0, "startup_stress"),
    WindowSpec("short_o900_d040", 900.0, 40.0, "startup_stress"),
)

ACCEPTANCE: Final = {
    "primary_windows": len(LONG_WINDOWS),
    "minimum_raw_rmse_wins": 5,
    "minimum_median_raw_improvement_percent": 5.0,
    "maximum_worst_raw_regression_percent": 10.0,
    "require_zero_failed_updates": True,
    "require_equal_measurement_counts": True,
}


def _load_completed_manifest(output_dir: Path, spec: WindowSpec, model: str) -> dict[str, Any] | None:
    manifest_path = output_dir / "experiment_manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    contract = manifest.get("window_contract", {})
    if manifest.get("status") != "completed":
        raise RuntimeError(f"Existing run is not completed: {manifest_path}")
    if manifest.get("model") != model:
        raise RuntimeError(f"Existing run has a different model: {manifest_path}")
    if not math.isclose(float(contract.get("requested_offset_sec", -1)), spec.offset_sec):
        raise RuntimeError(f"Existing run has a different offset: {manifest_path}")
    if not math.isclose(float(contract.get("requested_duration_sec", -1)), spec.duration_sec):
        raise RuntimeError(f"Existing run has a different duration: {manifest_path}")
    return manifest


def _run_or_reuse(
    root: Path,
    spec: WindowSpec,
    model: str,
    *,
    source_root: Path | None,
    artifacts_root: Path | None,
    reuse_existing: bool,
) -> dict[str, Any]:
    output_dir = root / spec.name / model / "gtsam"
    if reuse_existing:
        existing = _load_completed_manifest(output_dir, spec, model)
        if existing is not None:
            return existing
    return run_window_experiment(
        model,  # type: ignore[arg-type]
        offset_sec=spec.offset_sec,
        duration_sec=spec.duration_sec,
        source_root=source_root,
        artifacts_root=artifacts_root,
        output_dir=output_dir,
    )


def _row(spec: WindowSpec, manifests: dict[str, dict[str, Any]]) -> dict[str, Any]:
    baseline = manifests["raw_start_reset"]["fusion_summary"]
    candidate = manifests["final_candidate"]["fusion_summary"]
    base_raw = float(baseline["rmse_gps1_nodes"]["rmse_u"])
    cand_raw = float(candidate["rmse_gps1_nodes"]["rmse_u"])
    base_output = float(baseline["rmse_gps1_10hz"]["rmse_u"])
    cand_output = float(candidate["rmse_gps1_10hz"]["rmse_u"])
    count_keys = (
        "nodes_saved",
        "used_starlink_sparse_gps",
        "used_vision_velocity",
        "used_baro",
    )
    counts_equal = all(int(baseline[key]) == int(candidate[key]) for key in count_keys)
    return {
        "window": spec.name,
        "role": spec.role,
        "offset_sec": spec.offset_sec,
        "duration_sec": spec.duration_sec,
        "actual_t_start": float(baseline["active_t_start"]),
        "actual_t_end": float(baseline["active_t_end"]),
        "baseline_raw_rmse_u_m": base_raw,
        "candidate_raw_rmse_u_m": cand_raw,
        "raw_improvement_percent": 100.0 * (base_raw - cand_raw) / base_raw,
        "raw_candidate_wins": cand_raw < base_raw,
        "baseline_output_rmse_u_m": base_output,
        "candidate_output_rmse_u_m": cand_output,
        "output_improvement_percent": 100.0 * (base_output - cand_output) / base_output,
        "output_candidate_wins": cand_output < base_output,
        "baseline_starlink_updates": int(baseline["used_starlink_sparse_gps"]),
        "candidate_starlink_updates": int(candidate["used_starlink_sparse_gps"]),
        "baseline_failed_updates": int(baseline["failed_updates"]),
        "candidate_failed_updates": int(candidate["failed_updates"]),
        "measurement_counts_equal": counts_equal,
    }


def _decision(rows: list[dict[str, Any]]) -> dict[str, Any]:
    primary = [row for row in rows if row["role"] == "primary"]
    improvements = [float(row["raw_improvement_percent"]) for row in primary]
    win_count = sum(bool(row["raw_candidate_wins"]) for row in primary)
    failed_updates = sum(
        int(row["baseline_failed_updates"]) + int(row["candidate_failed_updates"])
        for row in rows
    )
    unequal_counts = sum(not bool(row["measurement_counts_equal"]) for row in rows)
    checks = {
        "enough_raw_rmse_wins": win_count >= ACCEPTANCE["minimum_raw_rmse_wins"],
        "median_raw_improvement": median(improvements)
        >= ACCEPTANCE["minimum_median_raw_improvement_percent"],
        "worst_raw_regression": min(improvements)
        >= -ACCEPTANCE["maximum_worst_raw_regression_percent"],
        "zero_failed_updates": failed_updates == 0,
        "equal_measurement_counts": unequal_counts == 0,
    }
    return {
        "accepted": all(checks.values()),
        "checks": checks,
        "primary_raw_win_count": win_count,
        "primary_window_count": len(primary),
        "median_primary_raw_improvement_percent": median(improvements),
        "worst_primary_raw_improvement_percent": min(improvements),
        "best_primary_raw_improvement_percent": max(improvements),
        "total_failed_updates": failed_updates,
        "windows_with_unequal_measurement_counts": unequal_counts,
        "interpretation_limit": (
            "Blocked replay on one flight can detect temporal brittleness but cannot "
            "prove generalization to a new flight, sensor, weather, or camera geometry."
        ),
    }


def run_temporal_validation(
    output_root: str | Path,
    *,
    source_root: str | Path | None = None,
    artifacts_root: str | Path | None = None,
    include_short_stress: bool = True,
    reuse_existing: bool = False,
) -> dict[str, Any]:
    root = Path(output_root).expanduser().resolve()
    specs = LONG_WINDOWS + (SHORT_WINDOWS if include_short_stress else ())
    manifests: dict[str, dict[str, dict[str, Any]]] = {}
    rows: list[dict[str, Any]] = []
    for spec in specs:
        manifests[spec.name] = {}
        for model in WINDOW_MODELS:
            manifests[spec.name][model] = _run_or_reuse(
                root,
                spec,
                model,
                source_root=Path(source_root) if source_root else None,
                artifacts_root=Path(artifacts_root) if artifacts_root else None,
                reuse_existing=reuse_existing,
            )
        rows.append(_row(spec, manifests[spec.name]))

    report = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "kind": "blocked chronological cold-start replay",
            "parameters_retuned_during_validation": False,
            "dense_gps_graph_factor": False,
            "window_specs": [asdict(spec) for spec in specs],
        },
        "acceptance_criteria": ACCEPTANCE,
        "decision": _decision(rows),
        "rows": rows,
    }
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "temporal_validation_summary.json"
    csv_path = root / "temporal_validation_summary.csv"
    if json_path.exists() or csv_path.exists():
        raise FileExistsError(
            "Validation summary already exists; use a fresh output root or remove "
            "only the obsolete summary files explicitly."
        )
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--artifacts-root", type=Path)
    parser.add_argument("--skip-short-stress", action="store_true")
    parser.add_argument("--reuse-existing", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run_temporal_validation(
        args.output_root,
        source_root=args.source_root,
        artifacts_root=args.artifacts_root,
        include_short_stress=not args.skip_short_stress,
        reuse_existing=args.reuse_existing,
    )
    print(json.dumps(report["decision"], indent=2))
    return 0 if report["decision"]["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
