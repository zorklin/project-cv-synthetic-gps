"""Leave-one-temporal-block-out validation of causal confidence policies."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import itertools
import json
import math
from pathlib import Path
from typing import Any, Final, Sequence

import numpy as np
import pandas as pd

from project_cv.experiments.height.confidence_fallback import (
    POLICY_CANDIDATES,
    ConfidencePolicyConfig,
    apply_confidence_to_correction,
)


ACCEPTANCE: Final = {
    "minimum_pooled_rmse_improvement_percent": 10.0,
    "minimum_heldout_wins": 5,
    "maximum_worst_heldout_regression_percent": 10.0,
}


def _causal_reference(reference: pd.DataFrame, query_time: np.ndarray) -> np.ndarray:
    ref_time = reference["tr"].to_numpy(dtype=float)
    ref_height = reference["u"].to_numpy(dtype=float)
    indices = np.searchsorted(ref_time, query_time, side="right") - 1
    result = np.full(len(query_time), np.nan, dtype=float)
    valid = indices >= 0
    result[valid] = ref_height[indices[valid]]
    return result


def _window_policy_scores(
    work_dir: Path,
    validation_root: Path,
) -> tuple[list[str], dict[str, dict[str, dict[str, Any]]]]:
    validation = json.loads(
        (validation_root / "temporal_validation_summary.json").read_text(
            encoding="utf-8"
        )
    )
    primary_rows = [row for row in validation["rows"] if row["role"] == "primary"]
    gps = pd.read_csv(work_dir / "gps_ref_enu.csv").sort_values("tr")
    scores: dict[str, dict[str, dict[str, Any]]] = {}

    for window_row in primary_rows:
        name = str(window_row["window"])
        start = float(window_row["actual_t_start"])
        end = float(window_row["actual_t_end"])
        baseline = pd.read_csv(
            validation_root / name / "raw_start_reset" / "gtsam" / "baro_graph_input.csv"
        )
        candidate = pd.read_csv(
            validation_root / name / "final_candidate" / "gtsam" / "baro_graph_input.csv"
        )
        trace = pd.read_csv(
            validation_root / name / "final_candidate" / "gtsam" / "barometer_bias_trace.csv"
        )
        baseline = baseline.loc[(baseline["tr"] >= start) & (baseline["tr"] <= end)]
        candidate = candidate.loc[(candidate["tr"] >= start) & (candidate["tr"] <= end)]
        joined = baseline[["tr", "u_graph"]].merge(
            candidate[["tr", "u_graph"]],
            on="tr",
            how="inner",
            suffixes=("_baseline", "_candidate"),
            validate="one_to_one",
        )
        times = joined["tr"].to_numpy(dtype=float)
        base_height = joined["u_graph_baseline"].to_numpy(dtype=float)
        proposed = base_height - joined["u_graph_candidate"].to_numpy(dtype=float)
        reference = _causal_reference(gps, times)
        valid = np.isfinite(reference) & np.isfinite(base_height) & np.isfinite(proposed)
        base_error = base_height[valid] - reference[valid]
        scores[name] = {
            "baseline": {
                "mse_m2": float(np.mean(base_error**2)),
                "rmse_m": float(np.sqrt(np.mean(base_error**2))),
                "n": int(np.sum(valid)),
            }
        }
        for policy in POLICY_CANDIDATES:
            applied, metadata = apply_confidence_to_correction(
                times,
                proposed,
                trace,
                config=policy,
            )
            corrected_error = base_height[valid] - applied[valid] - reference[valid]
            scores[name][policy.name] = {
                "mse_m2": float(np.mean(corrected_error**2)),
                "rmse_m": float(np.sqrt(np.mean(corrected_error**2))),
                "n": int(np.sum(valid)),
                "policy_runtime": metadata,
            }
    return [str(row["window"]) for row in primary_rows], scores


def _choose_policy(
    training_windows: list[str],
    scores: dict[str, dict[str, dict[str, Any]]],
) -> ConfidencePolicyConfig:
    ranked = []
    for priority, policy in enumerate(POLICY_CANDIDATES):
        training_mse = float(
            np.mean([scores[name][policy.name]["mse_m2"] for name in training_windows])
        )
        ranked.append((training_mse, priority, policy))
    return min(ranked, key=lambda item: (item[0], item[1]))[2]


def run_confidence_cross_validation(
    work_dir: str | Path,
    validation_root: str | Path,
) -> dict[str, Any]:
    work = Path(work_dir).expanduser().resolve()
    root = Path(validation_root).expanduser().resolve()
    windows, scores = _window_policy_scores(work, root)
    folds: list[dict[str, Any]] = []

    for heldout in windows:
        training = [name for name in windows if name != heldout]
        selected = _choose_policy(training, scores)
        baseline_mse = float(scores[heldout]["baseline"]["mse_m2"])
        policy_mse = float(scores[heldout][selected.name]["mse_m2"])
        baseline_rmse = math.sqrt(baseline_mse)
        policy_rmse = math.sqrt(policy_mse)
        folds.append(
            {
                "heldout_window": heldout,
                "selected_policy": selected.name,
                "selected_policy_config": asdict(selected),
                "baseline_baro_rmse_m": baseline_rmse,
                "heldout_baro_rmse_m": policy_rmse,
                "heldout_improvement_percent": (
                    100.0 * (baseline_rmse - policy_rmse) / baseline_rmse
                ),
                "heldout_wins": policy_rmse < baseline_rmse,
                "heldout_policy_runtime": scores[heldout][selected.name].get(
                    "policy_runtime", {}
                ),
            }
        )

    baseline_mse = np.asarray(
        [scores[fold["heldout_window"]]["baseline"]["mse_m2"] for fold in folds],
        dtype=float,
    )
    heldout_mse = np.asarray(
        [
            scores[fold["heldout_window"]][fold["selected_policy"]]["mse_m2"]
            for fold in folds
        ],
        dtype=float,
    )
    baseline_pooled = float(np.sqrt(np.mean(baseline_mse)))
    heldout_pooled = float(np.sqrt(np.mean(heldout_mse)))
    improvements = [float(fold["heldout_improvement_percent"]) for fold in folds]
    wins = sum(bool(fold["heldout_wins"]) for fold in folds)
    mse_difference = baseline_mse - heldout_mse
    observed = float(np.mean(mse_difference))
    permutations = np.asarray(
        [
            np.mean(mse_difference * np.asarray(signs, dtype=float))
            for signs in itertools.product((-1.0, 1.0), repeat=len(folds))
        ],
        dtype=float,
    )
    decision_checks = {
        "pooled_rmse_improvement": (
            100.0 * (baseline_pooled - heldout_pooled) / baseline_pooled
            >= ACCEPTANCE["minimum_pooled_rmse_improvement_percent"]
        ),
        "heldout_win_count": wins >= ACCEPTANCE["minimum_heldout_wins"],
        "worst_heldout_regression": min(improvements)
        >= -ACCEPTANCE["maximum_worst_heldout_regression_percent"],
    }
    all_data_policy = _choose_policy(windows, scores)
    report = {
        "schema_version": 1,
        "method": "leave-one-temporal-block-out policy selection",
        "leakage_controls": {
            "heldout_block_used_for_its_policy_selection": False,
            "candidate_policy_count": len(POLICY_CANDIDATES),
            "policy_candidates_predeclared": [asdict(policy) for policy in POLICY_CANDIDATES],
            "dense_gps_usage": "training-fold scoring and heldout evaluation only; never an online policy input",
            "remaining_limit": "all blocks are still from one flight and the underlying estimator was developed on it",
        },
        "acceptance_criteria": ACCEPTANCE,
        "folds": folds,
        "decision": {
            "accepted_as_large_robust_boost": all(decision_checks.values()),
            "checks": decision_checks,
            "heldout_win_count": wins,
            "heldout_window_count": len(folds),
            "worst_heldout_improvement_percent": min(improvements),
            "median_heldout_improvement_percent": float(np.median(improvements)),
            "baseline_pooled_baro_rmse_m": baseline_pooled,
            "heldout_pooled_baro_rmse_m": heldout_pooled,
            "pooled_baro_rmse_improvement_percent": (
                100.0 * (baseline_pooled - heldout_pooled) / baseline_pooled
            ),
            "exact_paired_permutation_two_sided_p": float(
                np.mean(np.abs(permutations) >= abs(observed) - 1e-15)
            ),
        },
        "all_data_selected_policy": asdict(all_data_policy),
        "all_window_policy_scores": scores,
    }
    json_path = root / "confidence_lobo_summary.json"
    csv_path = root / "confidence_lobo_folds.csv"
    if json_path.exists() or csv_path.exists():
        raise FileExistsError("Confidence cross-validation outputs already exist")
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        flat_rows = [
            {
                "heldout_window": fold["heldout_window"],
                "selected_policy": fold["selected_policy"],
                "baseline_baro_rmse_m": fold["baseline_baro_rmse_m"],
                "heldout_baro_rmse_m": fold["heldout_baro_rmse_m"],
                "heldout_improvement_percent": fold["heldout_improvement_percent"],
                "heldout_wins": fold["heldout_wins"],
            }
            for fold in folds
        ]
        writer = csv.DictWriter(stream, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--validation-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run_confidence_cross_validation(args.work_dir, args.validation_root)
    print(json.dumps(report["decision"], indent=2))
    print(json.dumps({"all_data_selected_policy": report["all_data_selected_policy"]}, indent=2))
    return 0 if report["decision"]["accepted_as_large_robust_boost"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
