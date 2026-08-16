"""Run mean-reverting barometer and low-lag output-height experiments.

The graph cell remains hash-verified and frozen.  Only the barometer input
construction and output-height policy are changed in memory; every run writes
to a new directory and refuses to overwrite existing artifacts.
"""

from __future__ import annotations

import argparse
import ast
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
from types import CodeType
from typing import Any, Final, Literal, Sequence

from project_cv.experiments.runners import fusion_experiment_runner as baseline_runner
from project_cv.experiments.runners.bias_fusion_experiment_runner import BIAS_OUTPUTS


ReversionVariant = Literal[
    "raw_start_reset_bias_speed_reversion_export_raw",
    "raw_start_reset_bias_speed_reversion_light_smoothing",
]
REVERSION_VARIANTS: Final[tuple[ReversionVariant, ...]] = (
    "raw_start_reset_bias_speed_reversion_export_raw",
    "raw_start_reset_bias_speed_reversion_light_smoothing",
)
REVERSION_OUTPUTS: Final = BIAS_OUTPUTS + ("output_height_policy.json",)


@dataclass(frozen=True)
class PreparedReversionExperiment:
    code: CodeType
    variant: ReversionVariant
    notebook_path: Path
    notebook_sha256: str
    source_cell_sha256: str
    transformations: tuple[str, ...]


_REVERSION_SOURCE: Final = r'''
from project_cv.experiments.height.barometer_bias_reversion import apply_causal_reverting_bias_speed

if not HAS_VISION_VEL_FILE or not len(vision_vel_raw):
    raise RuntimeError("Mean-reverting bias experiment requires vision velocity")
if "tr_corr" not in vision_vel_raw.columns:
    vision_vel_raw = vision_vel_raw.copy()
    vision_vel_raw["tr_corr"] = (
        vision_vel_raw["tr"].to_numpy(dtype=float) - float(FUSION_VISION_VEL_DELAY_SEC)
    )

_barometer_bias_result = apply_causal_reverting_bias_speed(
    baro,
    slc,
    vision_vel_raw,
)
baro = _barometer_bias_result.frame
_barometer_bias_result.trace.to_csv(
    FUSION_RESULTS_PATH / "barometer_bias_trace.csv",
    index=False,
)
(FUSION_RESULTS_PATH / "barometer_bias_summary.json").write_text(
    json.dumps(_barometer_bias_result.metadata, indent=2),
    encoding="utf-8",
)
print("BARO robust mean-reverting bias + speed^2 experiment:")
print("  accepted Starlink =", _barometer_bias_result.metadata["accepted_starlink_updates"])
print("  rejected Starlink =", _barometer_bias_result.metadata["rejected_starlink_updates"])
'''


_OUTPUT_POLICY_SOURCE: Final = r'''
_output_height_policy = {
    "smooth_height_output": bool(SMOOTH_HEIGHT_OUTPUT),
    "use_smooth_height_for_export": bool(USE_SMOOTH_HEIGHT_FOR_EXPORT),
    "median_window_samples": int(HEIGHT_SMOOTH_MEDIAN_WIN),
    "mean_window_samples": int(HEIGHT_SMOOTH_MEAN_WIN),
    "use_causal_iir": bool(HEIGHT_USE_CAUSAL_IIR_LPF),
    "iir_tau_sec": float(HEIGHT_LPF_TAU_SEC),
    "graph_hz": float(GRAPH_HZ),
}
(FUSION_RESULTS_PATH / "output_height_policy.json").write_text(
    json.dumps(_output_height_policy, indent=2),
    encoding="utf-8",
)
'''


def prepare_reversion_experiment(
    notebook_path: str | Path,
    variant: ReversionVariant,
) -> PreparedReversionExperiment:
    """Compile a mean-reverting variant from the frozen GTSAM cell."""

    if variant not in REVERSION_VARIANTS:
        raise ValueError(f"Unknown reversion experiment variant: {variant}")
    notebook = Path(notebook_path).expanduser().resolve()
    source, notebook_sha = baseline_runner._load_fusion_cell(notebook)
    source_sha = hashlib.sha256(source.encode("utf-8")).hexdigest()
    tree = ast.parse(
        source,
        filename=f"{notebook}#{baseline_runner.BASELINE_CELL_ID}",
    )

    transformations = [
        "align raw causal u_corr to p0, then restart median/mean/IIR at active start",
        "estimate robust bias+speed^2 error with 60 s / 240 s state mean reversion",
    ]
    if variant == "raw_start_reset_bias_speed_reversion_export_raw":
        baseline_runner._replace_top_level_constant(
            tree,
            name="USE_SMOOTH_HEIGHT_FOR_EXPORT",
            value=False,
        )
        transformations.append("export raw graph height without final smoothing")
    else:
        baseline_runner._replace_top_level_constant(
            tree,
            name="HEIGHT_SMOOTH_MEDIAN_WIN",
            value=1,
        )
        baseline_runner._replace_top_level_constant(
            tree,
            name="HEIGHT_SMOOTH_MEAN_WIN",
            value=1,
        )
        baseline_runner._replace_top_level_constant(
            tree,
            name="HEIGHT_LPF_TAU_SEC",
            value=0.2,
        )
        transformations.append(
            "replace final 11/11 + 0.8 s cascade with causal IIR tau=0.2 s"
        )

    alignment_indices = [
        index
        for index, statement in enumerate(tree.body)
        if baseline_runner._is_alignment_block(statement)
    ]
    if len(alignment_indices) != 1:
        raise baseline_runner.FusionExperimentError(
            "Expected exactly one top-level barometer alignment block; "
            f"found {len(alignment_indices)}"
        )
    replacement = (
        ast.parse(baseline_runner._RAW_START_RESET_SOURCE).body
        + ast.parse(_REVERSION_SOURCE).body
        + ast.parse(baseline_runner._SAVE_BARO_GRAPH_SOURCE).body
        + ast.parse(_OUTPUT_POLICY_SOURCE).body
    )
    index = alignment_indices[0]
    tree.body[index : index + 1] = replacement
    ast.fix_missing_locations(tree)
    code = compile(tree, filename=f"{notebook}#{variant}", mode="exec")
    transformations.extend(
        [
            "persist causal bias state/update trace",
            "persist exact corrected barometer graph input and output-height policy",
        ]
    )
    return PreparedReversionExperiment(
        code=code,
        variant=variant,
        notebook_path=notebook,
        notebook_sha256=notebook_sha,
        source_cell_sha256=source_sha,
        transformations=tuple(transformations),
    )


def run_reversion_experiment(
    variant: ReversionVariant,
    *,
    source_root: str | Path | None = None,
    artifacts_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Compile or run one isolated mean-reversion/smoothing variant."""

    paths = baseline_runner._resolve_paths(
        source_root=source_root,
        artifacts_root=artifacts_root,
        output_dir=output_dir,
        variant=variant,  # type: ignore[arg-type]
    )
    prepared = prepare_reversion_experiment(
        paths.source_root / baseline_runner.BASELINE_NOTEBOOK,
        variant,
    )
    expected_outputs = baseline_runner.EXPECTED_OUTPUTS + REVERSION_OUTPUTS
    input_files = sorted(path for path in paths.work_dir.iterdir() if path.is_file())
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "variant": variant,
        "status": "dry_run" if dry_run else "running",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(paths.source_root),
        "artifacts_root": str(paths.artifacts_root),
        "baseline_work_dir": str(paths.work_dir),
        "output_dir": str(paths.output_dir),
        "baseline_notebook": str(prepared.notebook_path),
        "baseline_notebook_sha256": prepared.notebook_sha256,
        "baseline_cell_id": baseline_runner.BASELINE_CELL_ID,
        "baseline_cell_sha256": prepared.source_cell_sha256,
        "transformations": list(prepared.transformations),
        "input_sha256": {
            path.name: baseline_runner._sha256(path) for path in input_files
        },
        "dense_gps_usage": (
            "unchanged from baseline for origin/calibration/p0/v0; evaluation only "
            "outside the correction estimator"
        ),
    }
    if dry_run:
        return manifest

    paths.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = paths.output_dir / "experiment_manifest.json"
    log_path = paths.output_dir / "fusion_stdout.log"
    started = time.perf_counter()
    namespace: dict[str, Any] = {
        "__name__": "__reversion_fusion_experiment__",
        "WORK_DIR": paths.work_dir,
        "FUSION_OUTPUT_DIR": paths.output_dir,
        "display": lambda *_args, **_kwargs: None,
    }
    os.environ.setdefault("MPLBACKEND", "Agg")

    try:
        with log_path.open("w", encoding="utf-8") as log:
            with redirect_stdout(log), redirect_stderr(log):
                exec(prepared.code, namespace)
        missing = [
            name for name in expected_outputs if not (paths.output_dir / name).is_file()
        ]
        if missing:
            raise baseline_runner.FusionExperimentError(
                "Fusion finished but expected outputs are missing: " + ", ".join(missing)
            )
        fusion_summary = json.loads(
            (paths.output_dir / "gtsam_incremental_fixedlag_summary.json").read_text(
                encoding="utf-8"
            )
        )
        if int(fusion_summary.get("failed_updates", -1)) != 0:
            raise baseline_runner.FusionExperimentError(
                f"Fusion reported failed_updates={fusion_summary.get('failed_updates')}"
            )
        manifest["status"] = "completed"
        manifest["elapsed_sec"] = time.perf_counter() - started
        manifest["output_files"] = {
            name: {
                "bytes": (paths.output_dir / name).stat().st_size,
                "sha256": baseline_runner._sha256(paths.output_dir / name),
            }
            for name in expected_outputs
        }
        manifest["fusion_summary"] = fusion_summary
        manifest["barometer_bias_summary"] = json.loads(
            (paths.output_dir / "barometer_bias_summary.json").read_text(
                encoding="utf-8"
            )
        )
        manifest["output_height_policy"] = json.loads(
            (paths.output_dir / "output_height_policy.json").read_text(
                encoding="utf-8"
            )
        )
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["elapsed_sec"] = time.perf_counter() - started
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        raise

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=REVERSION_VARIANTS)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--artifacts-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_reversion_experiment(
        args.variant,
        source_root=args.source_root,
        artifacts_root=args.artifacts_root,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
