"""Assembled, provenance-locked entry point for final fusion runs.

``safe`` is the supported default.  ``adaptive_experimental`` exposes the
confidence-gated research candidate without silently promoting it to the
default.  Both modes compile the hash-locked GTSAM cell, use isolated output
directories and refuse to overwrite prior results.
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
from typing import Any, Sequence

from project_cv.experiments.runners import fusion_experiment_runner as baseline_runner
from project_cv.experiments.runners.bias_fusion_experiment_runner import BIAS_OUTPUTS
from project_cv.final.config import (
    ADAPTIVE_HEIGHT_POLICY,
    DEFAULT_MODE,
    FINAL_MODES,
    SAFE_HEIGHT_POLICY,
    FinalMode,
)
from project_cv.experiments.runners.reversion_fusion_experiment_runner import (
    _OUTPUT_POLICY_SOURCE,
    _REVERSION_SOURCE,
)


FINAL_COMMON_OUTPUTS = ("output_height_policy.json", "final_pipeline_config.json")
FINAL_ADAPTIVE_OUTPUTS = BIAS_OUTPUTS + ("confidence_policy_summary.json",)


_CONFIDENCE_SOURCE = r'''
from project_cv.experiments.height.confidence_fallback import POLICY_CANDIDATES, apply_confidence_to_correction

_final_confidence_policy = next(
    policy for policy in POLICY_CANDIDATES if policy.name == "balanced"
)
_final_applied_correction, _final_confidence_metadata = apply_confidence_to_correction(
    baro["tr"].to_numpy(dtype=float),
    baro["estimated_total_error_m"].to_numpy(dtype=float),
    _barometer_bias_result.trace,
    config=_final_confidence_policy,
)
baro["u_graph_before_confidence"] = baro["u_graph"].to_numpy(dtype=float)
baro["confidence_applied_correction_m"] = _final_applied_correction
baro["u_graph"] = (
    baro["u_graph_uncorrected"].to_numpy(dtype=float)
    - _final_applied_correction
)
(FUSION_RESULTS_PATH / "confidence_policy_summary.json").write_text(
    json.dumps(_final_confidence_metadata, indent=2),
    encoding="utf-8",
)
print("Confidence fallback policy:", _final_confidence_policy.name)
print("  mean confidence =", _final_confidence_metadata["mean_confidence"])
print("  active fraction =", _final_confidence_metadata["active_fraction"])
'''


_SAVE_FINAL_BAROMETER_SOURCE = r'''
if USE_BARO_Z and len(baro) and "u_graph" in baro.columns:
    _final_baro_columns = [
        name
        for name in (
            "tr",
            "t",
            "alt",
            "u_corr",
            "u_aligned_raw",
            "u_smooth",
            "u_graph_uncorrected",
            "estimated_bias_m",
            "estimated_speed_gain_m",
            "horizontal_speed_proxy_mps",
            "estimated_total_error_m",
            "u_graph_before_confidence",
            "confidence_applied_correction_m",
            "u_graph",
        )
        if name in baro.columns
    ]
    baro.loc[:, _final_baro_columns].to_csv(
        FUSION_RESULTS_PATH / "baro_graph_input.csv",
        index=False,
    )
'''


@dataclass(frozen=True)
class PreparedFinalFusion:
    code: CodeType
    mode: FinalMode
    notebook_path: Path
    notebook_sha256: str
    source_cell_sha256: str
    transformations: tuple[str, ...]


def prepare_final_fusion(
    notebook_path: str | Path,
    mode: FinalMode = DEFAULT_MODE,
) -> PreparedFinalFusion:
    if mode not in FINAL_MODES:
        raise ValueError(f"Unknown final fusion mode: {mode}")
    notebook = Path(notebook_path).expanduser().resolve()
    source, notebook_sha = baseline_runner._load_fusion_cell(notebook)
    source_sha = hashlib.sha256(source.encode("utf-8")).hexdigest()
    tree = ast.parse(source, filename=f"{notebook}#final:{mode}")
    for name, value in (
        ("HEIGHT_SMOOTH_MEDIAN_WIN", 1),
        ("HEIGHT_SMOOTH_MEAN_WIN", 1),
        ("HEIGHT_LPF_TAU_SEC", 0.2),
    ):
        baseline_runner._replace_top_level_constant(tree, name=name, value=value)

    indices = [
        index
        for index, statement in enumerate(tree.body)
        if baseline_runner._is_alignment_block(statement)
    ]
    if len(indices) != 1:
        raise baseline_runner.FusionExperimentError(
            "Expected exactly one top-level barometer alignment block; "
            f"found {len(indices)}"
        )
    replacement = ast.parse(baseline_runner._RAW_START_RESET_SOURCE).body
    transformations = [
        "align raw causal barometer sample to p0 and reset causal filters",
        "replace final 11/11 + 0.8 s cascade with causal IIR tau=0.2 s",
    ]
    if mode == "adaptive_experimental":
        replacement += ast.parse(_REVERSION_SOURCE).body
        replacement += ast.parse(_CONFIDENCE_SOURCE).body
        transformations.extend(
            [
                "propose robust mean-reverting bias + speed^2 correction",
                "gate, cap and blend correction with balanced causal confidence policy",
            ]
        )
    replacement += ast.parse(_SAVE_FINAL_BAROMETER_SOURCE).body
    replacement += ast.parse(_OUTPUT_POLICY_SOURCE).body
    index = indices[0]
    tree.body[index : index + 1] = replacement
    ast.fix_missing_locations(tree)
    return PreparedFinalFusion(
        code=compile(tree, filename=f"{notebook}#final:{mode}", mode="exec"),
        mode=mode,
        notebook_path=notebook,
        notebook_sha256=notebook_sha,
        source_cell_sha256=source_sha,
        transformations=tuple(transformations),
    )


def run_final_fusion(
    mode: FinalMode = DEFAULT_MODE,
    *,
    source_root: str | Path | None = None,
    artifacts_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    paths = baseline_runner._resolve_paths(
        source_root=source_root,
        artifacts_root=artifacts_root,
        output_dir=output_dir,
        variant=f"final_{mode}",  # type: ignore[arg-type]
    )
    prepared = prepare_final_fusion(
        paths.source_root / baseline_runner.BASELINE_NOTEBOOK,
        mode,
    )
    policy = SAFE_HEIGHT_POLICY if mode == "safe" else ADAPTIVE_HEIGHT_POLICY
    expected = baseline_runner.EXPECTED_OUTPUTS + FINAL_COMMON_OUTPUTS
    if mode == "adaptive_experimental":
        expected += FINAL_ADAPTIVE_OUTPUTS
    input_files = sorted(path for path in paths.work_dir.iterdir() if path.is_file())
    final_config = {
        "schema_version": 1,
        "mode": mode,
        "default_mode": DEFAULT_MODE,
        "height_policy": policy,
        "dense_gps_graph_factor": False,
        "core_status": "hash-locked notebook cell behind stable Python entry point",
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "pipeline": "project_cv.final",
        "mode": mode,
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
        "final_config": final_config,
        "input_sha256": {
            path.name: baseline_runner._sha256(path) for path in input_files
        },
    }
    if dry_run:
        return manifest

    paths.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = paths.output_dir / "final_pipeline_manifest.json"
    log_path = paths.output_dir / "fusion_stdout.log"
    (paths.output_dir / "final_pipeline_config.json").write_text(
        json.dumps(final_config, indent=2),
        encoding="utf-8",
    )
    namespace: dict[str, Any] = {
        "__name__": "__project_cv_final_fusion__",
        "WORK_DIR": paths.work_dir,
        "FUSION_OUTPUT_DIR": paths.output_dir,
        "display": lambda *_args, **_kwargs: None,
    }
    os.environ.setdefault("MPLBACKEND", "Agg")
    started = time.perf_counter()
    try:
        with log_path.open("w", encoding="utf-8") as log:
            with redirect_stdout(log), redirect_stderr(log):
                exec(prepared.code, namespace)
        missing = [name for name in expected if not (paths.output_dir / name).is_file()]
        if missing:
            raise baseline_runner.FusionExperimentError(
                "Final fusion outputs are missing: " + ", ".join(missing)
            )
        summary = json.loads(
            (paths.output_dir / "gtsam_incremental_fixedlag_summary.json").read_text(
                encoding="utf-8"
            )
        )
        if int(summary.get("failed_updates", -1)) != 0:
            raise baseline_runner.FusionExperimentError(
                f"Final fusion reported failed_updates={summary.get('failed_updates')}"
            )
        manifest["status"] = "completed"
        manifest["elapsed_sec"] = time.perf_counter() - started
        manifest["fusion_summary"] = summary
        manifest["output_files"] = {
            name: {
                "bytes": (paths.output_dir / name).stat().st_size,
                "sha256": baseline_runner._sha256(paths.output_dir / name),
            }
            for name in expected
        }
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
    parser.add_argument("--mode", choices=FINAL_MODES, default=DEFAULT_MODE)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--artifacts-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_final_fusion(
        args.mode,
        source_root=args.source_root,
        artifacts_root=args.artifacts_root,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
