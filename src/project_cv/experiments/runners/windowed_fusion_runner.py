"""Cold-start temporal-window replay for baseline and final height candidate.

Every window restarts motion initialization, raw barometer alignment, causal
filters and (for the candidate) bias states.  The dense GPS reference supplies
the existing baseline p0/v0 initialization at the new start and is otherwise
used only by the notebook's offline diagnostics, never as a graph factor.
"""

from __future__ import annotations

import argparse
import ast
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time
from types import CodeType
from typing import Any, Final, Literal, Sequence

from project_cv.experiments.runners import fusion_experiment_runner as baseline_runner
from project_cv.experiments.runners.bias_fusion_experiment_runner import BIAS_OUTPUTS
from project_cv.experiments.runners.reversion_fusion_experiment_runner import (
    _OUTPUT_POLICY_SOURCE,
    _REVERSION_SOURCE,
)


WindowModel = Literal["raw_start_reset", "final_candidate"]
WINDOW_MODELS: Final[tuple[WindowModel, ...]] = (
    "raw_start_reset",
    "final_candidate",
)
WINDOW_OUTPUTS: Final = ("output_height_policy.json", "window_contract.json")


@dataclass(frozen=True)
class PreparedWindowExperiment:
    code: CodeType
    model: WindowModel
    offset_sec: float
    duration_sec: float
    notebook_path: Path
    notebook_sha256: str
    source_cell_sha256: str
    transformations: tuple[str, ...]


def _validate_window(offset_sec: float, duration_sec: float) -> None:
    if not math.isfinite(offset_sec) or offset_sec < 0.0:
        raise ValueError("offset_sec must be finite and non-negative")
    if not math.isfinite(duration_sec) or duration_sec <= 0.0:
        raise ValueError("duration_sec must be finite and greater than zero")


def prepare_window_experiment(
    notebook_path: str | Path,
    model: WindowModel,
    *,
    offset_sec: float,
    duration_sec: float,
) -> PreparedWindowExperiment:
    """Compile one cold-start replay window from the frozen fusion cell."""

    if model not in WINDOW_MODELS:
        raise ValueError(f"Unknown window model: {model}")
    _validate_window(offset_sec, duration_sec)
    notebook = Path(notebook_path).expanduser().resolve()
    source, notebook_sha = baseline_runner._load_fusion_cell(notebook)
    source_sha = hashlib.sha256(source.encode("utf-8")).hexdigest()
    tree = ast.parse(
        source,
        filename=f"{notebook}#{baseline_runner.BASELINE_CELL_ID}",
    )

    for name, value in (
        ("DEBUG_SHORT_RUN", True),
        ("DEBUG_START_OFFSET_SEC", float(offset_sec)),
        ("DEBUG_DURATION_SEC", float(duration_sec)),
        ("DEBUG_MAX_NODES", 100_000),
        ("HEIGHT_SMOOTH_MEDIAN_WIN", 1),
        ("HEIGHT_SMOOTH_MEAN_WIN", 1),
        ("HEIGHT_LPF_TAU_SEC", 0.2),
    ):
        baseline_runner._replace_top_level_constant(tree, name=name, value=value)

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
    replacement = ast.parse(baseline_runner._RAW_START_RESET_SOURCE).body
    transformations = [
        f"cold-start replay offset={float(offset_sec):.3f} s duration={float(duration_sec):.3f} s",
        "reset raw barometer alignment and all causal filters at window start",
        "use identical causal output IIR tau=0.2 s for both compared models",
    ]
    if model == "final_candidate":
        replacement += ast.parse(_REVERSION_SOURCE).body
        transformations.append(
            "apply robust mean-reverting bias+speed^2 height correction"
        )
    replacement += ast.parse(baseline_runner._SAVE_BARO_GRAPH_SOURCE).body
    replacement += ast.parse(_OUTPUT_POLICY_SOURCE).body
    index = alignment_indices[0]
    tree.body[index : index + 1] = replacement
    ast.fix_missing_locations(tree)
    code = compile(
        tree,
        filename=f"{notebook}#window:{model}:{offset_sec}:{duration_sec}",
        mode="exec",
    )
    return PreparedWindowExperiment(
        code=code,
        model=model,
        offset_sec=float(offset_sec),
        duration_sec=float(duration_sec),
        notebook_path=notebook,
        notebook_sha256=notebook_sha,
        source_cell_sha256=source_sha,
        transformations=tuple(transformations),
    )


def run_window_experiment(
    model: WindowModel,
    *,
    offset_sec: float,
    duration_sec: float,
    source_root: str | Path | None = None,
    artifacts_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Compile or execute one isolated cold-start validation window."""

    _validate_window(offset_sec, duration_sec)
    variant_label = f"window_{model}_{offset_sec:.0f}_{duration_sec:.0f}"
    paths = baseline_runner._resolve_paths(
        source_root=source_root,
        artifacts_root=artifacts_root,
        output_dir=output_dir,
        variant=variant_label,  # type: ignore[arg-type]
    )
    prepared = prepare_window_experiment(
        paths.source_root / baseline_runner.BASELINE_NOTEBOOK,
        model,
        offset_sec=offset_sec,
        duration_sec=duration_sec,
    )
    expected_outputs = baseline_runner.EXPECTED_OUTPUTS + WINDOW_OUTPUTS
    if model == "final_candidate":
        expected_outputs += BIAS_OUTPUTS
    input_files = sorted(path for path in paths.work_dir.iterdir() if path.is_file())
    window_contract = {
        "schema_version": 1,
        "model": model,
        "requested_offset_sec": float(offset_sec),
        "requested_duration_sec": float(duration_sec),
        "cold_start": True,
        "barometer_filter_reset": True,
        "bias_state_reset": model == "final_candidate",
        "initial_position_policy": "existing causal past GPS sample at window start",
        "dense_gps_graph_factor": False,
        "output_height_policy": "causal IIR tau=0.2 s, no rolling windows",
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "variant": variant_label,
        "model": model,
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
        "window_contract": window_contract,
        "input_sha256": {
            path.name: baseline_runner._sha256(path) for path in input_files
        },
        "dense_gps_usage": (
            "existing baseline p0/v0 initialization at the restarted window; "
            "diagnostic scoring only thereafter; never added as a graph factor"
        ),
    }
    if dry_run:
        return manifest

    paths.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = paths.output_dir / "experiment_manifest.json"
    log_path = paths.output_dir / "fusion_stdout.log"
    (paths.output_dir / "window_contract.json").write_text(
        json.dumps(window_contract, indent=2),
        encoding="utf-8",
    )
    started = time.perf_counter()
    namespace: dict[str, Any] = {
        "__name__": "__windowed_fusion_experiment__",
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
                "Window replay finished but outputs are missing: " + ", ".join(missing)
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
        actual_duration = float(fusion_summary["active_t_end"]) - float(
            fusion_summary["active_t_start"]
        )
        if actual_duration + 1e-6 < min(float(duration_sec), 1.0):
            raise baseline_runner.FusionExperimentError(
                f"Window is unexpectedly short: {actual_duration:.3f} s"
            )
        manifest["status"] = "completed"
        manifest["elapsed_sec"] = time.perf_counter() - started
        manifest["fusion_summary"] = fusion_summary
        manifest["output_files"] = {
            name: {
                "bytes": (paths.output_dir / name).stat().st_size,
                "sha256": baseline_runner._sha256(paths.output_dir / name),
            }
            for name in expected_outputs
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
    parser.add_argument("--model", required=True, choices=WINDOW_MODELS)
    parser.add_argument("--offset-sec", type=float, required=True)
    parser.add_argument("--duration-sec", type=float, required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--artifacts-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_window_experiment(
        args.model,
        offset_sec=args.offset_sec,
        duration_sec=args.duration_sec,
        source_root=args.source_root,
        artifacts_root=args.artifacts_root,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
