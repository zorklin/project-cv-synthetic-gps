"""Run controlled GTSAM variants from the frozen baseline notebook.

The current fusion implementation still lives in the ``gtsam-fusion`` cell of
``02_sparse_gps_fusion.ipynb``.  Copying that 2,200-line cell for every small
experiment would create several subtly different algorithms.  This module
therefore treats the tracked cell as an immutable source, verifies its hash,
applies one small in-memory AST transformation, and writes every result to a
new experiment directory.

This is deliberately an experiment harness, not the final home of the fusion
algorithm.  Once a change is accepted, the fusion implementation should be
moved to a normal Python module and covered by regression tests.
"""

from __future__ import annotations

import argparse
import ast
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
from types import CodeType
from typing import Any, Final, Literal, Sequence


Variant = Literal[
    "legacy_replay",
    "no_p0_realign",
    "raw_start_reset",
    "raw_start_reset_sl_z12",
    "raw_start_reset_sl_z8",
    "raw_start_reset_baro_s3",
    "raw_start_reset_sl_z12_baro_s3",
]

BASELINE_NOTEBOOK: Final = Path("notebooks/pipeline/02_sparse_gps_fusion.ipynb")
BASELINE_CELL_ID: Final = "gtsam-fusion"
BASELINE_CELL_SHA256: Final = (
    "f8cbbdb1e2155802e6127734c1773d6e47059affcb81fd9d4a18df9e82054d20"
)
EXPECTED_OUTPUTS: Final = (
    "gtsam_incremental_fixedlag_nodes.csv",
    "synthetic_gps_5hz_gtsam_fixedlag.csv",
    "gtsam_incremental_fixedlag_summary.json",
    "baro_graph_input.csv",
)


class FusionExperimentError(RuntimeError):
    """Raised when an experiment would be unsafe or non-reproducible."""


@dataclass(frozen=True)
class PreparedExperiment:
    """Compiled experiment and provenance captured before execution."""

    code: CodeType
    variant: Variant
    notebook_path: Path
    notebook_sha256: str
    source_cell_sha256: str
    transformations: tuple[str, ...]


@dataclass(frozen=True)
class ExperimentPaths:
    """Validated read-only inputs and isolated output directory."""

    source_root: Path
    artifacts_root: Path
    work_dir: Path
    output_dir: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_below(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return path != root


def _resolve_paths(
    *,
    source_root: str | Path | None,
    artifacts_root: str | Path | None,
    output_dir: str | Path | None,
    variant: Variant,
) -> ExperimentPaths:
    source_value = source_root or os.environ.get("PROJECT_CV_SOURCE")
    artifacts_value = artifacts_root or os.environ.get("PROJECT_CV_ARTIFACTS")
    if not source_value or not artifacts_value:
        raise FusionExperimentError(
            "PROJECT_CV_SOURCE and PROJECT_CV_ARTIFACTS must be set, or their "
            "paths must be passed explicitly"
        )

    source = Path(source_value).expanduser().resolve()
    artifacts = Path(artifacts_value).expanduser().resolve()
    work = (artifacts / "fusion_v1" / "work").resolve()

    if output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = (
            artifacts / "height_experiments_v1" / f"{stamp}_{variant}" / "gtsam"
        ).resolve()
    else:
        output = Path(output_dir).expanduser().resolve()

    notebook = source / BASELINE_NOTEBOOK
    required_inputs = (
        work / "calibration.json",
        work / "imu_timebase.csv",
        work / "euler_timebase.csv",
        work / "starlink_corrected_enu.csv",
        work / "baro_corrected.csv",
        work / "gps_ref_enu.csv",
        work / "vision_velocity_raw_frd.csv",
    )

    if not source.is_dir():
        raise FileNotFoundError(f"Project source does not exist: {source}")
    if not artifacts.is_dir():
        raise FileNotFoundError(f"Artifacts root does not exist: {artifacts}")
    if not notebook.is_file():
        raise FileNotFoundError(f"Baseline notebook does not exist: {notebook}")
    missing = [str(path) for path in required_inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing prepared fusion inputs: " + ", ".join(missing))

    if not _is_below(output, artifacts):
        raise FusionExperimentError("Experiment output must be below artifacts root")
    protected = {
        (artifacts / "fusion_v1").resolve(),
        (artifacts / "fusion_v1" / "gtsam").resolve(),
        work,
    }
    if output in protected or any(_is_below(output, path) for path in protected):
        raise FusionExperimentError(
            "Experiment output must not overlap the protected fusion_v1 baseline"
        )
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Experiment output is not empty; refusing to overwrite it: {output}"
        )

    return ExperimentPaths(source, artifacts, work, output)


def _load_fusion_cell(notebook_path: Path) -> tuple[str, str]:
    notebook_bytes = notebook_path.read_bytes()
    notebook_sha = hashlib.sha256(notebook_bytes).hexdigest()
    notebook = json.loads(notebook_bytes)

    matches = [
        cell
        for cell in notebook.get("cells", [])
        if cell.get("id") == BASELINE_CELL_ID and cell.get("cell_type") == "code"
    ]
    if len(matches) != 1:
        raise FusionExperimentError(
            f"Expected exactly one code cell with id {BASELINE_CELL_ID!r}; "
            f"found {len(matches)}"
        )
    source = "".join(matches[0].get("source", []))
    source_sha = hashlib.sha256(source.encode("utf-8")).hexdigest()
    if source_sha != BASELINE_CELL_SHA256:
        raise FusionExperimentError(
            "Frozen fusion cell changed. Review it and deliberately update the "
            f"experiment harness hash. Expected {BASELINE_CELL_SHA256}, got {source_sha}."
        )
    return source, notebook_sha


def _assignment_name(statement: ast.stmt) -> str | None:
    if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
        return None
    target = statement.targets[0]
    return target.id if isinstance(target, ast.Name) else None


def _replace_top_level_constant(
    tree: ast.Module,
    *,
    name: str,
    value: bool | int | float | str,
) -> None:
    matches = [statement for statement in tree.body if _assignment_name(statement) == name]
    if len(matches) != 1:
        raise FusionExperimentError(
            f"Expected one top-level assignment for {name}; found {len(matches)}"
        )
    assignment = matches[0]
    assert isinstance(assignment, ast.Assign)
    assignment.value = ast.copy_location(ast.Constant(value=value), assignment.value)


def _is_alignment_block(statement: ast.stmt) -> bool:
    if not isinstance(statement, ast.If):
        return False

    # Match AST structure instead of `ast.unparse()` text.  The unparser is
    # free to choose single or double quotes, so quote-specific substrings
    # made the dry-run report zero blocks even for the frozen baseline cell.
    referenced_names = {
        node.id for node in ast.walk(statement) if isinstance(node, ast.Name)
    }
    if not {"USE_BARO_Z", "BARO_ALIGN_TO_P0", "baro"}.issubset(
        referenced_names
    ):
        return False

    assigns_u_graph = any(
        isinstance(node, ast.Subscript)
        and isinstance(node.ctx, ast.Store)
        and isinstance(node.value, ast.Name)
        and node.value.id == "baro"
        and isinstance(node.slice, ast.Constant)
        and node.slice.value == "u_graph"
        for node in ast.walk(statement)
    )
    has_alignment_label = any(
        isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value == "BARO align:"
        for node in ast.walk(statement)
    )
    return assigns_u_graph and has_alignment_label


_SAVE_BARO_GRAPH_SOURCE: Final = r'''
if USE_BARO_Z and len(baro) and "u_graph" in baro.columns:
    _baro_graph_columns = [
        name
        for name in (
            "tr",
            "t",
            "alt",
            "u_corr",
            "u_aligned_raw",
            "u_smooth",
            "u_graph",
        )
        if name in baro.columns
    ]
    baro.loc[:, _baro_graph_columns].to_csv(
        FUSION_RESULTS_PATH / "baro_graph_input.csv",
        index=False,
    )
'''


_RAW_START_RESET_SOURCE: Final = r'''
if USE_BARO_Z and len(baro):
    _baro_times = baro["tr"].to_numpy(dtype=float)
    _baro_start_index = int(np.searchsorted(_baro_times, float(t_start), side="right") - 1)
    if _baro_start_index < 0:
        raise RuntimeError("No causal barometer sample exists at or before t_start")

    # Keep the final causal sample before t_start, discard irrelevant history,
    # and align the raw (not lagged/filtered) measurement to the existing p0.
    baro = baro.iloc[_baro_start_index:].copy().reset_index(drop=True)
    _baro_raw_at_start = float(baro["u_corr"].iloc[0])
    baro_offset = float(p0[2]) - _baro_raw_at_start
    baro["u_aligned_raw"] = baro["u_corr"].to_numpy(dtype=float) + baro_offset

    _baro_aligned_series = cast(pd.Series, baro["u_aligned_raw"])
    _baro_aligned_base = cast(
        pd.Series,
        _baro_aligned_series
        .rolling(window=int(BARO_SMOOTH_MEDIAN_WIN), center=False, min_periods=1)
        .median()
        .rolling(window=int(BARO_SMOOTH_MEAN_WIN), center=False, min_periods=1)
        .mean(),
    ).to_numpy(dtype=float)

    if BARO_USE_CAUSAL_IIR_LPF:
        baro["u_smooth"] = causal_iir_lpf_by_time(
            baro["tr"].to_numpy(dtype=float),
            _baro_aligned_base,
            BARO_LPF_TAU_SEC,
            y0=float(p0[2]),
        )
    else:
        baro["u_smooth"] = _baro_aligned_base
    baro["u_graph"] = baro["u_smooth"]

    print("BARO align (raw_start_reset experiment):")
    print("  p0[U]             =", float(p0[2]))
    print("  raw causal U      =", _baro_raw_at_start)
    print("  raw offset        =", baro_offset)
    print("  first filtered U  =", float(baro["u_smooth"].iloc[0]))
'''


def prepare_experiment(
    notebook_path: str | Path,
    variant: Variant,
) -> PreparedExperiment:
    """Verify and compile one controlled transformation of the fusion cell."""

    _ALL_VARIANTS = (
        "legacy_replay",
        "no_p0_realign",
        "raw_start_reset",
        "raw_start_reset_sl_z12",
        "raw_start_reset_sl_z8",
        "raw_start_reset_baro_s3",
        "raw_start_reset_sl_z12_baro_s3",
    )
    if variant not in _ALL_VARIANTS:
        raise ValueError(f"Unknown fusion experiment variant: {variant}")

    notebook = Path(notebook_path).expanduser().resolve()
    source, notebook_sha = _load_fusion_cell(notebook)
    source_sha = hashlib.sha256(source.encode("utf-8")).hexdigest()
    tree = ast.parse(source, filename=f"{notebook}#{BASELINE_CELL_ID}")
    transformations: list[str] = []

    if variant == "no_p0_realign":
        _replace_top_level_constant(tree, name="BARO_ALIGN_TO_P0", value=False)
        transformations.append("BARO_ALIGN_TO_P0: True -> False")

    # --- Sigma tuning for new variants ---
    _uses_raw_start_reset = variant.startswith("raw_start_reset")

    if variant in ("raw_start_reset_sl_z12", "raw_start_reset_sl_z12_baro_s3"):
        _replace_top_level_constant(tree, name="STARLINK_Z_SIGMA_M", value=12.0)
        transformations.append("STARLINK_Z_SIGMA_M: 100.0 -> 12.0")

    if variant == "raw_start_reset_sl_z8":
        _replace_top_level_constant(tree, name="STARLINK_Z_SIGMA_M", value=8.0)
        transformations.append("STARLINK_Z_SIGMA_M: 100.0 -> 8.0")

    if variant in ("raw_start_reset_baro_s3", "raw_start_reset_sl_z12_baro_s3"):
        _replace_top_level_constant(tree, name="BARO_Z_SIGMA_M", value=3.0)
        transformations.append("BARO_Z_SIGMA_M: 1.5 -> 3.0")

    alignment_indices = [
        index for index, statement in enumerate(tree.body) if _is_alignment_block(statement)
    ]
    if len(alignment_indices) != 1:
        raise FusionExperimentError(
            "Expected exactly one top-level barometer alignment block; "
            f"found {len(alignment_indices)}"
        )

    index = alignment_indices[0]
    save_statements = ast.parse(_SAVE_BARO_GRAPH_SOURCE).body
    if _uses_raw_start_reset:
        corrected_statements = ast.parse(_RAW_START_RESET_SOURCE).body
        tree.body[index : index + 1] = corrected_statements + save_statements
        transformations.append(
            "align raw causal u_corr to p0, then restart median/mean/IIR at active start"
        )
    else:
        tree.body[index + 1 : index + 1] = save_statements
    transformations.append("persist exact baro graph input to baro_graph_input.csv")

    ast.fix_missing_locations(tree)
    code = compile(tree, filename=f"{notebook}#{BASELINE_CELL_ID}:{variant}", mode="exec")
    return PreparedExperiment(
        code=code,
        variant=variant,
        notebook_path=notebook,
        notebook_sha256=notebook_sha,
        source_cell_sha256=source_sha,
        transformations=tuple(transformations),
    )


def run_fusion_experiment(
    variant: Variant,
    *,
    source_root: str | Path | None = None,
    artifacts_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Compile or execute one variant without mutating the baseline artifacts."""

    paths = _resolve_paths(
        source_root=source_root,
        artifacts_root=artifacts_root,
        output_dir=output_dir,
        variant=variant,
    )
    prepared = prepare_experiment(paths.source_root / BASELINE_NOTEBOOK, variant)

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
        "baseline_cell_id": BASELINE_CELL_ID,
        "baseline_cell_sha256": prepared.source_cell_sha256,
        "transformations": list(prepared.transformations),
        "input_sha256": {path.name: _sha256(path) for path in input_files},
        "dense_gps_usage": (
            "unchanged from baseline for origin/calibration/p0/v0; evaluation only "
            "outside the fusion cell"
        ),
    }
    if dry_run:
        return manifest

    paths.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = paths.output_dir / "experiment_manifest.json"
    log_path = paths.output_dir / "fusion_stdout.log"
    started = time.perf_counter()
    namespace: dict[str, Any] = {
        "__name__": "__fusion_experiment__",
        "WORK_DIR": paths.work_dir,
        "FUSION_OUTPUT_DIR": paths.output_dir,
        "display": lambda *_args, **_kwargs: None,
    }

    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        with log_path.open("w", encoding="utf-8") as log:
            with redirect_stdout(log), redirect_stderr(log):
                exec(prepared.code, namespace)
        missing_outputs = [
            name for name in EXPECTED_OUTPUTS if not (paths.output_dir / name).is_file()
        ]
        if missing_outputs:
            raise FusionExperimentError(
                "Fusion finished but expected outputs are missing: "
                + ", ".join(missing_outputs)
            )
        summary = json.loads(
            (paths.output_dir / "gtsam_incremental_fixedlag_summary.json").read_text(
                encoding="utf-8"
            )
        )
        if int(summary.get("failed_updates", -1)) != 0:
            raise FusionExperimentError(
                f"Fusion reported failed_updates={summary.get('failed_updates')}"
            )
        manifest["status"] = "completed"
        manifest["elapsed_sec"] = time.perf_counter() - started
        manifest["output_files"] = {
            name: {
                "bytes": (paths.output_dir / name).stat().st_size,
                "sha256": _sha256(paths.output_dir / name),
            }
            for name in EXPECTED_OUTPUTS
        }
        manifest["fusion_summary"] = summary
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
    parser.add_argument(
        "--variant",
        required=True,
        choices=(
            "legacy_replay",
            "no_p0_realign",
            "raw_start_reset",
            "raw_start_reset_sl_z12",
            "raw_start_reset_sl_z8",
            "raw_start_reset_baro_s3",
            "raw_start_reset_sl_z12_baro_s3",
        ),
    )
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--artifacts-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_fusion_experiment(
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
