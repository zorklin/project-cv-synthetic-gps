# Project structure after the experimental phase

## Supported path

```text
raw MCAP
  -> 01_build_vision_velocity.ipynb
  -> prepared fusion inputs / derived MCAP
  -> project_cv.final_cli --mode safe
  -> synthetic GPS CSV + manifest + diagnostics
```

The supported entry point is `src/project_cv/final/`. By default it runs the
`safe` mode: raw causal barometer alignment at startup, filter-state reset, and
a low-lag causal IIR filter for exported height.

The adaptive bias correction with confidence/fallback is available only through
the explicit `adaptive_experimental` mode. It did not pass the predefined
stability criteria and therefore cannot become the default accidentally.

## Notebook roles

| Notebook | Role | Part of the normal workflow |
|---|---|---|
| `notebooks/pipeline/00_environment_and_data_check.ipynb` | Checks WSL, ROS 2, Python, and data | Yes, before the first run or after rebuilding the environment |
| `notebooks/pipeline/01_build_vision_velocity.ipynb` | Thermal video -> LK flow -> gyro compensation -> velocity | Yes, when velocity must be rebuilt from the raw MCAP |
| `notebooks/pipeline/02_sparse_gps_fusion.ipynb` | Frozen source of the current GTSAM core | Do not run manually for a final result |
| `notebooks/pipeline/03_fusion_diagnostics.ipynb` | Read-only plots and checks for completed fusion output | Optional after a run |
| `notebooks/experiments/04_height_experiments.ipynb` | Research history for the height channel | No, experimental only |
| `notebooks/reference/teacher_colab_reference.ipynb` | Original teacher-provided Colab draft | No, reference only |

## Python package

### Supported interface

- `src/project_cv/final/config.py` defines the two explicitly classified modes
  and their status.
- `src/project_cv/final/runner.py` performs isolated execution, verifies the
  fusion-core hash, writes manifests, and refuses to overwrite artifacts.
- `src/project_cv/final_cli.py` is the recommended command-line interface.

### Reproducible preprocessing

- `src/project_cv/vision_velocity.py` builds velocity from camera frames and IMU
  data.
- `src/project_cv/preprocessing_validation.py` compares generated output with
  the teacher-derived MCAP.

### Research and validation

- `experiments/height/` contains the initial height filters, barometer-bias
  estimators, mean reversion, and confidence fallback.
- `experiments/runners/` contains controlled, hash-locked experiment runners.
- `experiments/validation/` contains sensor diagnostics, cold-start replay, and
  leave-one-temporal-block-out validation.

Research modules remain in the repository because they explain the final
decision and reproduce both positive and negative results. Normal users should
not import them directly.

## Data and artifacts

- `data/01_raw_k2r/` is the immutable raw input.
- `data/02_derived_with_velocity/` is the teacher-provided derived MCAP.
- `calibration/` contains camera intrinsics and the camera-to-IMU transform.
- `/home/fedor/project_cv_runtime/artifacts/` contains generated intermediate
  and final files. They are not duplicated in Git.

## Known technical limitation

The GTSAM core still physically resides in the hash-locked `gtsam-fusion` cell
of notebook `02`. The final runner exposes it through a stable Python API,
verifies its SHA-256 hash, and applies only controlled AST transformations.

This is reproducible, but a future refactor may move the approximately 2,200
lines into normal Python modules without changing the mathematics.
