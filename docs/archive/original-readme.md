# Historical notebook-centered project notes

> **Archive status:** this document preserves the state of the project before
> the final package and documentation reorganization. It is useful for history,
> but it is not the current execution guide. Use
> [`docs/usage/final-pipeline.md`](../usage/final-pipeline.md) for current
> commands.

## Original objective

Estimate position between sparse Starlink/GPS measurements using:

- optical flow from a downward-looking thermal camera;
- IMU measurements;
- range or height estimates;
- sparse absolute coordinates.

## Notebook roles at the time

- `00_environment_and_data_check.ipynb` provided an independent environment and
  data-readability check.
- `01_build_vision_velocity.ipynb` reproduced the preprocessing chain:
  thermal frames -> sparse LK -> pixel-flow filtering -> gyro compensation ->
  `/vision/velocity_frd` -> derived MCAP. Its default mode was a safe 15-second
  smoke test.
- `02_sparse_gps_fusion.ipynb` contained preflight checks, topic extraction,
  calibration, and the GTSAM fixed-lag baseline. It used the teacher-derived bag
  while preprocessing reproducibility was being established.
- `03_fusion_diagnostics.ipynb` performed read-only analysis of existing
  `fusion_v1` artifacts, including XY and height error, Starlink outliers,
  timing, and vision-factor use.
- `04_height_experiments.ipynb` investigated causal barometer alignment and
  height correction while preserving the baseline.
- The teacher-provided Colab notebook preserved historical outputs and upstream
  raw-to-LK-to-velocity experiments. It was not safe to execute locally with
  `Run All`.

The notebooks were later moved into `notebooks/pipeline/`,
`notebooks/experiments/`, and `notebooks/reference/`. See the current
[notebook index](../../notebooks/README.md).

## Python modules at the time

- `vision_velocity.py` split preprocessing into reusable `smoke`, `full`, and
  artifact-reuse stages.
- `preprocessing_validation.py` compared topics, types, counts, flow schema,
  timestamps, coordinate frames, and values against the teacher-derived MCAP.
- The original height experiment module reproduced the legacy causal filtering,
  barometer alignment, and corrected raw-at-start/filter-reset variant.
- The original fusion experiment runner loaded the hash-locked `gtsam-fusion`
  cell, applied controlled AST changes, and wrote each run into a new isolated
  directory.

These modules now live under `src/project_cv/` and are separated into
`final/` and `experiments/`.

`__pycache__/` directories and `*.pyc` files are generated Python bytecode
caches. They are not part of the algorithm and are ignored by Git.

## Data and calibration

### Calibration

- `thermal_5_9x7_30mm_384x288_20260123_102051.yml` contains the 384x288
  thermal-camera intrinsics.
- `сam_to_imu_rot_mtrx.yml` contains the camera-to-IMU rotation and temporal
  parameter `td`.

The first character in `сam_to_imu_rot_mtrx.yml` is Cyrillic. The repository
keeps the original filename to preserve existing references; runtime setup also
provides a normalized ASCII working copy.

### Raw input

- `data/01_raw_k2r/K2R00005_20260607_194949_0.mcap` is the immutable source
  recording.
- `data/01_raw_k2r/metadata.yaml` describes that recording.

### Teacher-derived input

- `data/02_derived_with_velocity/K2R00005_20260607_194949_with_velocity.mcap` is
  the provided derived recording with optical-flow and velocity results.
- `data/02_derived_with_velocity/metadata_velocity.yaml` describes the derived
  recording.

The derived MCAP is a useful checkpoint for fast fusion work, but the project
also preserves the ability to reproduce it from the raw MCAP.

## Immutable-input rule

Do not modify or overwrite `data/01_raw_k2r/` or
`data/02_derived_with_velocity/`. Store generated CSV, NPZ, JSON, PNG, and MCAP
files below `$PROJECT_CV_ARTIFACTS` in the WSL filesystem.

Historical artifact groups included:

- `preprocess_smoke/` for short preprocessing checks;
- `preprocess_v1/` for full preprocessing and a generated velocity MCAP;
- `fusion_v1/` and `fusion_v1/diagnostics/` for baseline fusion;
- `height_experiments_v1/` for isolated height-correction runs, manifests,
  diagnostics, and comparisons.

Dense GPS was used as a diagnostic reference in height experiments, not as an
online height-correction input. Early experiments still inherited the baseline
origin, calibration, and initialization behavior.

Verify immutable inputs from WSL:

```bash
source /home/fedor/project_cv_runtime/paths.env
cd $PROJECT_CV_SOURCE
sha256sum --check integrity/SHA256SUMS
```

Every line should end in `OK`.

## Local environment

Environment setup was moved out of notebooks into idempotent scripts under
`environment/`. The runtime uses WSL 2, Ubuntu 22.04, ROS 2 Humble, Python 3.10,
Jupyter, and the project Python dependencies.

After restarting Windows, Jupyter is started from the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File .\environment\windows\jupyter.ps1 start
```

Cursor connects through `Existing Jupyter Server` and uses the
`Project CV (ROS 2 Humble)` kernel.

## Historical execution order

1. Check the environment and data with notebook `00`.
2. Run notebook `01` in `smoke` mode, then use `full` only when rebuilding the
   complete velocity dataset.
3. Run the baseline fusion in notebook `02`.
4. Record baseline diagnostics in notebook `03` before changing height or graph
   behavior.
5. Use notebook `04` for explicit, isolated height experiments.

The preprocessing stage intentionally reproduced legacy mathematics first,
including documented questionable assumptions. Correcting those assumptions was
kept separate from reproducibility verification.
