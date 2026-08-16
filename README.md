# Project CV — Synthetic GPS

This project reconstructs a dense flight trajectory between sparse
Starlink/GPS measurements. It combines a downward-looking thermal camera,
optical flow, IMU, barometer data, and a GTSAM fixed-lag smoother.

## Current status

- `safe` is the supported default mode.
- `adaptive_experimental` is an opt-in, confidence-gated height correction. It
  improves the full recorded flight, but blocked validation did not demonstrate
  reliable transfer to arbitrary cold starts.
- Raw and teacher-provided MCAP files are immutable inputs.
- Large generated outputs live in the WSL runtime and are not committed to Git.

## Repository layout

```text
project_cv/
├── calibration/              # Camera intrinsics and camera-to-IMU rotation
├── data/                     # Immutable raw and teacher-derived MCAP files
├── docs/                     # Architecture, usage, results, and experiments
├── environment/              # WSL, ROS 2, Python, and Jupyter setup
├── integrity/                # SHA-256 checks for immutable inputs
├── notebooks/
│   ├── pipeline/             # 00–03: normal working sequence
│   ├── experiments/          # Research notebooks, not production
│   └── reference/            # Original teacher-provided Colab notebook
├── src/project_cv/
│   ├── final/                # Supported fusion configuration and runner
│   ├── experiments/          # Research estimators, runners, and validation
│   ├── final_cli.py          # Recommended command-line entry point
│   ├── vision_velocity.py    # Optical-flow preprocessing
│   └── preprocessing_validation.py
└── tests/                    # Unit and compile-contract tests
```

See the detailed [project structure](docs/architecture/project-structure.md).

## Quick start

After restarting Windows, start Jupyter from the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File .\environment\windows\jupyter.ps1 start
```

In a WSL terminal:

```bash
source /opt/ros/humble/setup.bash
source /home/fedor/project_cv_runtime/paths.env
cd $PROJECT_CV_SOURCE
sha256sum --check integrity/SHA256SUMS
```

The notebook order is documented in [notebooks/README.md](notebooks/README.md).

## Final pipeline

Validate the configuration without running GTSAM:

```bash
python -m project_cv.final_cli --mode safe --dry-run
```

Run the supported pipeline:

```bash
python -m project_cv.final_cli \
  --mode safe \
  --output-dir /home/fedor/project_cv_runtime/artifacts/final_pipeline_v1/safe/gtsam
```

The runner will not overwrite a non-empty output directory. Use a new path,
such as `final_pipeline_v2/safe/gtsam`, for another complete run.

See the [final pipeline guide](docs/usage/final-pipeline.md) for details.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Documentation

The [documentation index](docs/README.md) separates current usage instructions,
architecture, experimental history, and validation results.
