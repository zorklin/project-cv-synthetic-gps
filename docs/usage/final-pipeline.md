# Final pipeline v1

## Decision

The default mode is `safe`. It does not use adaptive Starlink-based barometer
bias correction because that correction did not demonstrate sufficient
stability in blocked validation.

The research result remains available as an explicit opt-in mode rather than
being deleted.

## Run

After starting the WSL/Jupyter environment, or in a regular Ubuntu terminal:

```bash
source /opt/ros/humble/setup.bash
source /home/fedor/project_cv_runtime/paths.env
cd $PROJECT_CV_SOURCE

python -m project_cv.final_cli \
  --mode safe \
  --output-dir /home/fedor/project_cv_runtime/artifacts/final_pipeline_v1/safe/gtsam
```

Check the configuration without running GTSAM:

```bash
python -m project_cv.final_cli --mode safe --dry-run
```

Run the experimental mode only when explicitly needed:

```bash
python -m project_cv.final_cli \
  --mode adaptive_experimental \
  --output-dir /home/fedor/project_cv_runtime/artifacts/final_pipeline_v1/adaptive_experimental/gtsam
```

The runner refuses to write into a non-empty output directory. Every run stores:

- synthetic GPS CSV;
- graph-node CSV;
- the exact barometer input;
- GTSAM summary;
- final configuration;
- SHA-256 hashes of all inputs and the frozen fusion core;
- standard-output log;
- complete run manifest.

## Mode status

### `safe`

- supported default;
- raw-at-start barometer alignment;
- causal filter reset;
- output IIR `tau = 0.2 s`;
- no hidden adaptive corrections.

### `adaptive_experimental`

- robust mean-reverting bias plus a speed-squared proposal;
- three consecutive consistent sparse updates before confidence can rise;
- causal innovation, MAD, and update-age checks;
- correction capped at `3 m`;
- smooth fallback to the safe baseline;
- blocked cross-validation: `6.98%` pooled barometer improvement, four wins out
  of six, and a worst regression of `12.21%`;
- not accepted as the default.

## What counts as the final result

Use the `safe` output for demonstrations, integration, and the main report.

The full-flight adaptive result may be described as an in-sample experiment,
not as demonstrated accuracy on unseen flights.
