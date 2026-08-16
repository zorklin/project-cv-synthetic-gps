# Barometer bias experiment

This experiment keeps `notebooks/pipeline/02_sparse_gps_fusion.ipynb` frozen and
constructs two causal barometer inputs in memory:

- `raw_start_reset_robust_bias`: one slowly varying barometer-height bias,
  updated only by robustly accepted sparse Starlink altitude;
- `raw_start_reset_robust_bias_speed`: the same bias plus a quadratic
  horizontal-speed term, using causally smoothed optical-flow velocity as a
  proxy for dynamic-pressure error.

Dense GPS is not used by either estimator. It remains an offline diagnostic
reference and retains its existing baseline roles in origin/calibration and
initialization.

## Static-pressure diagnostic

The reproducible diagnostic reads `/mavros/imu/static_pressure` directly from
the derived MCAP and compares its residual with horizontal speed. On the
current dataset:

- 11,918 pressure samples were extracted;
- pressure residual vs GPS horizontal speed correlation: `-0.672`;
- barometer height error vs GPS horizontal speed correlation: `+0.658`;
- slow-half height bias/RMSE: `+0.306 m / 2.174 m`;
- fast-half height bias/RMSE: `+3.263 m / 3.936 m`.

This supports a speed-dependent static-pressure disturbance rather than a
single constant altitude scale error.

## Full GTSAM results

All runs use the same 5,650 graph nodes, 2,825 barometer factors, 125 Starlink
factors, and 6,070 vision-velocity factors. Every run completed with zero
failed smoother updates.

| Run | Raw U RMSE, m | Exported U RMSE, m | XY RMSE, m |
|---|---:|---:|---:|
| `raw_start_reset` | 3.196 | 3.595 | 20.418 |
| `raw_start_reset_robust_bias` | 2.880 | 3.317 | 20.419 |
| `raw_start_reset_robust_bias_speed` | **2.444** | **2.946** | 20.418 |

Relative to `raw_start_reset`, the speed-aware estimator improves raw U RMSE
by `23.5%` and exported U RMSE by `18.0%`. The XY RMSE is effectively
unchanged. Its maximum pointwise XY difference from the baseline run is
`0.572 m`.

The result is not uniformly better. Splitting the flight at `t=711.818 s`,
raw U RMSE changes from `4.065 -> 2.622 m` in the first half but from
`1.960 -> 2.251 m` in the second half. The model is therefore a successful
experiment on overall RMSE, but it should not replace the baseline until it is
validated on another flight and the late-flight over-correction is addressed.

## Reproduction

Run from the configured WSL environment:

```bash
python -m project_cv.experiments.validation.static_pressure_diagnostics \
  --barometer-graph-input /home/fedor/project_cv_runtime/artifacts/height_experiments_v1/baro_alignment_v1/raw_start_reset/gtsam/baro_graph_input.csv \
  --output-dir /home/fedor/project_cv_runtime/artifacts/height_experiments_v1/baro_bias_v1/static_pressure_diagnostic

python -m project_cv.experiments.runners.bias_fusion_experiment_runner \
  --variant raw_start_reset_robust_bias \
  --output-dir /home/fedor/project_cv_runtime/artifacts/height_experiments_v1/baro_bias_v1/raw_start_reset_robust_bias/gtsam

python -m project_cv.experiments.runners.bias_fusion_experiment_runner \
  --variant raw_start_reset_robust_bias_speed \
  --output-dir /home/fedor/project_cv_runtime/artifacts/height_experiments_v1/baro_bias_v1/raw_start_reset_robust_bias_speed/gtsam
```

The runners refuse to overwrite non-empty output directories. Use a new
directory name for a repeated experiment.
