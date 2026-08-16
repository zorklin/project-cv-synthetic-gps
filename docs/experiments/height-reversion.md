# Mean-reverting height correction and low-lag export

This follow-up addresses two failure modes observed in the first speed-aware
barometer experiment:

1. a stale bias/speed correction persisted into the late part of the flight;
2. the final trailing median-11, mean-11 and IIR-0.8 s cascade added enough lag
   to increase height RMSE.

The new estimator remains causal. It uses sparse corrected Starlink altitude
for state updates and optical-flow horizontal velocity for the quadratic speed
term. Dense GPS is used only for offline scoring and hyperparameter selection.

## State prediction

Between Starlink fixes, the estimator now applies independent exponential
mean reversion:

- slow barometer bias time constant: `60 s`;
- quadratic speed-gain time constant: `240 s`;
- bias random walk: `0.2 m / sqrt(s)`;
- robust Starlink gate/Huber clip: `10 m / 3 m`.

This prevents a correction learned during fast flight from remaining frozen
after the pressure disturbance changes.

## Full GTSAM result

| Run | Full U RMSE, m | First half, m | Second half, m |
|---|---:|---:|---:|
| `raw_start_reset` | 3.196 | 4.065 | 1.960 |
| first bias+speed model | 2.444 | 2.622 | 2.251 |
| mean reversion, raw export | **2.186** | **2.462** | **1.867** |
| mean reversion, IIR 0.2 s | 2.198 | 2.475 | 1.876 |

The late-flight regression is removed: second-half RMSE is now lower than both
the first speed-aware model and the original baseline. Relative to
`raw_start_reset`, raw/exported RMSE improves by `31.6% / 39.2%`. Relative to
the first speed-aware experiment, it improves by `10.6% / 25.8%`.

All compared runs use the same 5,650 nodes, 2,825 barometer factors, 125
Starlink factors, and 6,070 optical-flow velocity factors. There were zero
failed smoother updates. XY RMSE remains effectively unchanged (`20.419 m`),
and the maximum pointwise XY difference from the baseline is `0.319 m`.

## Output-height policy

Two byte-identical graph runs verified that output smoothing does not affect
the graph state:

- raw export: RMSE `2.186 m`, height-step standard deviation `0.1565 m`;
- causal IIR `tau=0.2 s`: RMSE `2.198 m`, height-step standard deviation
  `0.1473 m`.

The low-lag IIR costs only `0.011 m` RMSE while reducing step variability by
about `5.9%`, so it is the recommended operational export. The raw export is
retained as the accuracy reference.

## Artifacts

```text
/home/fedor/project_cv_runtime/artifacts/height_experiments_v1/baro_bias_v2/
├── reversion_export_raw/gtsam
└── reversion_light_smoothing/gtsam
```

The result still needs validation on another flight. Although dense GPS never
enters the online estimator, the `60/240 s` hyperparameters were selected from
this flight's offline GPS diagnostics and may not generalize unchanged.
