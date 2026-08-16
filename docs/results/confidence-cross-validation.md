# Confidence/fallback cross-validation

**Status:** failed the predefined acceptance gate. The selected policy is kept
only in `adaptive_experimental`.

Five predefined causal policies were evaluated. For each of six 180-second
flight blocks, the policy was selected using the other five blocks and tested
on the held-out block. Dense GPS was used only to score training folds and the
held-out result; it was never an online estimator input.

| Held-out offset | Selected policy | Baseline RMSE | Policy RMSE | Change |
|---:|---|---:|---:|---:|
| 0 s | balanced | 2.666 m | 2.747 m | -3.0% |
| 180 s | conservative | 2.240 m | 2.029 m | +9.4% |
| 360 s | balanced | 2.553 m | 1.851 m | +27.5% |
| 540 s | balanced | 2.022 m | 1.868 m | +7.6% |
| 720 s | balanced | 1.643 m | 1.437 m | +12.5% |
| 900 s | balanced | 1.859 m | 2.086 m | -12.2% |

Pooled barometer RMSE changes from `2.194 m` to `2.041 m`, an improvement of
`6.98%`. Four of six blocks improve, the worst regression is `12.21%`, and the
exact paired permutation result is `p = 0.3125`.

The predefined acceptance gate required all of the following:

- pooled improvement of at least `10%`;
- improvement in at least five of six blocks;
- no regression worse than `10%`.

The candidate failed all three thresholds. Confidence fallback reduces the risk
of an ungated correction but does not establish a sufficiently large or robust
advantage. The `balanced` policy is therefore preserved only as
`adaptive_experimental`, while `safe` remains the supported default.

## Machine-readable result

`/home/fedor/project_cv_runtime/artifacts/height_experiments_v1/temporal_validation_v1/confidence_lobo_summary.json`
