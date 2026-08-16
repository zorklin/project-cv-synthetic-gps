# Temporal validation of the height correction

**Status:** failed the predefined acceptance gate. This report evaluates the
first mean-reverting candidate before confidence/fallback was added.

## Decision

The mean-reverting barometer bias + horizontal-speed correction is **not yet
accepted as the final algorithm**.  It improves most of this flight, but it is
not stable enough across independent cold starts.

The full-flight result (`2.186 m` raw U RMSE) remains a valid in-sample result.
It must not be presented as evidence that the same gain will transfer to a new
flight.

## Frozen protocol

- Baseline: raw barometer alignment/filter reset at each window start.
- Candidate: the same baseline plus the already selected robust, mean-reverting
  bias + speed-squared correction.
- Candidate parameters were not retuned during validation: bias decay `60 s`,
  speed-gain decay `240 s`, bias random walk `0.2 m/sqrt(s)`, innovation gate
  `10 m`, Huber clip `3 m`.
- Both models use the same final causal output IIR (`tau = 0.2 s`) and no rolling
  median/mean windows.
- Six non-overlapping `180 s` primary windows were replayed with all causal state
  restarted.  Four `40 s` windows tested startup behavior.
- The existing baseline initializes position/velocity from a causal dense-GPS
  sample at the new start.  Dense GPS is then used only for diagnostics and is
  never inserted into the GTSAM graph as a measurement factor.
- Acceptance was fixed before reading the results: win at least 5/6 primary
  windows, median improvement at least 5%, no primary regression worse than
  10%, zero failed GTSAM updates, and identical measurement counts.

## Primary 180-second windows

| Offset | Baseline raw U RMSE | Candidate raw U RMSE | Change |
|---:|---:|---:|---:|
| 0 s | 2.721 m | 3.007 m | -10.5% |
| 180 s | 2.241 m | 1.768 m | +21.1% |
| 360 s | 2.574 m | 2.365 m | +8.1% |
| 540 s | 1.983 m | 1.752 m | +11.6% |
| 720 s | 1.627 m | 1.482 m | +8.9% |
| 900 s | 1.887 m | 2.334 m | -23.7% |

The candidate wins 4/6 windows and has a median improvement of `8.49%`, but its
worst regression is `23.74%`.  Pooled RMSE over the six equal-duration windows
changes only from `2.205 m` to `2.179 m` (`1.22%`).  All optimizer runs have zero
failed updates and paired measurement counts are identical.

## Startup stress tests

The user-proposed replay from offset `30 s` to `70 s` changes raw U RMSE from
`4.452 m` to `4.647 m` (`-4.4%`).  Other 40-second windows are mixed: `+20.4%`,
`-62.9%`, and `+16.1%`.  With only four or five sparse Starlink fixes in such a
window, early state updates can dominate the whole score.

## Diagnosis

The regressions are not caused by GTSAM failures or different factor counts.
They originate before the graph, in the correction applied to barometer height.

- In the first long window, sparse Starlink height itself has about `21.87 m`
  RMSE against the diagnostic dense reference.  The hard gate rejects several
  extreme fixes, but a near-threshold update still moves the correction in the
  wrong direction.
- In the last long window, Starlink height RMSE rises to about `3.92 m`; baseline
  barometer RMSE is already about `1.86 m`, so following Starlink worsens it.
- In the 600-second short-start case, the candidate applies about `+1.06 m` mean
  correction while the newly aligned baseline is already biased low.  This is a
  cold-start/state-identifiability problem, especially for the non-negative
  speed-squared term.

Therefore this design was retained as an experiment rather than promoted to the
supported default. The subsequent confidence/fallback policy is evaluated in
[confidence-cross-validation.md](confidence-cross-validation.md); it reduced
risk but also failed its predefined acceptance gate. A genuinely independent
flight is still required for a generalization claim.

## Reproducible outputs

The machine-readable summaries and every isolated replay are under:

`/home/fedor/project_cv_runtime/artifacts/height_experiments_v1/temporal_validation_v1/`

Key files are `temporal_validation_summary.csv`,
`temporal_validation_summary.json`, `temporal_sensor_diagnostics.csv`, and
`temporal_sensor_diagnostics.json`.
