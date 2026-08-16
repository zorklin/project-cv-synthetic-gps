# Final pipeline full-flight verification

**Status:** current full-flight reference. The `safe` mode remains the supported
default; `adaptive_experimental` remains opt-in.

Both modes were executed on the same complete continuous flight through the same
Python entry point. They used identical graph-node and barometer, Starlink, and
vision-factor counts. Both runs reported `failed_updates = 0`.

| Mode | Raw graph U RMSE | Exported U RMSE | Status |
|---|---:|---:|---|
| `safe` | 3.196 m | 3.202 m | supported default |
| `adaptive_experimental` | 2.299 m | 2.308 m | opt-in experiment |

On this particular complete flight, the confidence-gated mode reduces raw U
RMSE by approximately `28.1%`. Its mean confidence is `0.698`, the correction is
active for approximately `78.7%` of barometer samples, and the correction is
capped at `3 m`.

This does not override the blocked cross-validation result. Across independent
cold starts, the pooled gain was only `6.98%`, four of six blocks improved, and
the worst regression was `12.21%`. Therefore, the table above is an in-sample
full-flight result, not evidence of generalization to a new flight.

## Artifacts

- `/home/fedor/project_cv_runtime/artifacts/final_pipeline_v1/safe/gtsam/`
- `/home/fedor/project_cv_runtime/artifacts/final_pipeline_v1/adaptive_experimental/gtsam/`

Each directory contains `final_pipeline_manifest.json`, exact input hashes,
configuration, fusion summary, synthetic GPS, and graph-node CSV files.
