# Research package

This package preserves the reproducible history of height and barometer-bias
experiments, temporal validation, and hash-locked experiment runners. It is
deliberately separated from `project_cv.final`.

Production code must not import an experiment as its default behavior. The
single explicit exception is `final/runner.py`, which reuses verified building
blocks for the opt-in `adaptive_experimental` mode. The `safe` mode remains the
default.
