# Documentation

This directory separates current instructions from research history. Start with
the usage and architecture documents. Read experiment and result reports when
you need to understand why a design was accepted or rejected.

## Current documentation

- [Project structure](architecture/project-structure.md) describes production,
  preprocessing, experiments, reference material, and generated artifacts.
- [Final pipeline](usage/final-pipeline.md) is the single supported execution
  guide.

## Results

The files under [results/](results/README.md) are evidence records, not separate
ways to run the project:

- [Final full-flight runs](results/final-run-results.md) records the supported
  baseline and the opt-in full-flight experiment.
- [Temporal validation](results/temporal-validation.md) records why the first
  adaptive correction was rejected.
- [Confidence cross-validation](results/confidence-cross-validation.md) records
  why the fallback policy also remained experimental.

Negative results are retained intentionally. Removing them would hide the
reason that `safe` remains the default.

## Experiment history

The [experiment index](experiments/README.md) links the chronological research
notes:

- [Barometer bias](experiments/height-bias.md);
- [Mean reversion and output smoothing](experiments/height-reversion.md).

## Archive

- [Historical migration notes](archive/original-readme.md) preserve context from
  the earlier notebook-centered project. They are not current instructions.
