"""Reusable code for the Project CV experiments."""

from .preprocessing_validation import validate_preprocessing_bags

from .vision_velocity import (
    LEGACY_BASELINE_NOTES,
    PreprocessingArtifacts,
    VisionVelocityConfig,
    build_derived_bag,
    compute_gyrocompensated_velocity,
    compute_sparse_lk,
    filter_sparse_lk,
    run_preprocessing,
)

__all__ = [
    "LEGACY_BASELINE_NOTES",
    "PreprocessingArtifacts",
    "VisionVelocityConfig",
    "build_derived_bag",
    "compute_gyrocompensated_velocity",
    "compute_sparse_lk",
    "filter_sparse_lk",
    "run_preprocessing",
    "validate_preprocessing_bags",
]
