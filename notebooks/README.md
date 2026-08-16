# Notebooks

## `pipeline/` — normal sequence

1. `00_environment_and_data_check.ipynb` checks WSL, ROS 2, Python, and data
   availability.
2. `01_build_vision_velocity.ipynb` converts thermal frames and IMU data into
   optical-flow velocity. Run `smoke` first and `full` only when needed.
3. `02_sparse_gps_fusion.ipynb` contains the frozen GTSAM core and baseline
   pipeline. Use `python -m project_cv.final_cli` instead of a manual `Run All`
   when producing a final result.
4. `03_fusion_diagnostics.ipynb` performs read-only analysis of completed
   fusion artifacts.

## `experiments/`

- `04_height_experiments.ipynb` records research on the vertical channel. It is
  not part of the supported pipeline.

## `reference/`

- `teacher_colab_reference.ipynb` preserves the original teacher-provided Colab
  draft. Do not execute it with `Run All` in the local pipeline.

All working notebooks use the `Project CV (ROS 2 Humble)` kernel.
