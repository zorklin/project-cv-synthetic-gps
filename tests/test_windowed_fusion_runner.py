"""Compile and validation-contract tests for temporal window replays."""

from __future__ import annotations

from pathlib import Path
import sys
from types import CodeType
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from project_cv.experiments.runners.fusion_experiment_runner import (  # noqa: E402
    BASELINE_CELL_SHA256,
    BASELINE_NOTEBOOK,
)
from project_cv.experiments.runners.windowed_fusion_runner import (  # noqa: E402
    WINDOW_MODELS,
    prepare_window_experiment,
)


class PrepareWindowExperimentTests(unittest.TestCase):
    def test_both_models_compile_from_frozen_cell(self) -> None:
        notebook_path = PROJECT_ROOT / BASELINE_NOTEBOOK
        for model in WINDOW_MODELS:
            with self.subTest(model=model):
                prepared = prepare_window_experiment(
                    notebook_path,
                    model,
                    offset_sec=30.0,
                    duration_sec=40.0,
                )
                self.assertIsInstance(prepared.code, CodeType)
                self.assertEqual(prepared.model, model)
                self.assertEqual(prepared.offset_sec, 30.0)
                self.assertEqual(prepared.duration_sec, 40.0)
                self.assertEqual(prepared.source_cell_sha256, BASELINE_CELL_SHA256)
                self.assertEqual(prepared.notebook_path, notebook_path.resolve())

    def test_rejects_invalid_windows(self) -> None:
        notebook_path = PROJECT_ROOT / BASELINE_NOTEBOOK
        bad_windows = ((-1.0, 40.0), (0.0, 0.0), (0.0, float("nan")))
        for offset_sec, duration_sec in bad_windows:
            with self.subTest(offset_sec=offset_sec, duration_sec=duration_sec):
                with self.assertRaises(ValueError):
                    prepare_window_experiment(
                        notebook_path,
                        "raw_start_reset",
                        offset_sec=offset_sec,
                        duration_sec=duration_sec,
                    )


if __name__ == "__main__":
    unittest.main()
