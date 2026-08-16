"""Compile-only tests for the assembled final pipeline entry point."""

from __future__ import annotations

from pathlib import Path
import sys
from types import CodeType
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from project_cv.final.config import DEFAULT_MODE, FINAL_MODES  # noqa: E402
from project_cv.final.runner import prepare_final_fusion  # noqa: E402
from project_cv.experiments.runners.fusion_experiment_runner import (  # noqa: E402
    BASELINE_CELL_SHA256,
    BASELINE_NOTEBOOK,
)


class FinalFusionRunnerTests(unittest.TestCase):
    def test_safe_is_the_default(self) -> None:
        self.assertEqual(DEFAULT_MODE, "safe")

    def test_all_final_modes_compile_from_frozen_core(self) -> None:
        notebook = PROJECT_ROOT / BASELINE_NOTEBOOK
        for mode in FINAL_MODES:
            with self.subTest(mode=mode):
                prepared = prepare_final_fusion(notebook, mode)
                self.assertIsInstance(prepared.code, CodeType)
                self.assertEqual(prepared.source_cell_sha256, BASELINE_CELL_SHA256)
                self.assertEqual(prepared.mode, mode)


if __name__ == "__main__":
    unittest.main()
