"""Compile-only tests for causal barometer-bias fusion experiments."""

from __future__ import annotations

from pathlib import Path
import sys
from types import CodeType
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from project_cv.experiments.runners.bias_fusion_experiment_runner import (  # noqa: E402
    BIAS_VARIANTS,
    prepare_bias_experiment,
)
from project_cv.experiments.runners.fusion_experiment_runner import (  # noqa: E402
    BASELINE_CELL_SHA256,
    BASELINE_NOTEBOOK,
)


class PrepareBiasExperimentTests(unittest.TestCase):
    def test_all_bias_variants_compile_from_frozen_cell(self) -> None:
        notebook_path = PROJECT_ROOT / BASELINE_NOTEBOOK
        for variant in BIAS_VARIANTS:
            with self.subTest(variant=variant):
                prepared = prepare_bias_experiment(notebook_path, variant)
                self.assertIsInstance(prepared.code, CodeType)
                self.assertEqual(prepared.variant, variant)
                self.assertEqual(prepared.source_cell_sha256, BASELINE_CELL_SHA256)
                self.assertIn("robustly gated sparse Starlink", " ".join(prepared.transformations))
                self.assertEqual(prepared.notebook_path, notebook_path.resolve())


if __name__ == "__main__":
    unittest.main()
