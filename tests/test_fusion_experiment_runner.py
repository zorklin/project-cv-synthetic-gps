"""Compile-only tests for the frozen-notebook fusion experiment harness."""

from __future__ import annotations

from pathlib import Path
import sys
from types import CodeType
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from project_cv.fusion_experiment_runner import (  # noqa: E402
    BASELINE_CELL_SHA256,
    BASELINE_NOTEBOOK,
    prepare_experiment,
)


class PrepareExperimentTests(unittest.TestCase):
    def test_all_variants_compile_without_executing_gtsam(self) -> None:
        notebook_path = PROJECT_ROOT / BASELINE_NOTEBOOK
        expected_transformations = {
            "legacy_replay": (
                "persist exact baro graph input to baro_graph_input.csv",
            ),
            "no_p0_realign": (
                "BARO_ALIGN_TO_P0: True -> False",
                "persist exact baro graph input to baro_graph_input.csv",
            ),
            "raw_start_reset": (
                "align raw causal u_corr to p0, then restart median/mean/IIR at active start",
                "persist exact baro graph input to baro_graph_input.csv",
            ),
        }

        for variant, transformations in expected_transformations.items():
            with self.subTest(variant=variant):
                prepared = prepare_experiment(notebook_path, variant)  # type: ignore[arg-type]

                self.assertIsInstance(prepared.code, CodeType)
                self.assertEqual(prepared.variant, variant)
                self.assertEqual(prepared.source_cell_sha256, BASELINE_CELL_SHA256)
                self.assertEqual(prepared.transformations, transformations)
                self.assertEqual(prepared.notebook_path, notebook_path.resolve())


if __name__ == "__main__":
    unittest.main()
