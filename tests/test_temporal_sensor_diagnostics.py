"""Small correctness tests for validation sensor matching."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from project_cv.experiments.validation.temporal_sensor_diagnostics import _causal_reference  # noqa: E402


class CausalReferenceTests(unittest.TestCase):
    def test_does_not_borrow_future_reference(self) -> None:
        reference = pd.DataFrame({"tr": [1.0, 3.0], "u": [10.0, 30.0]})
        matched = _causal_reference(reference, np.array([0.5, 1.0, 2.9, 3.0]))
        self.assertTrue(np.isnan(matched[0]))
        np.testing.assert_allclose(matched[1:], np.array([10.0, 10.0, 30.0]))


if __name__ == "__main__":
    unittest.main()
