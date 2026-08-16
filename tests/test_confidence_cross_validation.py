"""Selection tests for leave-one-block-out confidence validation."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from project_cv.experiments.validation.confidence_cross_validation import _choose_policy  # noqa: E402
from project_cv.experiments.height.confidence_fallback import POLICY_CANDIDATES  # noqa: E402


class PolicySelectionTests(unittest.TestCase):
    def test_heldout_score_cannot_affect_selection(self) -> None:
        training = ["a", "b"]
        all_windows = training + ["heldout"]
        scores = {}
        for window in all_windows:
            scores[window] = {"baseline": {"mse_m2": 1.0}}
            for index, policy in enumerate(POLICY_CANDIDATES):
                scores[window][policy.name] = {"mse_m2": float(index + 1)}
        selected_before = _choose_policy(training, scores)
        for index, policy in enumerate(POLICY_CANDIDATES):
            scores["heldout"][policy.name]["mse_m2"] = float(100 - index)
        selected_after = _choose_policy(training, scores)
        self.assertEqual(selected_before, selected_after)


if __name__ == "__main__":
    unittest.main()
