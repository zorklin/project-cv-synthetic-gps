"""Tests for blocked temporal-validation acceptance logic."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from project_cv.experiments.validation.temporal_validation import LONG_WINDOWS, _decision  # noqa: E402


def _row(improvement: float, *, counts_equal: bool = True) -> dict[str, object]:
    return {
        "role": "primary",
        "raw_improvement_percent": improvement,
        "raw_candidate_wins": improvement > 0.0,
        "baseline_failed_updates": 0,
        "candidate_failed_updates": 0,
        "measurement_counts_equal": counts_equal,
    }


class TemporalDecisionTests(unittest.TestCase):
    def test_primary_windows_are_non_overlapping(self) -> None:
        self.assertEqual(len(LONG_WINDOWS), 6)
        for first, second in zip(LONG_WINDOWS, LONG_WINDOWS[1:]):
            self.assertLessEqual(
                first.offset_sec + first.duration_sec,
                second.offset_sec,
            )

    def test_accepts_stable_improvement(self) -> None:
        decision = _decision([_row(value) for value in (8, 7, 6, 5, 4, -2)])
        self.assertTrue(decision["accepted"])

    def test_rejects_one_large_regression(self) -> None:
        decision = _decision([_row(value) for value in (20, 18, 15, 12, 9, -11)])
        self.assertFalse(decision["accepted"])
        self.assertFalse(decision["checks"]["worst_raw_regression"])


if __name__ == "__main__":
    unittest.main()
