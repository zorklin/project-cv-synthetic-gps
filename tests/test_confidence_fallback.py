"""Causality and fallback tests for the confidence safety layer."""

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

from project_cv.experiments.height.confidence_fallback import (  # noqa: E402
    ConfidencePolicyConfig,
    apply_confidence_to_correction,
    causal_confidence_series,
)


class ConfidenceFallbackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ConfidencePolicyConfig(
            name="test",
            min_consistent_updates=2,
            consistency_window_updates=3,
            max_abs_innovation_m=3.0,
            max_bias_mad_m=1.0,
            max_abs_correction_m=2.0,
            confidence_rise_tau_sec=1.0,
            confidence_fall_tau_sec=1.0,
        )

    def test_future_update_does_not_change_confidence_prefix(self) -> None:
        times = np.arange(0.0, 8.0)
        prefix_trace = pd.DataFrame(
            {
                "tr": [1.0, 3.0],
                "accepted": [True, True],
                "innovation_m": [0.5, 0.5],
                "measurement_bias_m": [1.0, 1.2],
            }
        )
        full_trace = pd.concat(
            [
                prefix_trace,
                pd.DataFrame(
                    {
                        "tr": [20.0],
                        "accepted": [True],
                        "innovation_m": [0.0],
                        "measurement_bias_m": [100.0],
                    }
                ),
            ],
            ignore_index=True,
        )
        np.testing.assert_allclose(
            causal_confidence_series(times, prefix_trace, config=self.config),
            causal_confidence_series(times, full_trace, config=self.config),
        )

    def test_rejected_update_forces_fallback_and_correction_is_capped(self) -> None:
        times = np.arange(0.0, 8.0)
        trace = pd.DataFrame(
            {
                "tr": [1.0, 2.0, 5.0],
                "accepted": [True, True, False],
                "innovation_m": [0.1, 0.2, 10.0],
                "measurement_bias_m": [1.0, 1.1, 20.0],
            }
        )
        applied, metadata = apply_confidence_to_correction(
            times,
            np.full(len(times), 20.0),
            trace,
            config=self.config,
        )
        self.assertTrue(np.all(np.abs(applied) <= 2.0 + 1e-12))
        self.assertGreater(applied[4], applied[6])
        self.assertLess(float(metadata["max_abs_applied_correction_m"]), 2.01)

    def test_disabled_policy_is_exact_baseline(self) -> None:
        disabled = ConfidencePolicyConfig(name="off", enabled=False)
        applied, _ = apply_confidence_to_correction(
            np.array([0.0, 1.0]),
            np.array([2.0, -3.0]),
            pd.DataFrame(),
            config=disabled,
        )
        np.testing.assert_array_equal(applied, np.zeros(2))


if __name__ == "__main__":
    unittest.main()
