"""Tests for causal robust barometer-bias estimators."""

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

from project_cv.experiments.height.barometer_bias import (  # noqa: E402
    RobustBiasConfig,
    RobustBiasSpeedConfig,
    apply_causal_robust_bias,
    apply_causal_robust_bias_speed,
)


class RobustBiasTests(unittest.TestCase):
    def test_future_starlink_does_not_change_past_barometer_samples(self) -> None:
        barometer = pd.DataFrame(
            {"tr": np.arange(0.0, 31.0), "u_graph": np.full(31, 5.0)}
        )
        starlink = pd.DataFrame({"tr_corr": [10.0, 20.0], "u_corr": [2.0, 2.0]})
        result = apply_causal_robust_bias(
            barometer,
            starlink,
            config=RobustBiasConfig(
                innovation_gate_m=6.0,
                huber_clip_m=3.0,
                time_constant_sec=10.0,
            ),
        )

        before_first_fix = result.frame.loc[result.frame["tr"] < 10.0]
        self.assertTrue(np.allclose(before_first_fix["u_graph"], 5.0))
        self.assertGreater(float(result.frame["estimated_bias_m"].iloc[-1]), 0.0)
        self.assertLess(float(result.frame["u_graph"].iloc[-1]), 5.0)

    def test_large_starlink_outlier_is_rejected(self) -> None:
        barometer = pd.DataFrame({"tr": [0.0, 10.0, 20.0], "u_graph": [5.0] * 3})
        starlink = pd.DataFrame({"tr_corr": [10.0, 20.0], "u_corr": [2.0, 100.0]})
        result = apply_causal_robust_bias(barometer, starlink)

        self.assertEqual(result.trace["accepted"].tolist(), [True, False])
        self.assertEqual(result.metadata["accepted_starlink_updates"], 1)
        self.assertEqual(result.metadata["rejected_starlink_updates"], 1)
        self.assertFalse(result.metadata["dense_gps_used"])


class RobustBiasSpeedTests(unittest.TestCase):
    def test_speed_model_is_causal_and_rejects_outlier(self) -> None:
        barometer = pd.DataFrame(
            {"tr": np.arange(0.0, 41.0), "u_graph": np.full(41, 8.0)}
        )
        starlink = pd.DataFrame(
            {"tr_corr": [5.0, 10.0, 20.0, 30.0], "u_corr": [5.0, 5.0, 5.0, 100.0]}
        )
        # The first velocity arrives after the first Starlink fix.  That fix
        # must be skipped rather than paired with a future velocity sample.
        vision = pd.DataFrame(
            {
                "tr_corr": [6.0, 10.0, 20.0, 30.0, 40.0],
                "vx": [10.0] * 5,
                "vy": [0.0] * 5,
            }
        )
        result = apply_causal_robust_bias_speed(
            barometer,
            starlink,
            vision,
            config=RobustBiasSpeedConfig(
                speed_median_window_samples=1,
                speed_mean_window_samples=1,
            ),
        )

        self.assertEqual(result.trace["tr"].tolist(), [10.0, 20.0, 30.0])
        self.assertEqual(result.trace["accepted"].tolist(), [True, True, False])
        self.assertTrue(np.allclose(result.frame.loc[result.frame["tr"] < 10.0, "u_graph"], 8.0))
        self.assertGreater(float(result.frame["estimated_total_error_m"].iloc[-1]), 0.0)
        self.assertFalse(result.metadata["dense_gps_used"])
        self.assertTrue(result.metadata["causal"])


if __name__ == "__main__":
    unittest.main()
