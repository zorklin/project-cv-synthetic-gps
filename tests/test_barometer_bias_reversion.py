"""Tests for mean-reverting causal barometer correction."""

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

from project_cv.experiments.height.barometer_bias_reversion import (  # noqa: E402
    RevertingBiasSpeedConfig,
    apply_causal_reverting_bias_speed,
)


class RevertingBiasSpeedTests(unittest.TestCase):
    def setUp(self) -> None:
        self.barometer = pd.DataFrame(
            {"tr": np.arange(0.0, 101.0), "u_graph": np.full(101, 8.0)}
        )
        self.vision = pd.DataFrame(
            {
                "tr_corr": np.arange(0.0, 101.0),
                "vx": np.zeros(101),
                "vy": np.zeros(101),
            }
        )
        self.config = RevertingBiasSpeedConfig(
            speed_median_window_samples=1,
            speed_mean_window_samples=1,
            bias_mean_reversion_tau_sec=20.0,
            speed_gain_mean_reversion_tau_sec=40.0,
        )

    def test_bias_decays_without_later_sparse_updates(self) -> None:
        result = apply_causal_reverting_bias_speed(
            self.barometer,
            pd.DataFrame({"tr_corr": [10.0], "u_corr": [5.0]}),
            self.vision,
            config=self.config,
        )

        at_update = float(
            result.frame.loc[result.frame["tr"] == 10.0, "estimated_total_error_m"].iloc[0]
        )
        at_end = float(result.frame["estimated_total_error_m"].iloc[-1])
        self.assertGreater(at_update, 0.0)
        self.assertLess(at_end, at_update * 0.02)
        self.assertTrue(result.metadata["causal"])
        self.assertFalse(result.metadata["dense_gps_used"])

    def test_future_fix_does_not_change_prefix_and_outlier_is_rejected(self) -> None:
        first_only = pd.DataFrame({"tr_corr": [10.0], "u_corr": [5.0]})
        with_future = pd.DataFrame(
            {"tr_corr": [10.0, 20.0], "u_corr": [5.0, 100.0]}
        )
        first_result = apply_causal_reverting_bias_speed(
            self.barometer,
            first_only,
            self.vision,
            config=self.config,
        )
        future_result = apply_causal_reverting_bias_speed(
            self.barometer,
            with_future,
            self.vision,
            config=self.config,
        )

        prefix = future_result.frame["tr"] < 20.0
        self.assertTrue(
            np.allclose(
                first_result.frame.loc[prefix, "u_graph"],
                future_result.frame.loc[prefix, "u_graph"],
            )
        )
        self.assertEqual(future_result.trace["accepted"].tolist(), [True, False])


if __name__ == "__main__":
    unittest.main()
