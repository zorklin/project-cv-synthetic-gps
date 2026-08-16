"""Tests for static-pressure diagnostic calculations."""

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

from project_cv.experiments.validation.static_pressure_diagnostics import (  # noqa: E402
    analyze_static_pressure_speed,
)


class StaticPressureDiagnosticsTests(unittest.TestCase):
    def test_relationship_is_reported_as_diagnostic_only(self) -> None:
        time_sec = np.arange(0.0, 20.0, 0.2)
        speed = np.linspace(0.0, 10.0, len(time_sec))
        true_height = 0.5 * time_sec
        height_error = 0.04 * speed**2
        gps = pd.DataFrame(
            {
                "tr": time_sec,
                "e": np.cumsum(speed) * 0.2,
                "n": np.zeros(len(time_sec)),
                "u": true_height,
            }
        )
        barometer = pd.DataFrame(
            {"tr": time_sec, "u_graph": true_height + height_error}
        )
        pressure = pd.DataFrame(
            {
                "tr": time_sec,
                "pressure_pa": 100000.0 - 12.0 * true_height - 0.5 * speed**2,
            }
        )
        vision = pd.DataFrame(
            {"tr_corr": time_sec, "vx": speed, "vy": np.zeros(len(time_sec))}
        )

        summary, aligned = analyze_static_pressure_speed(
            pressure, barometer, gps, vision
        )

        self.assertTrue(summary["contract"]["diagnostic_only"])
        self.assertFalse(summary["contract"]["dense_gps_fused"])
        self.assertGreater(
            summary["correlations"][
                "barometer_height_error_vs_vision_horizontal_speed"
            ],
            0.9,
        )
        self.assertGreater(summary["speed_split"]["fast_bias_m"], summary["speed_split"]["slow_bias_m"])
        self.assertGreater(len(aligned), 10)


if __name__ == "__main__":
    unittest.main()
