"""Regression tests for causal barometer-height experiment helpers."""

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

from project_cv.experiments.height.height_experiments import (  # noqa: E402
    CAUSAL_PREVIOUS_MODE,
    CORRECTED_ALIGNMENT_MODE,
    LEGACY_ALIGNMENT_MODE,
    OFFLINE_INTERPOLATION_MODE,
    CausalHeightFilterConfig,
    HeightGraphInput,
    build_corrected_graph_input,
    build_legacy_graph_input,
    causal_height_filter,
    causal_iir_by_time,
    compare_alignment_variants,
)


class CausalFilterTests(unittest.TestCase):
    def test_filter_prefix_does_not_depend_on_future_values(self) -> None:
        """Changing a suffix must not alter any earlier filter output."""

        time_sec = np.arange(8, dtype=float)
        original = np.array([0.0, 1.0, 4.0, 2.0, 3.0, 5.0, 6.0, 7.0])
        changed_future = original.copy()
        changed_future[5:] = [5000.0, -7000.0, 9000.0]
        config = CausalHeightFilterConfig(
            median_window_samples=3,
            mean_window_samples=3,
            iir_tau_sec=0.8,
        )

        original_base, original_filtered = causal_height_filter(
            time_sec,
            original,
            config=config,
        )
        changed_base, changed_filtered = causal_height_filter(
            time_sec,
            changed_future,
            config=config,
        )

        np.testing.assert_allclose(changed_base[:5], original_base[:5])
        np.testing.assert_allclose(changed_filtered[:5], original_filtered[:5])

    def test_iir_does_not_backfill_from_first_future_observation(self) -> None:
        """A leading missing prefix remains unknown instead of using future data."""

        output = causal_iir_by_time(
            [0.0, 1.0, 2.0, 3.0],
            [np.nan, np.nan, 10.0, 20.0],
            tau_sec=1.0,
        )

        self.assertTrue(np.isnan(output[0]))
        self.assertTrue(np.isnan(output[1]))
        self.assertEqual(output[2], 10.0)
        self.assertGreater(output[3], 10.0)
        self.assertLess(output[3], 20.0)


class AlignmentAnchorTests(unittest.TestCase):
    def test_corrected_mode_anchors_raw_sample_and_resets_filter(self) -> None:
        """Corrected alignment must not turn pre-start filter lag into bias."""

        barometer = pd.DataFrame(
            {
                "tr": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                "u_corr": [0.0, 0.0, 0.0, 10.0, 20.0, 20.0],
            }
        )
        config = CausalHeightFilterConfig(
            median_window_samples=3,
            mean_window_samples=3,
            iir_tau_sec=100.0,
        )
        active_start_sec = 4.0
        p0_u_m = 20.0

        legacy = build_legacy_graph_input(
            barometer,
            active_start_sec=active_start_sec,
            p0_u_m=p0_u_m,
            config=config,
        )
        corrected = build_corrected_graph_input(
            barometer,
            active_start_sec=active_start_sec,
            p0_u_m=p0_u_m,
            config=config,
        )

        self.assertEqual(legacy.mode, LEGACY_ALIGNMENT_MODE)
        self.assertEqual(corrected.mode, CORRECTED_ALIGNMENT_MODE)
        self.assertEqual(legacy.metadata["anchor_time_sec"], active_start_sec)
        self.assertEqual(corrected.metadata["anchor_time_sec"], active_start_sec)

        self.assertLess(
            legacy.metadata["anchor_filtered_u_m"],
            legacy.metadata["anchor_raw_u_m"],
        )
        self.assertGreater(legacy.metadata["alignment_offset_m"], 0.0)
        self.assertEqual(corrected.metadata["alignment_offset_m"], 0.0)

        corrected_first = corrected.frame.iloc[0]
        self.assertEqual(float(corrected_first["u_aligned_raw"]), p0_u_m)
        self.assertEqual(float(corrected_first["u_graph"]), p0_u_m)
        self.assertEqual(float(corrected_first["tr"]), active_start_sec)
        self.assertEqual(
            corrected.metadata["discarded_pre_anchor_rows"],
            4,
        )

        legacy_anchor = legacy.frame.loc[legacy.frame["tr"] == active_start_sec].iloc[0]
        self.assertGreater(float(legacy_anchor["u_aligned_raw"]), p0_u_m)
        self.assertAlmostEqual(float(legacy_anchor["u_graph"]), p0_u_m)


class CommonSupportMetricTests(unittest.TestCase):
    @staticmethod
    def _graph_input(
        *,
        mode: str,
        time_sec: list[float],
        height_m: float,
    ) -> HeightGraphInput:
        values = np.full(len(time_sec), height_m, dtype=float)
        return HeightGraphInput(
            mode=mode,  # type: ignore[arg-type]
            frame=pd.DataFrame(
                {
                    "tr": time_sec,
                    "u_raw": values,
                    "u_filter_base": values,
                    "u_smooth": values,
                    "u_graph": values,
                }
            ),
            metadata={"test_fixture": True},
        )

    def test_variants_are_scored_on_identical_common_support(self) -> None:
        dense_gps = pd.DataFrame(
            {
                "tr": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                "u": np.zeros(6, dtype=float),
            }
        )
        graph_inputs = {
            "long_early": self._graph_input(
                mode=LEGACY_ALIGNMENT_MODE,
                time_sec=[0.0, 1.0, 2.0, 3.0, 4.0],
                height_m=1.0,
            ),
            "long_late": self._graph_input(
                mode=CORRECTED_ALIGNMENT_MODE,
                time_sec=[2.0, 3.0, 4.0, 5.0],
                height_m=2.0,
            ),
        }

        comparison = compare_alignment_variants(graph_inputs, dense_gps)

        self.assertEqual(comparison.summary["common_support_start_sec"], 2.0)
        self.assertEqual(comparison.summary["common_support_end_sec"], 4.0)
        self.assertEqual(float(comparison.alignments["tr"].min()), 2.0)
        self.assertEqual(float(comparison.alignments["tr"].max()), 4.0)

        group_sizes = comparison.alignments.groupby(
            ["series", "matching_mode"],
            sort=True,
        ).size()
        self.assertEqual(group_sizes.to_dict(), {
            ("long_early", CAUSAL_PREVIOUS_MODE): 3,
            ("long_early", OFFLINE_INTERPOLATION_MODE): 3,
            ("long_late", CAUSAL_PREVIOUS_MODE): 3,
            ("long_late", OFFLINE_INTERPOLATION_MODE): 3,
        })
        self.assertTrue((comparison.metrics["n"] == 3).all())

        biases = comparison.metrics.groupby("series")["bias_u_m"].unique()
        np.testing.assert_allclose(biases["long_early"], [1.0])
        np.testing.assert_allclose(biases["long_late"], [2.0])


if __name__ == "__main__":
    unittest.main()
