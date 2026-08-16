"""Reviewed configuration and status of final-pipeline modes."""

from __future__ import annotations

from typing import Final, Literal


FinalMode = Literal["safe", "adaptive_experimental"]
FINAL_MODES: Final[tuple[FinalMode, ...]] = ("safe", "adaptive_experimental")
DEFAULT_MODE: Final[FinalMode] = "safe"

SAFE_HEIGHT_POLICY: Final = {
    "status": "default",
    "barometer_alignment": "raw causal sample aligned to p0 at startup",
    "barometer_filter_state": "reset at startup",
    "adaptive_barometer_bias": False,
    "output_filter": "causal IIR",
    "output_filter_tau_sec": 0.2,
    "reason": (
        "The adaptive candidate did not pass the predeclared blocked-validation "
        "stability gate; safe mode therefore retains the deterministic baseline."
    ),
}

ADAPTIVE_HEIGHT_POLICY: Final = {
    "status": "experimental_opt_in",
    "barometer_alignment": "raw causal sample aligned to p0 at startup",
    "barometer_filter_state": "reset at startup",
    "adaptive_barometer_bias": True,
    "bias_model": "robust mean-reverting bias plus horizontal-speed squared",
    "confidence_policy": "balanced",
    "output_filter": "causal IIR",
    "output_filter_tau_sec": 0.2,
    "validation": {
        "method": "leave-one-temporal-block-out on one flight",
        "pooled_baro_rmse_improvement_percent": 6.9787906757888125,
        "heldout_wins": "4/6",
        "worst_heldout_regression_percent": -12.209034472295563,
        "accepted_as_large_robust_boost": False,
    },
}
