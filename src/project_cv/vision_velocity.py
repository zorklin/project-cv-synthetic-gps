"""Legacy-compatible raw MCAP -> optical-flow velocity preprocessing.

This module is a structured port of the preprocessing cells from
``notebooks/reference/teacher_colab_reference.ipynb``.  It intentionally
preserves the baseline mathematics and constants so that a generated bag can
first be compared with the teacher-provided reference.  Suspect assumptions
are documented in :data:`LEGACY_BASELINE_NOTES`; they are not silently fixed.

The public stages are:

1. :func:`compute_sparse_lk` -- thermal images -> raw sparse LK CSV;
2. :func:`filter_sparse_lk` -- raw LK -> legacy physical/MAD filtered flow;
3. :func:`compute_gyrocompensated_velocity` -- filtered pixel displacement +
   IMU/range/Euler -> raw and filtered FRD velocity CSV files;
4. :func:`build_derived_bag` -- copy the source bag and add the filtered flow
   and ``/vision/velocity_frd`` topics;
5. :func:`run_preprocessing` -- run all stages.

All generated files are written below the caller-supplied ``artifact_dir``.
Source bags and calibration files are read-only inputs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import shutil
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


LEGACY_BASELINE_NOTES: tuple[str, ...] = (
    "The physical flow filter assumes a 384x288 image, 22 degree horizontal "
    "FOV and a fixed 8 m height; the later metric velocity stage instead uses "
    "the camera calibration and live rangefinder.",
    "Optical-flow lag is fixed at +0.037 s even though the supplied thermal "
    "calibration contains a different td value.",
    "The final velocity outlier filter uses a centered 7-sample rolling median "
    "and is therefore offline/non-causal.",
    "The causal pixel-flow rolling statistics use bfill at their leading edge, "
    "so the very beginning is not strictly causal.",
    "Thermal frames are CLAHE/blurred but are not undistorted before LK.",
    "The camera-to-IMU matrix is interpreted directly as camera-to-FRD, exactly "
    "as in the legacy notebook.",
)


FLOW_MESSAGE_COLUMNS: tuple[str, ...] = (
    "frame_idx",
    "stamp_ns",
    "t_sec",
    "dt_s",
    "flow_x_px_s_filt",
    "flow_y_px_s_filt",
    "dx_px_filt_equiv",
    "dy_px_filt_equiv",
    "flow_mag_filt",
    "quality_good_tracks_ratio",
    "n_good_tracks",
    "valid_filt",
    "is_bad_measurement",
    "outlier_score",
    "physical_bad",
    "mad_bad",
    "dt_bad",
    "quality_bad",
)


@dataclass
class VisionVelocityConfig:
    """Configuration for the legacy-compatible preprocessing pipeline.

    ``start_offset_sec`` and ``duration_sec`` select a relative window on the
    image stream.  ``duration_sec=None`` (the default) processes the full bag.
    A short duration is intended for smoke tests, not for final comparison.
    """

    input_bag_dir: Path
    thermal_calibration_file: Path
    camera_to_imu_file: Path
    artifact_dir: Path

    image_topic: str = "/camera/image_raw/compressed"
    flow_topic: str = "/vision/lk_flow_px_filtered"
    velocity_topic: str = "/vision/velocity_frd"
    imu_topic: str = "/imu/data_raw"
    range_topic: str = "/mavros/rangefinder/rangefinder"
    euler_topic: str = "/imu/euler"
    velocity_frame_id: str = "base_link_frd"

    start_offset_sec: float = 0.0
    duration_sec: float | None = None
    use_header_stamp: bool = False

    # Sparse LK values used by the actual legacy invocation cell.
    max_corners: int = 1500
    quality_level: float = 0.005
    min_distance: int = 5
    block_size: int = 7
    win_size: int = 21
    max_level: int = 3
    fb_check_px: float = 2.5
    max_motion_px: float = 120.0
    min_good_tracks: int = 10
    lk_lpf_cutoff_hz: float = 2.0

    # Legacy physical flow filter.
    image_width: int = 384
    image_height: int = 288
    nominal_fps: float = 60.0
    physical_height_m: float = 8.0
    physical_fov_x_deg: float = 22.0
    physical_v_max_mps: float = 10.0
    physical_lpf_cutoff_hz: float = 1.0
    physical_outlier_z_threshold: float = 6.0
    max_hold_gap_sec: float = 0.5

    # Gyro compensation and metric velocity filtering.
    flow_lag_sec: float = 0.037
    flow_sign: float = 1.0
    rotation_file_is_camera_to_imu: bool = True
    height_mode: str = "range_ray"
    min_range_m: float = 2.0
    max_range_m: float = 250.0
    max_abs_raw_velocity_mps: float = 80.0
    min_filter_height_m: float = 10.0
    max_filter_height_m: float = 120.0
    min_filter_dt_sec: float = 0.008
    max_filter_dt_sec: float = 0.035
    max_abs_filtered_velocity_mps: float = 45.0
    velocity_median_window: int = 7
    velocity_lpf_tau_sec: float = 0.30

    @classmethod
    def from_environment(
        cls,
        *,
        artifact_name: str = "preprocess_v1",
        start_offset_sec: float = 0.0,
        duration_sec: float | None = None,
        **overrides: Any,
    ) -> "VisionVelocityConfig":
        """Build a config from the ``PROJECT_CV_*`` WSL environment."""

        required = (
            "PROJECT_CV_RAW_BAG",
            "PROJECT_CV_CALIBRATION",
            "PROJECT_CV_ARTIFACTS",
        )
        missing = [name for name in required if not os.environ.get(name)]
        if missing:
            raise RuntimeError(
                "Missing environment variables: " + ", ".join(missing) +
                ". Start Jupyter through the project launcher."
            )

        calibration_dir = Path(os.environ["PROJECT_CV_CALIBRATION"])
        values: dict[str, Any] = {
            "input_bag_dir": Path(os.environ["PROJECT_CV_RAW_BAG"]),
            "thermal_calibration_file": calibration_dir / "thermal_camera.yml",
            "camera_to_imu_file": calibration_dir / "cam_to_imu_rot_mtrx.yml",
            "artifact_dir": Path(os.environ["PROJECT_CV_ARTIFACTS"]) / artifact_name,
            "start_offset_sec": start_offset_sec,
            "duration_sec": duration_sec,
        }
        values.update(overrides)
        return cls(**values)

    def __post_init__(self) -> None:
        self.input_bag_dir = Path(self.input_bag_dir)
        self.thermal_calibration_file = Path(self.thermal_calibration_file)
        self.camera_to_imu_file = Path(self.camera_to_imu_file)
        self.artifact_dir = Path(self.artifact_dir)

        if self.start_offset_sec < 0:
            raise ValueError("start_offset_sec must be >= 0")
        if self.duration_sec is not None and self.duration_sec <= 0:
            raise ValueError("duration_sec must be > 0 or None")
        if self.velocity_median_window < 1:
            raise ValueError("velocity_median_window must be >= 1")
        if self.height_mode not in {"range_ray", "vertical_agl", "vertical_to_ray"}:
            raise ValueError(f"Unknown height_mode: {self.height_mode}")

    def serializable(self) -> dict[str, Any]:
        values = asdict(self)
        for key in (
            "input_bag_dir",
            "thermal_calibration_file",
            "camera_to_imu_file",
            "artifact_dir",
        ):
            values[key] = str(values[key])
        return values


@dataclass(frozen=True)
class PreprocessingArtifacts:
    """Canonical generated paths below one artifact directory."""

    root: Path
    lk_raw_csv: Path
    lk_raw_npz: Path
    lk_summary_json: Path
    flow_filtered_csv: Path
    flow_filtered_npz: Path
    flow_summary_json: Path
    velocity_raw_csv: Path
    velocity_filtered_csv: Path
    velocity_summary_json: Path
    derived_bag_dir: Path
    bag_summary_json: Path
    pipeline_summary_json: Path

    @classmethod
    def under(cls, root: Path) -> "PreprocessingArtifacts":
        root = Path(root)
        return cls(
            root=root,
            lk_raw_csv=root / "lk_flow_lpf.csv",
            lk_raw_npz=root / "lk_flow_lpf.npz",
            lk_summary_json=root / "lk_stage_summary.json",
            flow_filtered_csv=root / "lk_flow_physical_filtered.csv",
            flow_filtered_npz=root / "lk_flow_physical_filtered.npz",
            flow_summary_json=root / "flow_filter_summary.json",
            velocity_raw_csv=root / "lk_flow_gyrocomp_velocity_frd_37ms.csv",
            velocity_filtered_csv=root / "lk_flow_gyrocomp_velocity_frd_37ms_filtered.csv",
            velocity_summary_json=root / "velocity_stage_summary.json",
            derived_bag_dir=root / "generated_with_velocity",
            bag_summary_json=root / "derived_bag_summary.json",
            pipeline_summary_json=root / "preprocessing_summary.json",
        )


def _artifacts(config: VisionVelocityConfig) -> PreprocessingArtifacts:
    return PreprocessingArtifacts.under(config.artifact_dir)


def _config_fingerprint(config: VisionVelocityConfig) -> str:
    payload = json.dumps(config.serializable(), sort_keys=True).encode("utf-8")
    return sha256(payload).hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot encode {type(value).__name__} as JSON")


def _sanitize_json(value: Any) -> Any:
    """Convert numpy values and non-finite floats to strict JSON values."""

    if isinstance(value, Mapping):
        return {str(key): _sanitize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_json(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return _sanitize_json(value.tolist())
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _sanitize_json(payload),
            indent=2,
            sort_keys=True,
            default=_json_value,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _summary_matches(path: Path, config: VisionVelocityConfig) -> bool:
    if not path.is_file():
        return False
    try:
        return _read_json(path).get("config_fingerprint") == _config_fingerprint(config)
    except (OSError, ValueError, TypeError):
        return False


def _validate_inputs(config: VisionVelocityConfig) -> None:
    required = (
        config.input_bag_dir,
        config.input_bag_dir / "metadata.yaml",
        config.thermal_calibration_file,
        config.camera_to_imu_file,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing preprocessing inputs:\n  " + "\n  ".join(missing))

    source = config.input_bag_dir.resolve()
    artifact = config.artifact_dir.resolve()
    if artifact == source or source in artifact.parents:
        raise ValueError("artifact_dir must not be inside the read-only source bag")


def _require_cv2() -> Any:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is unavailable in this kernel. Use the Project CV ROS 2 "
            "Humble kernel; do not install packages from inside the notebook."
        ) from exc
    return cv2


def _require_rosbags() -> tuple[Any, Any, Any]:
    try:
        from rosbags.highlevel import AnyReader
        from rosbags.typesys import Stores, get_typestore
    except ImportError as exc:
        raise RuntimeError(
            "rosbags is unavailable in this kernel. Use the Project CV ROS 2 "
            "Humble kernel."
        ) from exc
    return AnyReader, Stores, get_typestore


def _typestore() -> Any:
    _, stores, get_typestore = _require_rosbags()
    return get_typestore(stores.ROS2_HUMBLE)


def _stamp_to_ns(stamp: Any) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _choose_image_connection(reader: Any, requested: str) -> Any:
    image_types = {"sensor_msgs/msg/Image", "sensor_msgs/msg/CompressedImage"}
    candidates = [conn for conn in reader.connections if conn.msgtype in image_types]

    if requested:
        selected = [conn for conn in candidates if conn.topic == requested]
        if selected:
            return selected[0]
        available = ", ".join(f"{c.topic} ({c.msgtype})" for c in candidates)
        raise RuntimeError(
            f"Requested image topic {requested!r} was not found. "
            f"Available image topics: {available or 'none'}"
        )

    priorities = ("image_raw", "thermal", "ir", "rgb", "camera", "image", "video")
    for word in priorities:
        for conn in candidates:
            if word in conn.topic.lower():
                return conn
    if candidates:
        return candidates[0]
    raise RuntimeError("The source bag has no Image or CompressedImage topic")


def _normalize_u8(image: np.ndarray | None) -> np.ndarray | None:
    if image is None:
        return None
    if image.dtype == np.uint8:
        return image

    values = image.astype(np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(values.shape, dtype=np.uint8)
    low, high = np.percentile(finite, [1, 99])
    if high <= low + 1e-6:
        return np.zeros(values.shape, dtype=np.uint8)
    scaled = (values - low) * 255.0 / (high - low)
    return np.clip(scaled, 0, 255).astype(np.uint8)


def _image_message_to_gray(message: Any, message_type: str, cv2: Any) -> np.ndarray | None:
    if message_type == "sensor_msgs/msg/CompressedImage":
        encoded = np.asarray(message.data, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
        image = _normalize_u8(image)
        if image is None or image.ndim == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    encoding = str(message.encoding or "").lower()
    height = int(message.height)
    width = int(message.width)
    step = int(message.step)
    raw = np.asarray(message.data, dtype=np.uint8).tobytes()

    if encoding in {"mono8", "8uc1"}:
        return np.frombuffer(raw, dtype=np.uint8).reshape(height, step)[:, :width].copy()
    if encoding in {"mono16", "16uc1"}:
        row_words = step // 2
        image = np.frombuffer(raw, dtype=np.uint16).reshape(height, row_words)[:, :width]
        return _normalize_u8(image)
    if encoding == "32fc1":
        row_floats = step // 4
        image = np.frombuffer(raw, dtype=np.float32).reshape(height, row_floats)[:, :width]
        return _normalize_u8(image)
    if encoding in {"rgb8", "bgr8", "rgba8", "bgra8"}:
        channels = 4 if "a" in encoding else 3
        row_pixels = step // channels
        image = np.frombuffer(raw, dtype=np.uint8).reshape(
            height, row_pixels, channels
        )[:, :width, :]
        conversion = {
            "rgb8": cv2.COLOR_RGB2GRAY,
            "bgr8": cv2.COLOR_BGR2GRAY,
            "rgba8": cv2.COLOR_RGBA2GRAY,
            "bgra8": cv2.COLOR_BGRA2GRAY,
        }[encoding]
        return cv2.cvtColor(image, conversion)
    raise RuntimeError(f"Unsupported image encoding: {message.encoding!r}")


def _preprocess_gray(gray: np.ndarray | None, cv2: Any) -> np.ndarray | None:
    normalized = _normalize_u8(gray)
    if normalized is None:
        return None
    normalized = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(normalized)
    return cv2.GaussianBlur(normalized, (3, 3), 0)


def _invalid_lk(n_raw: int = 0, n_good: int = 0) -> dict[str, float | int | bool]:
    return {
        "dx": 0.0,
        "dy": 0.0,
        "n_raw": n_raw,
        "n_good": n_good,
        "quality": n_good / max(1, n_raw),
        "valid": False,
    }


def _robust_lk_flow(
    previous: np.ndarray,
    current: np.ndarray,
    config: VisionVelocityConfig,
    cv2: Any,
) -> dict[str, float | int | bool]:
    points0 = cv2.goodFeaturesToTrack(
        previous,
        maxCorners=config.max_corners,
        qualityLevel=config.quality_level,
        minDistance=config.min_distance,
        blockSize=config.block_size,
        useHarrisDetector=False,
    )
    if points0 is None or len(points0) < config.min_good_tracks:
        return _invalid_lk(0 if points0 is None else int(len(points0)))

    lk_params = {
        "winSize": (config.win_size, config.win_size),
        "maxLevel": config.max_level,
        "criteria": (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            30,
            0.01,
        ),
    }
    points1, status1, _ = cv2.calcOpticalFlowPyrLK(
        previous, current, points0, None, **lk_params
    )
    if points1 is None:
        return _invalid_lk(int(len(points0)))
    points0_back, status2, _ = cv2.calcOpticalFlowPyrLK(
        current, previous, points1, None, **lk_params
    )
    if points0_back is None:
        return _invalid_lk(int(len(points0)))

    p0 = points0.reshape(-1, 2)
    p1 = points1.reshape(-1, 2)
    p0_back = points0_back.reshape(-1, 2)
    status1 = status1.reshape(-1).astype(bool)
    status2 = status2.reshape(-1).astype(bool)

    displacement = p1 - p0
    forward_backward_error = np.linalg.norm(p0 - p0_back, axis=1)
    magnitude = np.linalg.norm(displacement, axis=1)
    keep = (
        status1
        & status2
        & np.isfinite(forward_backward_error)
        & np.isfinite(magnitude)
        & (forward_backward_error <= config.fb_check_px)
        & (magnitude <= config.max_motion_px)
    )
    displacement = displacement[keep]
    n_raw = int(len(p0))
    n_forward_backward = int(len(displacement))
    if n_forward_backward < config.min_good_tracks:
        return _invalid_lk(n_raw, n_forward_backward)

    median = np.median(displacement, axis=0)
    residual = np.linalg.norm(displacement - median[None, :], axis=1)
    residual_median = np.median(residual)
    mad = np.median(np.abs(residual - residual_median)) + 1e-6
    clean = displacement[residual <= residual_median + 3.5 * mad]
    n_good = int(len(clean))
    if n_good < config.min_good_tracks:
        final_median = median
        valid = False
    else:
        final_median = np.median(clean, axis=0)
        valid = True

    return {
        "dx": float(final_median[0]),
        "dy": float(final_median[1]),
        "n_raw": n_raw,
        "n_good": n_good,
        "quality": n_good / max(1, n_raw),
        "valid": valid,
    }


def compute_sparse_lk(
    config: VisionVelocityConfig,
    *,
    reuse_existing: bool = False,
) -> Path:
    """Compute legacy sparse Lucas-Kanade flow and write its raw CSV."""

    _validate_inputs(config)
    artifacts = _artifacts(config)
    artifacts.root.mkdir(parents=True, exist_ok=True)
    if (
        reuse_existing
        and artifacts.lk_raw_csv.is_file()
        and artifacts.lk_raw_npz.is_file()
        and _summary_matches(artifacts.lk_summary_json, config)
    ):
        return artifacts.lk_raw_csv

    cv2 = _require_cv2()
    AnyReader, _, _ = _require_rosbags()
    typestore = _typestore()

    columns = (
        "frame_idx",
        "stamp_ns",
        "t_sec",
        "dt_s",
        "dx_px_raw_median",
        "dy_px_raw_median",
        "flow_x_px_s_raw",
        "flow_y_px_s_raw",
        "flow_x_px_s_lpf",
        "flow_y_px_s_lpf",
        "dx_px_lpf_equiv",
        "dy_px_lpf_equiv",
        "quality_good_tracks_ratio",
        "n_good_tracks",
        "n_raw_features",
        "valid",
    )
    rows: list[list[float | int]] = []
    previous_gray: np.ndarray | None = None
    previous_stamp_ns: int | None = None
    filtered_x = 0.0
    filtered_y = 0.0
    image_count = 0
    selected_image_count = 0
    valid_count = 0
    decode_error_count = 0
    first_image_stamp_ns: int | None = None
    window_start_ns: int | None = None
    window_end_ns: int | None = None
    last_selected_stamp_ns: int | None = None

    with AnyReader([config.input_bag_dir], default_typestore=typestore) as reader:
        image_connection = _choose_image_connection(reader, config.image_topic)
        for connection, timestamp_ns, raw in reader.messages(connections=[image_connection]):
            image_count += 1
            message = reader.deserialize(raw, connection.msgtype)
            stamp_ns = int(timestamp_ns)
            if config.use_header_stamp and hasattr(message, "header"):
                header_stamp_ns = _stamp_to_ns(message.header.stamp)
                if header_stamp_ns > 0:
                    stamp_ns = header_stamp_ns

            if first_image_stamp_ns is None:
                first_image_stamp_ns = stamp_ns
                window_start_ns = first_image_stamp_ns + int(
                    round(config.start_offset_sec * 1e9)
                )
                if config.duration_sec is not None:
                    window_end_ns = window_start_ns + int(round(config.duration_sec * 1e9))

            assert window_start_ns is not None
            if stamp_ns < window_start_ns:
                continue
            if window_end_ns is not None and stamp_ns > window_end_ns:
                break

            selected_image_count += 1
            last_selected_stamp_ns = stamp_ns
            try:
                gray = _image_message_to_gray(message, connection.msgtype, cv2)
                gray = _preprocess_gray(gray, cv2)
            except Exception:
                decode_error_count += 1
                previous_gray = None
                previous_stamp_ns = None
                continue
            if gray is None:
                decode_error_count += 1
                previous_gray = None
                previous_stamp_ns = None
                continue

            if previous_gray is not None and previous_stamp_ns is not None:
                delta_sec = (stamp_ns - previous_stamp_ns) * 1e-9
                if 1e-4 <= delta_sec <= 1.0:
                    result = _robust_lk_flow(previous_gray, gray, config, cv2)
                    dx = float(result["dx"])
                    dy = float(result["dy"])
                    raw_x = dx / delta_sec
                    raw_y = dy / delta_sec
                    if config.lk_lpf_cutoff_hz > 0:
                        tau = 1.0 / (2.0 * math.pi * config.lk_lpf_cutoff_hz)
                        alpha = delta_sec / (tau + delta_sec)
                    else:
                        alpha = 1.0
                    filtered_x += alpha * (raw_x - filtered_x)
                    filtered_y += alpha * (raw_y - filtered_y)
                    valid = int(bool(result["valid"]))
                    valid_count += valid
                    rows.append(
                        [
                            image_count,
                            stamp_ns,
                            stamp_ns * 1e-9,
                            delta_sec,
                            dx,
                            dy,
                            raw_x,
                            raw_y,
                            filtered_x,
                            filtered_y,
                            filtered_x * delta_sec,
                            filtered_y * delta_sec,
                            float(result["quality"]),
                            int(result["n_good"]),
                            int(result["n_raw"]),
                            valid,
                        ]
                    )

            previous_gray = gray
            previous_stamp_ns = stamp_ns

        image_topic = image_connection.topic
        image_type = image_connection.msgtype

    if not rows:
        raise RuntimeError("Sparse LK produced no rows for the selected time window")

    frame = pd.DataFrame(rows, columns=columns)
    frame.to_csv(artifacts.lk_raw_csv, index=False)
    np.savez_compressed(
        artifacts.lk_raw_npz,
        data=frame.to_numpy(dtype=np.float64),
        columns=np.asarray(columns),
        image_topic=image_topic,
        image_type=image_type,
        lpf_cutoff_hz=float(config.lk_lpf_cutoff_hz),
    )
    summary = {
        "stage": "sparse_lk",
        "config_fingerprint": _config_fingerprint(config),
        "input_bag_dir": str(config.input_bag_dir),
        "output_csv": str(artifacts.lk_raw_csv),
        "output_npz": str(artifacts.lk_raw_npz),
        "image_topic": image_topic,
        "image_type": image_type,
        "image_messages_scanned": image_count,
        "image_messages_in_window": selected_image_count,
        "decode_errors": decode_error_count,
        "flow_rows": len(frame),
        "valid_rows": valid_count,
        "valid_ratio": valid_count / len(frame),
        "first_image_stamp_ns": first_image_stamp_ns,
        "window_start_ns": window_start_ns,
        "window_end_ns": window_end_ns,
        "last_selected_stamp_ns": last_selected_stamp_ns,
        "legacy_baseline_notes": LEGACY_BASELINE_NOTES,
    }
    _write_json(artifacts.lk_summary_json, summary)
    return artifacts.lk_raw_csv


def _rolling_mad(values: np.ndarray) -> float:
    median = np.median(values)
    return float(np.median(np.abs(values - median)) + 1e-9)


def filter_sparse_lk(
    config: VisionVelocityConfig,
    *,
    input_csv: Path | None = None,
    reuse_existing: bool = False,
) -> Path:
    """Apply the legacy physical gates, causal MAD filter and 1 Hz IIR."""

    artifacts = _artifacts(config)
    source = Path(input_csv) if input_csv is not None else artifacts.lk_raw_csv
    if not source.is_file():
        raise FileNotFoundError(f"Sparse LK CSV does not exist: {source}")
    if (
        reuse_existing
        and artifacts.flow_filtered_csv.is_file()
        and artifacts.flow_filtered_npz.is_file()
        and _summary_matches(artifacts.flow_summary_json, config)
    ):
        return artifacts.flow_filtered_csv

    frame = pd.read_csv(source)
    if frame.empty:
        raise ValueError(f"Sparse LK CSV is empty: {source}")

    width = float(config.image_width)
    height = float(config.image_height)
    focal_px = width / (2.0 * np.tan(np.deg2rad(config.physical_fov_x_deg) / 2.0))
    fov_y_deg = float(
        np.rad2deg(2.0 * np.arctan(height / (2.0 * focal_px)))
    )
    ground_width_m = 2.0 * config.physical_height_m * np.tan(
        np.deg2rad(config.physical_fov_x_deg) / 2.0
    )
    ground_height_m = 2.0 * config.physical_height_m * np.tan(
        np.deg2rad(fov_y_deg) / 2.0
    )
    metres_per_px_x = ground_width_m / width
    metres_per_px_y = ground_height_m / height
    max_flow_x_px_s = config.physical_v_max_mps / metres_per_px_x
    max_flow_y_px_s = config.physical_v_max_mps / metres_per_px_y
    max_dx_px_frame = max_flow_x_px_s / config.nominal_fps
    max_dy_px_frame = max_flow_y_px_s / config.nominal_fps

    frame["t_rel"] = frame["t_sec"] - frame["t_sec"].iloc[0]
    median_dt = float(frame["dt_s"].median())
    frame["flow_mag_raw"] = np.hypot(
        frame["flow_x_px_s_raw"], frame["flow_y_px_s_raw"]
    )
    frame["phys_bad_x"] = np.abs(frame["flow_x_px_s_raw"]) > max_flow_x_px_s
    frame["phys_bad_y"] = np.abs(frame["flow_y_px_s_raw"]) > max_flow_y_px_s
    frame["phys_bad_dx"] = np.abs(frame["dx_px_raw_median"]) > max_dx_px_frame
    frame["phys_bad_dy"] = np.abs(frame["dy_px_raw_median"]) > max_dy_px_frame
    frame["dt_bad"] = (frame["dt_s"] < 0.5 * median_dt) | (
        frame["dt_s"] > 2.0 * median_dt
    )
    frame["quality_bad"] = (
        (frame["valid"] < 0.5)
        | (frame["n_good_tracks"] < 10)
        | (frame["quality_good_tracks_ratio"] < 0.01)
    )
    frame["physical_bad"] = (
        frame["phys_bad_x"]
        | frame["phys_bad_y"]
        | frame["phys_bad_dx"]
        | frame["phys_bad_dy"]
    )
    frame["flow_x_px_s_clipped"] = frame["flow_x_px_s_raw"].clip(
        lower=-max_flow_x_px_s, upper=max_flow_x_px_s
    )
    frame["flow_y_px_s_clipped"] = frame["flow_y_px_s_raw"].clip(
        lower=-max_flow_y_px_s, upper=max_flow_y_px_s
    )

    estimated_fps = 1.0 / median_dt if median_dt > 1e-6 else config.nominal_fps
    window = max(31, int(round(estimated_fps)))
    if window % 2 == 0:
        window += 1
    min_periods = max(5, window // 5)
    rolling_x = (
        frame["flow_x_px_s_clipped"]
        .rolling(window, center=False, min_periods=min_periods)
        .median()
        .bfill()
        .ffill()
    )
    rolling_y = (
        frame["flow_y_px_s_clipped"]
        .rolling(window, center=False, min_periods=min_periods)
        .median()
        .bfill()
        .ffill()
    )
    frame["resid_x"] = frame["flow_x_px_s_clipped"] - rolling_x
    frame["resid_y"] = frame["flow_y_px_s_clipped"] - rolling_y
    frame["resid_mag"] = np.hypot(frame["resid_x"], frame["resid_y"])
    residual_median = (
        frame["resid_mag"]
        .rolling(window, center=False, min_periods=min_periods)
        .median()
        .bfill()
        .ffill()
    )
    residual_mad = (
        frame["resid_mag"]
        .rolling(window, center=False, min_periods=min_periods)
        .apply(_rolling_mad, raw=True)
        .bfill()
        .ffill()
    )
    frame["outlier_score"] = (
        0.6745 * (frame["resid_mag"] - residual_median) / residual_mad
    )
    frame["mad_bad"] = frame["outlier_score"] > config.physical_outlier_z_threshold
    frame["is_bad_measurement"] = (
        frame["dt_bad"]
        | frame["quality_bad"]
        | frame["physical_bad"]
        | frame["mad_bad"]
    )

    clean_x: list[float] = []
    clean_y: list[float] = []
    filtered_x: list[float] = []
    filtered_y: list[float] = []
    previous_good_x = 0.0
    previous_good_y = 0.0
    previous_filtered_x = 0.0
    previous_filtered_y = 0.0
    hold_time = 0.0

    for row in frame.itertuples(index=False):
        delta_sec = float(row.dt_s)
        raw_x = float(row.flow_x_px_s_clipped)
        raw_y = float(row.flow_y_px_s_clipped)
        if not bool(row.is_bad_measurement):
            value_x = raw_x
            value_y = raw_y
            previous_good_x = value_x
            previous_good_y = value_y
            hold_time = 0.0
        else:
            hold_time += max(delta_sec, 0.0)
            if hold_time <= config.max_hold_gap_sec:
                value_x = previous_good_x
                value_y = previous_good_y
            else:
                decay = np.exp(-(hold_time - config.max_hold_gap_sec) / 0.5)
                value_x = previous_good_x * decay
                value_y = previous_good_y * decay

        if config.physical_lpf_cutoff_hz > 0 and delta_sec > 0:
            tau = 1.0 / (2.0 * np.pi * config.physical_lpf_cutoff_hz)
            alpha = delta_sec / (tau + delta_sec)
        else:
            alpha = 1.0
        previous_filtered_x += alpha * (value_x - previous_filtered_x)
        previous_filtered_y += alpha * (value_y - previous_filtered_y)
        clean_x.append(float(value_x))
        clean_y.append(float(value_y))
        filtered_x.append(float(previous_filtered_x))
        filtered_y.append(float(previous_filtered_y))

    frame["flow_x_px_s_clean"] = clean_x
    frame["flow_y_px_s_clean"] = clean_y
    frame["flow_x_px_s_filt"] = filtered_x
    frame["flow_y_px_s_filt"] = filtered_y
    frame["flow_mag_clipped"] = np.hypot(
        frame["flow_x_px_s_clipped"], frame["flow_y_px_s_clipped"]
    )
    frame["flow_mag_filt"] = np.hypot(
        frame["flow_x_px_s_filt"], frame["flow_y_px_s_filt"]
    )
    frame["dx_px_filt_equiv"] = frame["flow_x_px_s_filt"] * frame["dt_s"]
    frame["dy_px_filt_equiv"] = frame["flow_y_px_s_filt"] * frame["dt_s"]
    frame["vx_mps_filt"] = frame["flow_x_px_s_filt"] * metres_per_px_x
    frame["vy_mps_filt"] = frame["flow_y_px_s_filt"] * metres_per_px_y
    frame["v_mag_mps_filt"] = np.hypot(
        frame["vx_mps_filt"], frame["vy_mps_filt"]
    )
    frame["valid_filt"] = (~frame["is_bad_measurement"]).astype(float)

    artifacts.root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(artifacts.flow_filtered_csv, index=False)
    np.savez_compressed(
        artifacts.flow_filtered_npz,
        data=frame.to_numpy(),
        columns=np.asarray(frame.columns),
    )
    summary = {
        "stage": "physical_flow_filter",
        "config_fingerprint": _config_fingerprint(config),
        "input_csv": str(source),
        "output_csv": str(artifacts.flow_filtered_csv),
        "rows": len(frame),
        "bad_measurements": int(frame["is_bad_measurement"].sum()),
        "bad_ratio": float(frame["is_bad_measurement"].mean()),
        "physical_bad": int(frame["physical_bad"].sum()),
        "mad_bad": int(frame["mad_bad"].sum()),
        "dt_bad": int(frame["dt_bad"].sum()),
        "quality_bad": int(frame["quality_bad"].sum()),
        "rolling_window": window,
        "estimated_fps": estimated_fps,
        "metres_per_px_x_at_legacy_8m": metres_per_px_x,
        "metres_per_px_y_at_legacy_8m": metres_per_px_y,
        "legacy_baseline_notes": LEGACY_BASELINE_NOTES,
    }
    _write_json(artifacts.flow_summary_json, summary)
    return artifacts.flow_filtered_csv


def _load_opencv_matrix(
    path: Path,
    *,
    preferred_keys: Sequence[str] = (),
) -> tuple[np.ndarray, str]:
    cv2 = _require_cv2()
    keys = list(preferred_keys) + [
        "extrinsicRotation",
        "extrinsic_rotation",
        "cam_to_imu_rot_mtrx",
        "R_cam_to_imu",
        "R_camera_to_imu",
        "camera_to_imu",
        "camera_matrix",
        "cameramatrix",
        "cameraMatrix",
        "intrinsic_matrix",
        "camera_mtx",
        "mtx",
        "K",
        "R",
        "rotation_matrix",
    ]
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if storage.isOpened():
        for key in keys:
            node = storage.getNode(key)
            if not node.empty():
                matrix = node.mat()
                if matrix is not None and matrix.size > 0:
                    storage.release()
                    return np.asarray(matrix, dtype=float), key
        storage.release()

    text = path.read_text(encoding="utf-8", errors="replace")
    pattern = re.compile(
        r"([A-Za-z0-9_]+)\s*:\s*!!opencv-matrix[^\n]*\n"
        r"\s*rows\s*:\s*(\d+)\s*\n"
        r"\s*cols\s*:\s*(\d+)\s*\n"
        r"\s*dt\s*:\s*[A-Za-z]+\s*\n"
        r"\s*data\s*:\s*\[([^\]]+)\]",
        flags=re.MULTILINE | re.DOTALL,
    )
    for match in pattern.finditer(text):
        key = match.group(1)
        rows = int(match.group(2))
        columns = int(match.group(3))
        values = [
            float(value)
            for value in re.split(r"[,\s]+", match.group(4).strip())
            if value
        ]
        return np.asarray(values, dtype=float).reshape(rows, columns), key
    raise RuntimeError(f"Could not read an OpenCV matrix from {path}")


def _interp_scalar(
    source_time: np.ndarray,
    source_value: np.ndarray,
    query_time: np.ndarray,
) -> np.ndarray:
    valid = np.isfinite(source_time) & np.isfinite(source_value)
    source_time = np.asarray(source_time[valid], dtype=float)
    source_value = np.asarray(source_value[valid], dtype=float)
    order = np.argsort(source_time)
    source_time = source_time[order]
    source_value = source_value[order]
    unique_time, unique_indices = np.unique(source_time, return_index=True)
    if len(unique_time) == 0:
        return np.full_like(query_time, np.nan, dtype=float)
    return np.interp(
        query_time,
        unique_time,
        source_value[unique_indices],
        left=np.nan,
        right=np.nan,
    )


def _interp_vector(
    source_time: np.ndarray,
    source_value: np.ndarray,
    query_time: np.ndarray,
) -> np.ndarray:
    output = np.empty((len(query_time), source_value.shape[1]), dtype=float)
    for column in range(source_value.shape[1]):
        output[:, column] = _interp_scalar(
            source_time, source_value[:, column], query_time
        )
    return output


def _interp_nan_by_time(time: np.ndarray, value: np.ndarray) -> np.ndarray:
    valid = np.isfinite(time) & np.isfinite(value)
    output = np.full_like(value, np.nan, dtype=float)
    if valid.sum() < 2:
        return output
    output[:] = np.interp(time, time[valid], value[valid], left=np.nan, right=np.nan)
    return output


def _robust_sigma(value: pd.Series) -> float:
    values = value.to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 10:
        return float("nan")
    median = np.median(values)
    return float(1.4826 * np.median(np.abs(values - median)))


def _causal_iir_lpf_by_time(
    time: np.ndarray,
    value: np.ndarray,
    tau_sec: float,
) -> np.ndarray:
    output = np.full_like(value, np.nan, dtype=float)
    valid = np.isfinite(time) & np.isfinite(value)
    if valid.sum() == 0:
        return output
    first = int(np.where(valid)[0][0])
    output[first] = value[first]
    for index in range(first + 1, len(value)):
        if (
            not np.isfinite(value[index])
            or not np.isfinite(time[index])
            or not np.isfinite(time[index - 1])
        ):
            output[index] = output[index - 1]
            continue
        delta_sec = max(1e-4, time[index] - time[index - 1])
        alpha = delta_sec / (tau_sec + delta_sec)
        if not np.isfinite(output[index - 1]):
            output[index] = value[index]
        else:
            output[index] = output[index - 1] + alpha * (
                value[index] - output[index - 1]
            )
    return output


def _read_auxiliary_topics(
    config: VisionVelocityConfig,
    *,
    start_sec: float,
    end_sec: float,
) -> tuple[list[tuple[float, float, float, float]], list[tuple[float, float]], list[tuple[float, float, float, float]]]:
    AnyReader, _, _ = _require_rosbags()
    typestore = _typestore()
    imu_rows: list[tuple[float, float, float, float]] = []
    range_rows: list[tuple[float, float]] = []
    euler_rows: list[tuple[float, float, float, float]] = []
    padding_sec = 1.0
    lower = start_sec - padding_sec
    upper = end_sec + padding_sec

    with AnyReader([config.input_bag_dir], default_typestore=typestore) as reader:
        by_topic = {
            topic: [conn for conn in reader.connections if conn.topic == topic]
            for topic in (config.imu_topic, config.range_topic, config.euler_topic)
        }
        if not by_topic[config.imu_topic]:
            raise RuntimeError(f"Missing topic: {config.imu_topic}")
        if not by_topic[config.range_topic]:
            raise RuntimeError(f"Missing topic: {config.range_topic}")
        connections = sum(by_topic.values(), [])
        for connection, timestamp_ns, raw in reader.messages(connections=connections):
            time_sec = timestamp_ns * 1e-9
            if time_sec < lower or time_sec > upper:
                continue
            message = reader.deserialize(raw, connection.msgtype)
            if connection.topic == config.imu_topic:
                imu_rows.append(
                    (
                        time_sec,
                        float(message.angular_velocity.x),
                        float(message.angular_velocity.y),
                        float(message.angular_velocity.z),
                    )
                )
            elif connection.topic == config.range_topic:
                range_rows.append((time_sec, float(message.range)))
            elif connection.topic == config.euler_topic:
                euler_rows.append(
                    (
                        time_sec,
                        float(message.vector.x),
                        float(message.vector.y),
                        float(message.vector.z),
                    )
                )
    return imu_rows, range_rows, euler_rows


def compute_gyrocompensated_velocity(
    config: VisionVelocityConfig,
    *,
    filtered_flow_csv: Path | None = None,
    reuse_existing: bool = False,
) -> tuple[Path, Path]:
    """Compute the legacy gyro-compensated FRD velocity CSV files."""

    _validate_inputs(config)
    artifacts = _artifacts(config)
    flow_path = (
        Path(filtered_flow_csv)
        if filtered_flow_csv is not None
        else artifacts.flow_filtered_csv
    )
    if not flow_path.is_file():
        raise FileNotFoundError(f"Filtered flow CSV does not exist: {flow_path}")
    if (
        reuse_existing
        and artifacts.velocity_raw_csv.is_file()
        and artifacts.velocity_filtered_csv.is_file()
        and _summary_matches(artifacts.velocity_summary_json, config)
    ):
        return artifacts.velocity_raw_csv, artifacts.velocity_filtered_csv

    flow = pd.read_csv(flow_path)
    required_flow = ("t_sec", "dt_s", "dx_px_filt_equiv", "dy_px_filt_equiv")
    missing = [column for column in required_flow if column not in flow.columns]
    if missing:
        raise ValueError(f"Filtered flow CSV is missing columns: {missing}")
    if flow.empty:
        raise ValueError(f"Filtered flow CSV is empty: {flow_path}")

    camera_matrix, camera_key = _load_opencv_matrix(
        config.thermal_calibration_file,
        preferred_keys=("cameramatrix", "camera_matrix", "cameraMatrix", "K"),
    )
    camera_matrix = camera_matrix.reshape(3, 3)
    focal_x = float(camera_matrix[0, 0])
    focal_y = float(camera_matrix[1, 1])
    center_x = float(camera_matrix[0, 2])
    center_y = float(camera_matrix[1, 2])

    rotation, rotation_key = _load_opencv_matrix(
        config.camera_to_imu_file,
        preferred_keys=("extrinsicRotation", "cam_to_imu_rot_mtrx", "R_cam_to_imu", "R"),
    )
    if rotation.shape == (4, 4):
        rotation = rotation[:3, :3]
    rotation = rotation.reshape(3, 3)
    camera_to_frd = rotation if config.rotation_file_is_camera_to_imu else rotation.T
    frd_to_camera = camera_to_frd.T

    time_flow = flow["t_sec"].to_numpy(dtype=float)
    delta_flow = flow["dt_s"].to_numpy(dtype=float)
    median_delta = float(
        np.nanmedian(
            np.diff(time_flow)[
                np.isfinite(np.diff(time_flow)) & (np.diff(time_flow) > 0)
            ]
        )
    )
    bad_delta = (
        ~np.isfinite(delta_flow) | (delta_flow <= 0) | (delta_flow > 0.5)
    )
    delta_flow[bad_delta] = median_delta
    dx_px = config.flow_sign * flow["dx_px_filt_equiv"].to_numpy(dtype=float)
    dy_px = config.flow_sign * flow["dy_px_filt_equiv"].to_numpy(dtype=float)
    effective_end = time_flow - config.flow_lag_sec
    effective_start = effective_end - delta_flow

    imu_rows, range_rows, euler_rows = _read_auxiliary_topics(
        config,
        start_sec=float(np.nanmin(effective_start)),
        end_sec=float(np.nanmax(effective_end)),
    )
    if not imu_rows:
        raise RuntimeError("No IMU messages overlap the selected flow window")
    if not range_rows:
        raise RuntimeError("No rangefinder messages overlap the selected flow window")

    imu = pd.DataFrame(
        imu_rows, columns=("t_imu", "gyro_x", "gyro_y", "gyro_z")
    ).sort_values("t_imu")
    time_imu = imu["t_imu"].to_numpy(dtype=float)
    gyro_frd = imu[["gyro_x", "gyro_y", "gyro_z"]].to_numpy(dtype=float)
    valid_imu = np.isfinite(time_imu) & np.isfinite(gyro_frd).all(axis=1)
    time_imu = time_imu[valid_imu]
    gyro_frd = gyro_frd[valid_imu]
    order = np.argsort(time_imu)
    time_imu = time_imu[order]
    gyro_frd = gyro_frd[order]
    time_imu, unique_indices = np.unique(time_imu, return_index=True)
    gyro_frd = gyro_frd[unique_indices]
    cumulative_gyro = np.zeros_like(gyro_frd)
    for index in range(1, len(time_imu)):
        step = time_imu[index] - time_imu[index - 1]
        cumulative_gyro[index] = cumulative_gyro[index - 1] + 0.5 * step * (
            gyro_frd[index - 1] + gyro_frd[index]
        )
    gyro_start = _interp_vector(time_imu, cumulative_gyro, effective_start)
    gyro_end = _interp_vector(time_imu, cumulative_gyro, effective_end)
    gyro_delta_frd = gyro_end - gyro_start
    gyro_delta_camera = (frd_to_camera @ gyro_delta_frd.T).T

    dx_normalized = dx_px / focal_x
    dy_normalized = dy_px / focal_y
    angular_flow_camera = np.column_stack(
        (dy_normalized, -dx_normalized, np.zeros_like(dx_normalized))
    )
    compensated_camera = angular_flow_camera - gyro_delta_camera
    dx_normalized_compensated = -compensated_camera[:, 1]
    dy_normalized_compensated = compensated_camera[:, 0]
    dx_normalized_rate = dx_normalized_compensated / delta_flow
    dy_normalized_rate = dy_normalized_compensated / delta_flow
    angular_flow_frd = (camera_to_frd @ angular_flow_camera.T).T
    compensated_frd = (camera_to_frd @ compensated_camera.T).T
    flow_rate_frd = angular_flow_frd / delta_flow[:, None]
    gyro_rate_frd = gyro_delta_frd / delta_flow[:, None]
    compensated_rate_frd = compensated_frd / delta_flow[:, None]

    ranges = pd.DataFrame(range_rows, columns=("t_range", "range_m")).sort_values(
        "t_range"
    )
    time_range = ranges["t_range"].to_numpy(dtype=float)
    range_m = ranges["range_m"].to_numpy(dtype=float)
    valid_range = (
        np.isfinite(time_range)
        & np.isfinite(range_m)
        & (range_m >= config.min_range_m)
        & (range_m <= config.max_range_m)
    )
    range_at_flow = _interp_scalar(
        time_range[valid_range], range_m[valid_range], effective_end
    )

    roll = np.zeros_like(time_flow)
    pitch = np.zeros_like(time_flow)
    yaw = np.zeros_like(time_flow)
    euler_converted_from_degrees = False
    if len(euler_rows) > 10:
        euler = pd.DataFrame(
            euler_rows, columns=("t_euler", "roll", "pitch", "yaw")
        ).sort_values("t_euler")
        time_euler = euler["t_euler"].to_numpy(dtype=float)
        euler_values = euler[["roll", "pitch", "yaw"]].to_numpy(dtype=float)
        finite_euler = euler_values[np.isfinite(euler_values)]
        if len(finite_euler) > 0 and np.nanpercentile(np.abs(finite_euler), 99) > 7.0:
            euler_values = np.deg2rad(euler_values)
            euler_converted_from_degrees = True
        roll = _interp_scalar(time_euler, euler_values[:, 0], effective_end)
        pitch = _interp_scalar(time_euler, euler_values[:, 1], effective_end)
        yaw = _interp_scalar(time_euler, euler_values[:, 2], effective_end)

    cosine_roll_pitch = np.cos(roll) * np.cos(pitch)
    cosine_roll_pitch = np.where(
        np.abs(cosine_roll_pitch) < 0.2, np.nan, cosine_roll_pitch
    )
    vertical_height = range_at_flow * cosine_roll_pitch
    if config.height_mode == "range_ray":
        ray_distance = range_at_flow.copy()
    elif config.height_mode == "vertical_agl":
        ray_distance = vertical_height.copy()
    else:  # vertical_to_ray
        ray_distance = range_at_flow / cosine_roll_pitch

    velocity_camera = np.column_stack(
        (
            -ray_distance * dx_normalized_rate,
            -ray_distance * dy_normalized_rate,
            np.zeros_like(ray_distance),
        )
    )
    velocity_frd = (camera_to_frd @ velocity_camera.T).T

    raw_output = pd.DataFrame(
        {
            "t_flow": time_flow,
            "t_eff0": effective_start,
            "t_eff1": effective_end,
            "dt_flow": delta_flow,
            "dx_px": dx_px,
            "dy_px": dy_px,
            "dx_norm": dx_normalized,
            "dy_norm": dy_normalized,
            "range_m": range_at_flow,
            "roll_rad": roll,
            "pitch_rad": pitch,
            "yaw_rad": yaw,
            "h_vertical_from_range_m": vertical_height,
            "Z_ray_used_m": ray_distance,
            "flow_rate_frd_x": flow_rate_frd[:, 0],
            "flow_rate_frd_y": flow_rate_frd[:, 1],
            "flow_rate_frd_z": flow_rate_frd[:, 2],
            "gyro_rate_frd_x": gyro_rate_frd[:, 0],
            "gyro_rate_frd_y": gyro_rate_frd[:, 1],
            "gyro_rate_frd_z": gyro_rate_frd[:, 2],
            "comp_rate_frd_x": compensated_rate_frd[:, 0],
            "comp_rate_frd_y": compensated_rate_frd[:, 1],
            "comp_rate_frd_z": compensated_rate_frd[:, 2],
            "dx_norm_comp": dx_normalized_compensated,
            "dy_norm_comp": dy_normalized_compensated,
            "dx_norm_comp_rate": dx_normalized_rate,
            "dy_norm_comp_rate": dy_normalized_rate,
            "vel_cam_x_mps": velocity_camera[:, 0],
            "vel_cam_y_mps": velocity_camera[:, 1],
            "vel_cam_z_mps": velocity_camera[:, 2],
            "vel_frd_x_mps": velocity_frd[:, 0],
            "vel_frd_y_mps": velocity_frd[:, 1],
            "vel_frd_z_mps": velocity_frd[:, 2],
            "FLOW_DX_COL": 6,
            "FLOW_DY_COL": 7,
        }
    )
    raw_output["valid_raw"] = (
        np.isfinite(raw_output["vel_frd_x_mps"])
        & np.isfinite(raw_output["vel_frd_y_mps"])
        & np.isfinite(raw_output["Z_ray_used_m"])
        & np.isfinite(raw_output["dt_flow"])
        & (raw_output["dt_flow"] > 0.001)
        & (raw_output["dt_flow"] < 0.2)
        & (raw_output["Z_ray_used_m"] >= config.min_range_m)
        & (raw_output["Z_ray_used_m"] <= config.max_range_m)
    )
    raw_output["valid"] = (
        raw_output["valid_raw"]
        & (np.abs(raw_output["vel_frd_x_mps"]) <= config.max_abs_raw_velocity_mps)
        & (np.abs(raw_output["vel_frd_y_mps"]) <= config.max_abs_raw_velocity_mps)
    )
    artifacts.root.mkdir(parents=True, exist_ok=True)
    raw_output.to_csv(artifacts.velocity_raw_csv, index=False)

    filtered = raw_output.copy()
    gate1 = (
        np.isfinite(filtered["vel_frd_x_mps"])
        & np.isfinite(filtered["vel_frd_y_mps"])
        & np.isfinite(filtered["Z_ray_used_m"])
        & np.isfinite(filtered["dt_flow"])
        & (filtered["dt_flow"] >= config.min_filter_dt_sec)
        & (filtered["dt_flow"] <= config.max_filter_dt_sec)
        & (filtered["Z_ray_used_m"] >= config.min_filter_height_m)
        & (filtered["Z_ray_used_m"] <= config.max_filter_height_m)
        & (np.abs(filtered["vel_frd_x_mps"]) <= config.max_abs_raw_velocity_mps)
        & (np.abs(filtered["vel_frd_y_mps"]) <= config.max_abs_raw_velocity_mps)
    )
    filtered["valid_gate1"] = gate1
    velocity_x = filtered["vel_frd_x_mps"].where(gate1)
    velocity_y = filtered["vel_frd_y_mps"].where(gate1)
    median_x = velocity_x.rolling(
        config.velocity_median_window, center=True, min_periods=1
    ).median()
    median_y = velocity_y.rolling(
        config.velocity_median_window, center=True, min_periods=1
    ).median()
    residual_x = velocity_x - median_x
    residual_y = velocity_y - median_y
    sigma_x = _robust_sigma(residual_x)
    sigma_y = _robust_sigma(residual_y)
    if not np.isfinite(sigma_x) or sigma_x < 1e-6:
        sigma_x = float(np.nanstd(residual_x.to_numpy(dtype=float)))
    if not np.isfinite(sigma_y) or sigma_y < 1e-6:
        sigma_y = float(np.nanstd(residual_y.to_numpy(dtype=float)))
    gate2 = gate1 & (np.abs(residual_x) <= 4.0 * sigma_x) & (
        np.abs(residual_y) <= 4.0 * sigma_y
    )
    filtered["valid_gate2"] = gate2
    clean_velocity_x = filtered["vel_frd_x_mps"].where(gate2)
    clean_velocity_y = filtered["vel_frd_y_mps"].where(gate2)
    time_values = filtered["t_flow"].to_numpy(dtype=float)
    interpolated_x = _interp_nan_by_time(
        time_values, clean_velocity_x.to_numpy(dtype=float)
    )
    interpolated_y = _interp_nan_by_time(
        time_values, clean_velocity_y.to_numpy(dtype=float)
    )
    lowpass_x = _causal_iir_lpf_by_time(
        time_values, interpolated_x, config.velocity_lpf_tau_sec
    )
    lowpass_y = _causal_iir_lpf_by_time(
        time_values, interpolated_y, config.velocity_lpf_tau_sec
    )
    valid_filtered = (
        gate2
        & np.isfinite(lowpass_x)
        & np.isfinite(lowpass_y)
        & (np.abs(lowpass_x) <= config.max_abs_filtered_velocity_mps)
        & (np.abs(lowpass_y) <= config.max_abs_filtered_velocity_mps)
    )
    filtered["vel_frd_x_mps_med"] = median_x
    filtered["vel_frd_y_mps_med"] = median_y
    filtered["vel_frd_x_mps_clean"] = clean_velocity_x
    filtered["vel_frd_y_mps_clean"] = clean_velocity_y
    filtered["vel_frd_x_mps_lpf"] = lowpass_x
    filtered["vel_frd_y_mps_lpf"] = lowpass_y
    filtered["valid_filtered"] = valid_filtered
    filtered.to_csv(artifacts.velocity_filtered_csv, index=False)

    valid_rows = filtered.loc[
        filtered["valid_filtered"],
        ["vel_frd_x_mps_lpf", "vel_frd_y_mps_lpf", "Z_ray_used_m"],
    ]
    summary = {
        "stage": "gyrocompensated_velocity",
        "config_fingerprint": _config_fingerprint(config),
        "input_flow_csv": str(flow_path),
        "raw_output_csv": str(artifacts.velocity_raw_csv),
        "filtered_output_csv": str(artifacts.velocity_filtered_csv),
        "rows": len(filtered),
        "valid_raw": int(raw_output["valid"].sum()),
        "valid_gate1": int(gate1.sum()),
        "valid_gate2": int(gate2.sum()),
        "valid_filtered": int(valid_filtered.sum()),
        "camera_matrix_key": camera_key,
        "fx": focal_x,
        "fy": focal_y,
        "cx": center_x,
        "cy": center_y,
        "rotation_matrix_key": rotation_key,
        "rotation_determinant": float(np.linalg.det(camera_to_frd)),
        "rotation_orthogonality_error": float(
            np.linalg.norm(camera_to_frd.T @ camera_to_frd - np.eye(3))
        ),
        "imu_rows": len(imu_rows),
        "range_rows": len(range_rows),
        "euler_rows": len(euler_rows),
        "euler_converted_from_degrees": euler_converted_from_degrees,
        "range_median": float(np.nanmedian(range_at_flow)),
        "vertical_height_median": float(np.nanmedian(vertical_height)),
        "ray_distance_median": float(np.nanmedian(ray_distance)),
        "filtered_velocity_statistics": valid_rows.describe().to_dict(),
        "legacy_baseline_notes": LEGACY_BASELINE_NOTES,
    }
    _write_json(artifacts.velocity_summary_json, summary)
    return artifacts.velocity_raw_csv, artifacts.velocity_filtered_csv


def _to_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, (bool, np.bool_)):
        return 1.0 if bool(value) else 0.0
    text = str(value).strip().lower()
    if not text or text in {"nan", "none", "null"}:
        return default
    if text in {"true", "t", "yes", "y"}:
        return 1.0
    if text in {"false", "f", "no", "n"}:
        return 0.0
    return float(text)


def _add_connection_compat(
    writer: Any,
    *,
    topic: str,
    msgtype: str,
    typestore: Any,
    offered_qos_profiles: Any = None,
) -> Any:
    if offered_qos_profiles is not None:
        try:
            return writer.add_connection(
                topic,
                msgtype,
                typestore=typestore,
                offered_qos_profiles=offered_qos_profiles,
            )
        except TypeError:
            pass
    try:
        return writer.add_connection(topic, msgtype, typestore=typestore)
    except TypeError:
        return writer.add_connection(topic, msgtype)


def _create_mcap_writer(output_dir: Path) -> Any:
    # rosbags 0.11.3 exposes StoragePlugin beside Writer, not from
    # rosbags.interfaces. There is deliberately no SQLite fallback: the
    # generated baseline must stay an MCAP bag.
    from rosbags.rosbag2 import Writer
    from rosbags.rosbag2.writer import StoragePlugin

    return Writer(output_dir, version=8, storage_plugin=StoragePlugin.MCAP)


def _flow_events(frame: pd.DataFrame) -> list[tuple[int, list[float]]]:
    missing = [column for column in FLOW_MESSAGE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Filtered flow CSV is missing columns: {missing}")
    events: list[tuple[int, list[float]]] = []
    for row in frame.loc[:, FLOW_MESSAGE_COLUMNS].itertuples(index=False, name=None):
        # Keep the legacy CSV conversion exactly: csv.DictReader -> float ->
        # int. A nanosecond timestamp is wider than exact float64 integer
        # precision, so replacing this with the original int64 would change
        # the golden bag by a sub-microsecond amount.
        stamp_ns = int(_to_float(row[1]))
        events.append((stamp_ns, [_to_float(value) for value in row]))
    events.sort(key=lambda item: item[0])
    return events


def _velocity_events(frame: pd.DataFrame) -> list[tuple[int, float, float]]:
    required = (
        "t_flow",
        "vel_frd_x_mps_lpf",
        "vel_frd_y_mps_lpf",
        "valid_filtered",
    )
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Velocity CSV is missing columns: {missing}")
    selected = frame[
        np.isfinite(frame["t_flow"])
        & np.isfinite(frame["vel_frd_x_mps_lpf"])
        & np.isfinite(frame["vel_frd_y_mps_lpf"])
        & frame["valid_filtered"].astype(bool)
    ].copy()
    selected["timestamp_ns"] = np.round(
        selected["t_flow"].to_numpy(dtype=float) * 1e9
    ).astype(np.int64)
    selected = selected.sort_values("timestamp_ns").drop_duplicates(
        "timestamp_ns", keep="last"
    )
    return [
        (int(row.timestamp_ns), float(row.vel_frd_x_mps_lpf), float(row.vel_frd_y_mps_lpf))
        for row in selected.itertuples(index=False)
    ]


def build_derived_bag(
    config: VisionVelocityConfig,
    *,
    filtered_flow_csv: Path | None = None,
    filtered_velocity_csv: Path | None = None,
    output_bag_dir: Path | None = None,
    reuse_existing: bool = False,
    overwrite: bool = False,
) -> Path:
    """Create a generated bag containing filtered flow and FRD velocity.

    For a smoke-test window only raw messages within the selected absolute
    window are copied.  A full config copies the complete source bag.
    """

    _validate_inputs(config)
    artifacts = _artifacts(config)
    flow_path = Path(filtered_flow_csv or artifacts.flow_filtered_csv)
    velocity_path = Path(filtered_velocity_csv or artifacts.velocity_filtered_csv)
    output_dir = Path(output_bag_dir or artifacts.derived_bag_dir)
    if not flow_path.is_file() or not velocity_path.is_file():
        raise FileNotFoundError("Flow and velocity CSV files must exist before bag creation")

    artifact_root = config.artifact_dir.resolve()
    resolved_output = output_dir.resolve()
    if resolved_output != artifact_root and artifact_root not in resolved_output.parents:
        raise ValueError("Generated bag must be below config.artifact_dir")
    if output_dir.exists():
        if reuse_existing and _summary_matches(artifacts.bag_summary_json, config):
            return output_dir
        if not overwrite:
            raise FileExistsError(
                f"Generated bag already exists: {output_dir}. "
                "Use reuse_existing=True or overwrite=True."
            )
        shutil.rmtree(output_dir)

    flow_frame = pd.read_csv(flow_path)
    velocity_frame = pd.read_csv(velocity_path)
    flow_events = _flow_events(flow_frame)
    velocity_events = _velocity_events(velocity_frame)
    if not flow_events:
        raise RuntimeError("No filtered flow messages are available to write")
    if not velocity_events:
        raise RuntimeError("No valid velocity messages are available to write")

    AnyReader, _, _ = _require_rosbags()
    typestore = _typestore()
    writer_object = _create_mcap_writer(output_dir)
    Float64MultiArray = typestore.types["std_msgs/msg/Float64MultiArray"]
    MultiArrayLayout = typestore.types["std_msgs/msg/MultiArrayLayout"]
    MultiArrayDimension = typestore.types["std_msgs/msg/MultiArrayDimension"]
    Time = typestore.types["builtin_interfaces/msg/Time"]
    Header = typestore.types["std_msgs/msg/Header"]
    Vector3 = typestore.types["geometry_msgs/msg/Vector3"]
    Twist = typestore.types["geometry_msgs/msg/Twist"]
    TwistStamped = typestore.types["geometry_msgs/msg/TwistStamped"]

    def make_flow_message(values: Sequence[float]) -> Any:
        dimension = MultiArrayDimension(
            label="lk_flow_px_filtered",
            size=len(FLOW_MESSAGE_COLUMNS),
            stride=len(FLOW_MESSAGE_COLUMNS),
        )
        layout = MultiArrayLayout(dim=[dimension], data_offset=0)
        return Float64MultiArray(layout=layout, data=np.asarray(values, dtype=np.float64))

    def make_velocity_message(timestamp_ns: int, vx: float, vy: float) -> Any:
        time = Time(
            sec=int(timestamp_ns // 1_000_000_000),
            nanosec=int(timestamp_ns % 1_000_000_000),
        )
        return TwistStamped(
            header=Header(stamp=time, frame_id=config.velocity_frame_id),
            twist=Twist(
                linear=Vector3(x=vx, y=vy, z=0.0),
                angular=Vector3(x=0.0, y=0.0, z=0.0),
            ),
        )

    copy_start_ns: int | None = None
    copy_end_ns: int | None = None
    if (
        (config.start_offset_sec > 0.0 or config.duration_sec is not None)
        and artifacts.lk_summary_json.is_file()
    ):
        lk_summary = _read_json(artifacts.lk_summary_json)
        copy_start_ns = int(lk_summary["window_start_ns"])
        end_value = lk_summary.get("window_end_ns")
        if end_value is not None:
            copy_end_ns = int(end_value)

    flow_index = 0
    velocity_index = 0
    copied_count = 0
    flow_count = 0
    velocity_count = 0
    with AnyReader([config.input_bag_dir], default_typestore=typestore) as reader, writer_object as writer:
        existing_topics = {connection.topic for connection in reader.connections}
        for topic in (config.flow_topic, config.velocity_topic):
            if topic in existing_topics:
                raise RuntimeError(f"Source bag already contains generated topic {topic}")

        connection_map: dict[int, Any] = {}
        for connection in reader.connections:
            connection_map[connection.id] = _add_connection_compat(
                writer,
                topic=connection.topic,
                msgtype=connection.msgtype,
                typestore=typestore,
                offered_qos_profiles=getattr(
                    getattr(connection, "ext", None),
                    "offered_qos_profiles",
                    None,
                ),
            )
        flow_connection = _add_connection_compat(
            writer,
            topic=config.flow_topic,
            msgtype="std_msgs/msg/Float64MultiArray",
            typestore=typestore,
        )
        velocity_connection = _add_connection_compat(
            writer,
            topic=config.velocity_topic,
            msgtype="geometry_msgs/msg/TwistStamped",
            typestore=typestore,
        )

        def write_next_flow() -> None:
            nonlocal flow_index, velocity_index, flow_count, velocity_count
            timestamp_ns, values = flow_events[flow_index]
            message = make_flow_message(values)
            writer.write(
                flow_connection,
                timestamp_ns,
                typestore.serialize_cdr(message, "std_msgs/msg/Float64MultiArray"),
            )
            flow_index += 1
            flow_count += 1

        def write_next_velocity() -> None:
            nonlocal flow_index, velocity_index, flow_count, velocity_count
            timestamp_ns, vx, vy = velocity_events[velocity_index]
            message = make_velocity_message(timestamp_ns, vx, vy)
            writer.write(
                velocity_connection,
                timestamp_ns,
                typestore.serialize_cdr(message, "geometry_msgs/msg/TwistStamped"),
            )
            velocity_index += 1
            velocity_count += 1

        def write_generated_before_raw(limit_ns: int) -> None:
            """Reproduce the legacy two-pass ordering around an original row."""

            while True:
                next_flow = (
                    flow_events[flow_index][0]
                    if flow_index < len(flow_events)
                    and flow_events[flow_index][0] < limit_ns
                    else None
                )
                next_velocity = (
                    velocity_events[velocity_index][0]
                    if velocity_index < len(velocity_events)
                    and velocity_events[velocity_index][0] <= limit_ns
                    else None
                )
                if next_flow is None and next_velocity is None:
                    break
                if next_velocity is not None and (
                    next_flow is None or next_velocity <= next_flow
                ):
                    write_next_velocity()
                else:
                    write_next_flow()

        def write_generated_tail() -> None:
            while flow_index < len(flow_events) or velocity_index < len(velocity_events):
                next_flow = (
                    flow_events[flow_index][0]
                    if flow_index < len(flow_events)
                    else None
                )
                next_velocity = (
                    velocity_events[velocity_index][0]
                    if velocity_index < len(velocity_events)
                    else None
                )
                if next_velocity is not None and (
                    next_flow is None or next_velocity <= next_flow
                ):
                    write_next_velocity()
                else:
                    write_next_flow()

        for connection, timestamp_ns, raw in reader.messages():
            timestamp_ns = int(timestamp_ns)
            if copy_start_ns is not None and timestamp_ns < copy_start_ns:
                continue
            if copy_end_ns is not None and timestamp_ns > copy_end_ns:
                break
            write_generated_before_raw(timestamp_ns)
            writer.write(connection_map[connection.id], timestamp_ns, raw)
            copied_count += 1
            while (
                flow_index < len(flow_events)
                and flow_events[flow_index][0] == timestamp_ns
            ):
                write_next_flow()
        write_generated_tail()

    summary = {
        "stage": "derived_bag",
        "config_fingerprint": _config_fingerprint(config),
        "input_bag_dir": str(config.input_bag_dir),
        "output_bag_dir": str(output_dir),
        "copied_source_messages": copied_count,
        "flow_messages_written": flow_count,
        "velocity_messages_written": velocity_count,
        "copy_window_start_ns": copy_start_ns,
        "copy_window_end_ns": copy_end_ns,
        "flow_topic": config.flow_topic,
        "velocity_topic": config.velocity_topic,
    }
    _write_json(artifacts.bag_summary_json, summary)
    return output_dir


def run_preprocessing(
    config: VisionVelocityConfig,
    *,
    build_bag: bool = True,
    reuse_existing: bool = False,
    overwrite_bag: bool = False,
) -> PreprocessingArtifacts:
    """Run all preprocessing stages with a full bag as the default window."""

    _validate_inputs(config)
    artifacts = _artifacts(config)
    artifacts.root.mkdir(parents=True, exist_ok=True)
    lk_csv = compute_sparse_lk(config, reuse_existing=reuse_existing)
    flow_csv = filter_sparse_lk(
        config, input_csv=lk_csv, reuse_existing=reuse_existing
    )
    _, velocity_csv = compute_gyrocompensated_velocity(
        config,
        filtered_flow_csv=flow_csv,
        reuse_existing=reuse_existing,
    )
    bag_path: Path | None = None
    if build_bag:
        bag_path = build_derived_bag(
            config,
            filtered_flow_csv=flow_csv,
            filtered_velocity_csv=velocity_csv,
            reuse_existing=reuse_existing,
            overwrite=overwrite_bag,
        )

    summary = {
        "stage": "complete_preprocessing_pipeline",
        "config_fingerprint": _config_fingerprint(config),
        "config": config.serializable(),
        "time_window": {
            "start_offset_sec": config.start_offset_sec,
            "duration_sec": config.duration_sec,
            "is_full_bag": config.duration_sec is None and config.start_offset_sec == 0.0,
        },
        "outputs": {
            "lk_raw_csv": str(artifacts.lk_raw_csv),
            "flow_filtered_csv": str(artifacts.flow_filtered_csv),
            "velocity_raw_csv": str(artifacts.velocity_raw_csv),
            "velocity_filtered_csv": str(artifacts.velocity_filtered_csv),
            "derived_bag_dir": str(bag_path) if bag_path is not None else None,
        },
        "legacy_baseline_notes": LEGACY_BASELINE_NOTES,
    }
    _write_json(artifacts.pipeline_summary_json, summary)
    return artifacts
