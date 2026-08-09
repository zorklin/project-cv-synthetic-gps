"""Read-only validation of a generated vision-velocity bag against a golden bag.

The validator intentionally has a narrow contract: it compares only
``/vision/lk_flow_px_filtered`` and ``/vision/velocity_frd``.  Both bags are
opened with :mod:`rosbags` and the ROS 2 Humble typestore.  No source bag is
ever opened for writing.

The single public entry point, :func:`validate_preprocessing_bags`, returns the
complete diagnostics dictionary and writes the same value to one explicitly
supplied JSON path below the supplied artifacts root.  Numeric comparisons are
streaming/online, so memory use does not grow with the number of messages.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from itertools import zip_longest
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence


FLOW_TOPIC = "/vision/lk_flow_px_filtered"
VELOCITY_TOPIC = "/vision/velocity_frd"

FLOW_MESSAGE_TYPE = "std_msgs/msg/Float64MultiArray"
VELOCITY_MESSAGE_TYPE = "geometry_msgs/msg/TwistStamped"

FLOW_FIELDS: tuple[str, ...] = (
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

VELOCITY_FIELDS: tuple[str, ...] = (
    "header_stamp_ns",
    "linear.x",
    "linear.y",
    "linear.z",
    "angular.x",
    "angular.y",
    "angular.z",
)

_FLOW_BINARY_FIELD_INDICES: tuple[int, ...] = (11, 12, 14, 15, 16, 17)
_FLOW_T_SEC_ROUNDTRIP_TOLERANCE_NS = 1_024
_MISSING = object()


def _require_rosbags() -> tuple[Any, Any]:
    try:
        from rosbags.highlevel import AnyReader
        from rosbags.typesys import Stores, get_typestore
    except ImportError as exc:  # pragma: no cover - depends on runtime setup.
        raise RuntimeError(
            "rosbags is unavailable. Run this validator in the Project CV "
            "ROS 2 Humble kernel."
        ) from exc
    return AnyReader, get_typestore(Stores.ROS2_HUMBLE)


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _is_below(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return path != directory


def _validate_paths(
    generated_bag_dir: Path,
    golden_bag_dir: Path,
    artifacts_root: Path,
    diagnostics_path: Path,
) -> None:
    for label, bag_dir in (
        ("generated", generated_bag_dir),
        ("golden", golden_bag_dir),
    ):
        if not bag_dir.is_dir():
            raise FileNotFoundError(f"{label} bag directory does not exist: {bag_dir}")
        if not (bag_dir / "metadata.yaml").is_file():
            raise FileNotFoundError(
                f"{label} bag has no metadata.yaml: {bag_dir / 'metadata.yaml'}"
            )

    if generated_bag_dir == golden_bag_dir:
        raise ValueError("generated_bag_dir and golden_bag_dir must be different")
    if not artifacts_root.is_dir():
        raise FileNotFoundError(f"Artifacts root does not exist: {artifacts_root}")
    if diagnostics_path.suffix.lower() != ".json":
        raise ValueError("diagnostics_path must name a .json file")
    if not _is_below(diagnostics_path, artifacts_root):
        raise ValueError("diagnostics_path must be below artifacts_root")
    if diagnostics_path.is_dir():
        raise ValueError(f"diagnostics_path is a directory: {diagnostics_path}")
    for bag_dir in (generated_bag_dir, golden_bag_dir):
        if _is_below(diagnostics_path, bag_dir):
            raise ValueError(
                "diagnostics_path must not be inside either bag directory; "
                "bags are read-only inputs"
            )


@dataclass
class _DifferenceStats:
    """Online signed and absolute differences with integer-safe accumulation."""

    tolerance: float
    count: int = 0
    exact_match_count: int = 0
    within_tolerance_count: int = 0
    minimum: float | int | None = None
    maximum: float | int | None = None
    maximum_absolute: float | int | None = None
    sum_difference: float | int = 0
    sum_absolute: float = 0.0
    sum_squared: float = 0.0

    def update(self, left: float | int, right: float | int) -> None:
        difference = left - right
        absolute = abs(difference)
        self.count += 1
        self.sum_difference += difference
        self.sum_absolute += float(absolute)
        self.sum_squared += float(difference) * float(difference)
        self.minimum = difference if self.minimum is None else min(self.minimum, difference)
        self.maximum = difference if self.maximum is None else max(self.maximum, difference)
        self.maximum_absolute = (
            absolute
            if self.maximum_absolute is None
            else max(self.maximum_absolute, absolute)
        )
        if difference == 0:
            self.exact_match_count += 1
        if absolute <= self.tolerance:
            self.within_tolerance_count += 1

    def result(self, *, units: str, convention: str) -> dict[str, Any]:
        return {
            "units": units,
            "signed_difference_convention": convention,
            "tolerance": self.tolerance,
            "count": self.count,
            "exact_match_count": self.exact_match_count,
            "within_tolerance_count": self.within_tolerance_count,
            "all_within_tolerance": (
                self.count > 0 and self.within_tolerance_count == self.count
            ),
            "minimum_signed_difference": self.minimum,
            "maximum_signed_difference": self.maximum,
            "maximum_absolute_difference": self.maximum_absolute,
            "mean_signed_difference": (
                float(self.sum_difference) / self.count if self.count else None
            ),
            "mean_absolute_difference": (
                self.sum_absolute / self.count if self.count else None
            ),
            "root_mean_square_difference": (
                math.sqrt(self.sum_squared / self.count) if self.count else None
            ),
        }


@dataclass
class _AbsoluteErrorStats:
    """Online numeric error summary that handles NaN and infinities explicitly."""

    tolerance: float
    pair_count: int = 0
    finite_pair_count: int = 0
    nonfinite_equal_count: int = 0
    nonfinite_mismatch_count: int = 0
    exact_match_count: int = 0
    within_tolerance_count: int = 0
    minimum_absolute_error: float | None = None
    maximum_absolute_error: float | None = None
    sum_absolute_error: float = 0.0
    sum_squared_error: float = 0.0

    def update(self, generated: float | int, golden: float | int) -> None:
        self.pair_count += 1
        if isinstance(generated, int) and isinstance(golden, int):
            # Keep epoch nanoseconds exact; float64 would hide sub-256 ns deltas.
            error = abs(generated - golden)
            self.finite_pair_count += 1
            self.sum_absolute_error += float(error)
            self.sum_squared_error += float(error) * float(error)
            self.minimum_absolute_error = (
                float(error)
                if self.minimum_absolute_error is None
                else min(self.minimum_absolute_error, float(error))
            )
            self.maximum_absolute_error = (
                float(error)
                if self.maximum_absolute_error is None
                else max(self.maximum_absolute_error, float(error))
            )
            if generated == golden:
                self.exact_match_count += 1
            if error <= self.tolerance:
                self.within_tolerance_count += 1
            return

        generated_value = float(generated)
        golden_value = float(golden)
        if math.isfinite(generated_value) and math.isfinite(golden_value):
            error = abs(generated_value - golden_value)
            self.finite_pair_count += 1
            self.sum_absolute_error += error
            self.sum_squared_error += error * error
            self.minimum_absolute_error = (
                error
                if self.minimum_absolute_error is None
                else min(self.minimum_absolute_error, error)
            )
            self.maximum_absolute_error = (
                error
                if self.maximum_absolute_error is None
                else max(self.maximum_absolute_error, error)
            )
            if generated_value == golden_value:
                self.exact_match_count += 1
            if error <= self.tolerance:
                self.within_tolerance_count += 1
            return

        equal_nonfinite = (
            (math.isnan(generated_value) and math.isnan(golden_value))
            or generated_value == golden_value
        )
        if equal_nonfinite:
            self.nonfinite_equal_count += 1
            self.exact_match_count += 1
            self.within_tolerance_count += 1
        else:
            self.nonfinite_mismatch_count += 1

    def result(self) -> dict[str, Any]:
        return {
            "tolerance": self.tolerance,
            "pair_count": self.pair_count,
            "finite_pair_count": self.finite_pair_count,
            "nonfinite_equal_count": self.nonfinite_equal_count,
            "nonfinite_mismatch_count": self.nonfinite_mismatch_count,
            "exact_match_count": self.exact_match_count,
            "within_tolerance_count": self.within_tolerance_count,
            "all_within_tolerance": (
                self.pair_count > 0
                and self.within_tolerance_count == self.pair_count
            ),
            "minimum_absolute_error": self.minimum_absolute_error,
            "maximum_absolute_error": self.maximum_absolute_error,
            "mean_absolute_error": (
                self.sum_absolute_error / self.finite_pair_count
                if self.finite_pair_count
                else None
            ),
            "root_mean_square_error": (
                math.sqrt(self.sum_squared_error / self.finite_pair_count)
                if self.finite_pair_count
                else None
            ),
        }


@dataclass
class _ZeroInvariantStats:
    tolerance: float
    count: int = 0
    nonfinite_count: int = 0
    outside_tolerance_count: int = 0
    maximum_absolute_value: float | None = None

    def update(self, value: float | int) -> None:
        self.count += 1
        number = float(value)
        if not math.isfinite(number):
            self.nonfinite_count += 1
            return
        absolute = abs(number)
        self.maximum_absolute_value = (
            absolute
            if self.maximum_absolute_value is None
            else max(self.maximum_absolute_value, absolute)
        )
        if absolute > self.tolerance:
            self.outside_tolerance_count += 1

    def result(self) -> dict[str, Any]:
        return {
            "tolerance": self.tolerance,
            "count": self.count,
            "nonfinite_count": self.nonfinite_count,
            "outside_tolerance_count": self.outside_tolerance_count,
            "maximum_absolute_value": self.maximum_absolute_value,
            "valid": (
                self.count > 0
                and self.nonfinite_count == 0
                and self.outside_tolerance_count == 0
            ),
        }


@dataclass(frozen=True)
class _DecodedRecord:
    bag_timestamp_ns: int
    numeric_values: tuple[float | int, ...]
    frame_id: str | None = None


@dataclass
class _TopicInspector:
    topic: str
    expected_message_type: str
    expected_frame_id: str
    zero_tolerance: float
    timestamp_tolerance_ns: int
    streamed_message_count: int = 0
    expected_type_message_count: int = 0
    unexpected_type_message_count: int = 0
    deserialize_error_count: int = 0
    schema_error_count: int = 0
    first_bag_timestamp_ns: int | None = None
    last_bag_timestamp_ns: int | None = None
    timestamp_decrease_count: int = 0
    duplicate_timestamp_count: int = 0
    _previous_timestamp_ns: int | None = None

    # Flow-only diagnostics.
    flow_data_length_counts: Counter[int] = field(default_factory=Counter)
    flow_layout_violations: Counter[str] = field(default_factory=Counter)
    flow_semantic_violations: Counter[str] = field(default_factory=Counter)
    flow_nonfinite_by_field: Counter[str] = field(default_factory=Counter)
    flow_embedded_stamp_vs_bag: _DifferenceStats = field(init=False)
    flow_seconds_stamp_vs_bag: _DifferenceStats = field(init=False)

    # Velocity-only diagnostics.
    frame_id_counts: Counter[str] = field(default_factory=Counter)
    unexpected_frame_id_count: int = 0
    velocity_nonfinite_by_field: Counter[str] = field(default_factory=Counter)
    velocity_header_stamp_violations: Counter[str] = field(default_factory=Counter)
    velocity_header_vs_bag: _DifferenceStats = field(init=False)
    velocity_zero_fields: dict[str, _ZeroInvariantStats] = field(init=False)

    def __post_init__(self) -> None:
        self.flow_embedded_stamp_vs_bag = _DifferenceStats(
            float(self.timestamp_tolerance_ns)
        )
        self.flow_seconds_stamp_vs_bag = _DifferenceStats(
            float(
                max(
                    self.timestamp_tolerance_ns,
                    _FLOW_T_SEC_ROUNDTRIP_TOLERANCE_NS,
                )
            )
        )
        self.velocity_header_vs_bag = _DifferenceStats(
            float(self.timestamp_tolerance_ns)
        )
        self.velocity_zero_fields = {
            name: _ZeroInvariantStats(self.zero_tolerance)
            for name in (
                "linear.z",
                "angular.x",
                "angular.y",
                "angular.z",
            )
        }

    def consume(
        self,
        reader: Any,
        connection: Any,
        timestamp_ns: int,
        raw: bytes,
    ) -> _DecodedRecord | None:
        timestamp_ns = int(timestamp_ns)
        self.streamed_message_count += 1
        if self.first_bag_timestamp_ns is None:
            self.first_bag_timestamp_ns = timestamp_ns
        self.last_bag_timestamp_ns = timestamp_ns
        if self._previous_timestamp_ns is not None:
            if timestamp_ns < self._previous_timestamp_ns:
                self.timestamp_decrease_count += 1
            elif timestamp_ns == self._previous_timestamp_ns:
                self.duplicate_timestamp_count += 1
        self._previous_timestamp_ns = timestamp_ns

        if connection.msgtype != self.expected_message_type:
            self.unexpected_type_message_count += 1
            return None
        self.expected_type_message_count += 1
        try:
            message = reader.deserialize(raw, connection.msgtype)
        except Exception:  # noqa: BLE001 - malformed bag data is a diagnostic.
            self.deserialize_error_count += 1
            return None

        try:
            if self.topic == FLOW_TOPIC:
                return self._inspect_flow(message, timestamp_ns)
            return self._inspect_velocity(message, timestamp_ns)
        except (AttributeError, IndexError, TypeError, ValueError, OverflowError):
            self.schema_error_count += 1
            return None

    def _inspect_flow(self, message: Any, timestamp_ns: int) -> _DecodedRecord | None:
        data = tuple(float(value) for value in message.data)
        self.flow_data_length_counts[len(data)] += 1

        dimensions = tuple(message.layout.dim)
        if len(dimensions) != 1:
            self.flow_layout_violations["dimension_count_not_one"] += 1
        else:
            dimension = dimensions[0]
            if str(dimension.label) != "lk_flow_px_filtered":
                self.flow_layout_violations["unexpected_dimension_label"] += 1
            if int(dimension.size) != len(FLOW_FIELDS):
                self.flow_layout_violations["unexpected_dimension_size"] += 1
            if int(dimension.stride) != len(FLOW_FIELDS):
                self.flow_layout_violations["unexpected_dimension_stride"] += 1
        if int(message.layout.data_offset) != 0:
            self.flow_layout_violations["nonzero_data_offset"] += 1
        if len(data) != len(FLOW_FIELDS):
            self.flow_layout_violations["unexpected_data_length"] += 1
            return None

        for index, (name, value) in enumerate(zip(FLOW_FIELDS, data)):
            if not math.isfinite(value):
                self.flow_nonfinite_by_field[name] += 1
                continue
            if index in _FLOW_BINARY_FIELD_INDICES and value not in (0.0, 1.0):
                self.flow_semantic_violations[f"{name}_not_binary"] += 1

        if math.isfinite(data[0]) and (
            data[0] < 0.0 or abs(data[0] - round(data[0])) > 1e-6
        ):
            self.flow_semantic_violations["frame_idx_not_nonnegative_integer"] += 1
        if math.isfinite(data[1]) and data[1] < 0.0:
            self.flow_semantic_violations["stamp_ns_negative"] += 1
        if math.isfinite(data[2]) and data[2] < 0.0:
            self.flow_semantic_violations["t_sec_negative"] += 1
        if math.isfinite(data[3]) and data[3] < 0.0:
            self.flow_semantic_violations["dt_s_negative"] += 1
        if math.isfinite(data[8]) and data[8] < 0.0:
            self.flow_semantic_violations["flow_mag_filt_negative"] += 1
        if math.isfinite(data[9]) and not 0.0 <= data[9] <= 1.0:
            self.flow_semantic_violations["quality_ratio_outside_0_1"] += 1
        if math.isfinite(data[10]) and (
            data[10] < 0.0 or abs(data[10] - round(data[10])) > 1e-6
        ):
            self.flow_semantic_violations["n_good_tracks_not_nonnegative_integer"] += 1
        if math.isfinite(data[1]):
            self.flow_embedded_stamp_vs_bag.update(int(data[1]), timestamp_ns)
        if math.isfinite(data[2]):
            self.flow_seconds_stamp_vs_bag.update(
                int(round(data[2] * 1_000_000_000.0)), timestamp_ns
            )
        return _DecodedRecord(timestamp_ns, data)

    def _inspect_velocity(
        self, message: Any, timestamp_ns: int
    ) -> _DecodedRecord:
        stamp = message.header.stamp
        seconds = int(stamp.sec)
        nanoseconds = int(stamp.nanosec)
        if seconds < 0:
            self.velocity_header_stamp_violations["negative_seconds"] += 1
        if not 0 <= nanoseconds < 1_000_000_000:
            self.velocity_header_stamp_violations["nanoseconds_out_of_range"] += 1
        header_stamp_ns = seconds * 1_000_000_000 + nanoseconds
        self.velocity_header_vs_bag.update(header_stamp_ns, timestamp_ns)

        frame_id = str(message.header.frame_id)
        self.frame_id_counts[frame_id] += 1
        if frame_id != self.expected_frame_id:
            self.unexpected_frame_id_count += 1

        values: tuple[float | int, ...] = (
            header_stamp_ns,
            float(message.twist.linear.x),
            float(message.twist.linear.y),
            float(message.twist.linear.z),
            float(message.twist.angular.x),
            float(message.twist.angular.y),
            float(message.twist.angular.z),
        )
        for name, value in zip(VELOCITY_FIELDS[1:], values[1:]):
            if not math.isfinite(float(value)):
                self.velocity_nonfinite_by_field[name] += 1
        for name, value in zip(VELOCITY_FIELDS[3:], values[3:]):
            self.velocity_zero_fields[name].update(value)
        return _DecodedRecord(timestamp_ns, values, frame_id)

    def common_result(self) -> dict[str, Any]:
        return {
            "streamed_message_count": self.streamed_message_count,
            "expected_type_message_count": self.expected_type_message_count,
            "unexpected_type_message_count": self.unexpected_type_message_count,
            "deserialize_error_count": self.deserialize_error_count,
            "schema_error_count": self.schema_error_count,
            "first_bag_timestamp_ns": self.first_bag_timestamp_ns,
            "last_bag_timestamp_ns": self.last_bag_timestamp_ns,
            "timestamp_decrease_count": self.timestamp_decrease_count,
            "duplicate_timestamp_count": self.duplicate_timestamp_count,
            "timestamps_nondecreasing": self.timestamp_decrease_count == 0,
        }

    def flow_result(self) -> dict[str, Any]:
        layout_violations = dict(sorted(self.flow_layout_violations.items()))
        semantic_violations = dict(sorted(self.flow_semantic_violations.items()))
        nonfinite = dict(sorted(self.flow_nonfinite_by_field.items()))
        valid = (
            self.streamed_message_count > 0
            and self.unexpected_type_message_count == 0
            and self.deserialize_error_count == 0
            and self.schema_error_count == 0
            and not layout_violations
            and not semantic_violations
            and not nonfinite
            and self.timestamp_decrease_count == 0
            and self.flow_embedded_stamp_vs_bag.result(
                units='ns', convention='data[stamp_ns] - bag timestamp'
            )['all_within_tolerance']
        )
        return {
            **self.common_result(),
            "data_length_counts": {
                str(key): value
                for key, value in sorted(self.flow_data_length_counts.items())
            },
            "expected_data_length": len(FLOW_FIELDS),
            "field_order": list(FLOW_FIELDS),
            "layout_violation_counts": layout_violations,
            "semantic_violation_counts": semantic_violations,
            "nonfinite_value_counts_by_field": nonfinite,
            "embedded_stamp_ns_minus_bag_timestamp": (
                self.flow_embedded_stamp_vs_bag.result(
                    units="ns", convention="data[stamp_ns] - bag timestamp"
                )
            ),
            "t_sec_minus_bag_timestamp": self.flow_seconds_stamp_vs_bag.result(
                units="ns", convention="round(data[t_sec] * 1e9) - bag timestamp"
            ),
            "schema_invariants_valid": valid,
        }

    def velocity_result(self) -> dict[str, Any]:
        zero_results = {
            name: stats.result() for name, stats in self.velocity_zero_fields.items()
        }
        valid = (
            self.streamed_message_count > 0
            and self.unexpected_type_message_count == 0
            and self.deserialize_error_count == 0
            and self.schema_error_count == 0
            and self.timestamp_decrease_count == 0
            and self.unexpected_frame_id_count == 0
            and not self.velocity_nonfinite_by_field
            and not self.velocity_header_stamp_violations
            and all(item["valid"] for item in zero_results.values())
            and self.velocity_header_vs_bag.result(
                units="ns", convention="header stamp - bag timestamp"
            )["all_within_tolerance"]
        )
        return {
            **self.common_result(),
            "field_order": list(VELOCITY_FIELDS),
            "expected_frame_id": self.expected_frame_id,
            "frame_id_counts": dict(sorted(self.frame_id_counts.items())),
            "unexpected_frame_id_count": self.unexpected_frame_id_count,
            "header_stamp_violation_counts": dict(
                sorted(self.velocity_header_stamp_violations.items())
            ),
            "nonfinite_value_counts_by_field": dict(
                sorted(self.velocity_nonfinite_by_field.items())
            ),
            "header_stamp_minus_bag_timestamp": self.velocity_header_vs_bag.result(
                units="ns", convention="header stamp - bag timestamp"
            ),
            "zero_field_invariants": zero_results,
            "message_invariants_valid": valid,
        }


def _connection_contract(
    reader: Any, topic: str, expected_message_type: str
) -> tuple[list[Any], dict[str, Any]]:
    connections = [connection for connection in reader.connections if connection.topic == topic]
    message_types = sorted({str(connection.msgtype) for connection in connections})
    metadata_message_count = sum(int(connection.msgcount) for connection in connections)
    return connections, {
        "topic_present": bool(connections),
        "connection_count": len(connections),
        "message_types": message_types,
        "expected_message_type": expected_message_type,
        "message_type_valid": message_types == [expected_message_type],
        "metadata_message_count": metadata_message_count,
    }


def _numeric_comparison(
    field_names: Sequence[str],
    numeric_tolerance: float,
    timestamp_tolerance_ns: int,
) -> dict[str, _AbsoluteErrorStats]:
    return {
        name: _AbsoluteErrorStats(
            float(timestamp_tolerance_ns)
            if name in {"stamp_ns", "header_stamp_ns"}
            else numeric_tolerance
        )
        for name in field_names
    }


def _comparison_reasons(
    generated_contract: dict[str, Any],
    golden_contract: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    if not generated_contract["topic_present"]:
        reasons.append("generated topic is missing")
    if not golden_contract["topic_present"]:
        reasons.append("golden topic is missing")
    if not generated_contract["message_type_valid"]:
        reasons.append("generated topic type differs from the expected type")
    if not golden_contract["message_type_valid"]:
        reasons.append("golden topic type differs from the expected type")
    if (
        generated_contract["metadata_message_count"]
        != golden_contract["metadata_message_count"]
    ):
        reasons.append("generated and golden message counts differ")
    return reasons


def _compare_topic(
    generated_reader: Any,
    golden_reader: Any,
    *,
    topic: str,
    expected_message_type: str,
    field_names: Sequence[str],
    expected_frame_id: str,
    numeric_tolerance: float,
    zero_tolerance: float,
    timestamp_tolerance_ns: int,
) -> dict[str, Any]:
    generated_connections, generated_contract = _connection_contract(
        generated_reader, topic, expected_message_type
    )
    golden_connections, golden_contract = _connection_contract(
        golden_reader, topic, expected_message_type
    )
    reasons = _comparison_reasons(generated_contract, golden_contract)
    comparison_enabled = not reasons

    generated_inspector = _TopicInspector(
        topic,
        expected_message_type,
        expected_frame_id,
        zero_tolerance,
        timestamp_tolerance_ns,
    )
    golden_inspector = _TopicInspector(
        topic,
        expected_message_type,
        expected_frame_id,
        zero_tolerance,
        timestamp_tolerance_ns,
    )
    timestamp_differences = _DifferenceStats(float(timestamp_tolerance_ns))
    per_field = _numeric_comparison(
        field_names, numeric_tolerance, timestamp_tolerance_ns
    )
    paired_message_count = 0
    comparable_message_count = 0
    uncomparable_message_count = 0
    frame_id_mismatch_count = 0

    # In rosbags an empty ``connections`` iterable means all connections.
    # Explicitly use an empty iterator when the target topic is absent.
    generated_messages: Iterable[Any] = (
        generated_reader.messages(connections=generated_connections)
        if generated_connections
        else ()
    )
    golden_messages: Iterable[Any] = (
        golden_reader.messages(connections=golden_connections)
        if golden_connections
        else ()
    )
    for generated_item, golden_item in zip_longest(
        generated_messages, golden_messages, fillvalue=_MISSING
    ):
        generated_record: _DecodedRecord | None = None
        golden_record: _DecodedRecord | None = None
        if generated_item is not _MISSING:
            connection, timestamp_ns, raw = generated_item
            generated_record = generated_inspector.consume(
                generated_reader, connection, timestamp_ns, raw
            )
        if golden_item is not _MISSING:
            connection, timestamp_ns, raw = golden_item
            golden_record = golden_inspector.consume(
                golden_reader, connection, timestamp_ns, raw
            )

        if not comparison_enabled:
            continue
        if generated_item is _MISSING or golden_item is _MISSING:
            # Metadata said counts match, but the streamed contents disagree.
            uncomparable_message_count += 1
            continue
        paired_message_count += 1
        generated_timestamp = int(generated_item[1])
        golden_timestamp = int(golden_item[1])
        timestamp_differences.update(generated_timestamp, golden_timestamp)
        if generated_record is None or golden_record is None:
            uncomparable_message_count += 1
            continue
        if len(generated_record.numeric_values) != len(field_names) or len(
            golden_record.numeric_values
        ) != len(field_names):
            uncomparable_message_count += 1
            continue
        comparable_message_count += 1
        for name, generated_value, golden_value in zip(
            field_names,
            generated_record.numeric_values,
            golden_record.numeric_values,
        ):
            per_field[name].update(generated_value, golden_value)
        if generated_record.frame_id != golden_record.frame_id:
            frame_id_mismatch_count += 1

    generated_contract["streamed_message_count"] = (
        generated_inspector.streamed_message_count
    )
    generated_contract["metadata_count_matches_stream"] = (
        generated_contract["metadata_message_count"]
        == generated_inspector.streamed_message_count
    )
    golden_contract["streamed_message_count"] = golden_inspector.streamed_message_count
    golden_contract["metadata_count_matches_stream"] = (
        golden_contract["metadata_message_count"]
        == golden_inspector.streamed_message_count
    )

    generated_invariants = (
        generated_inspector.flow_result()
        if topic == FLOW_TOPIC
        else generated_inspector.velocity_result()
    )
    golden_invariants = (
        golden_inspector.flow_result()
        if topic == FLOW_TOPIC
        else golden_inspector.velocity_result()
    )
    timestamp_result = timestamp_differences.result(
        units="ns", convention="generated bag timestamp - golden bag timestamp"
    )
    field_results = {name: per_field[name].result() for name in field_names}

    numeric_complete = (
        comparison_enabled
        and paired_message_count == generated_contract["metadata_message_count"]
        and comparable_message_count == paired_message_count
        and uncomparable_message_count == 0
    )
    values_within_tolerance = (
        numeric_complete
        and timestamp_result["all_within_tolerance"]
        and all(result["all_within_tolerance"] for result in field_results.values())
        and frame_id_mismatch_count == 0
    )
    if comparison_enabled and not numeric_complete:
        reasons.append("one or more message pairs could not be compared")

    generated_invariant_key = (
        "schema_invariants_valid" if topic == FLOW_TOPIC else "message_invariants_valid"
    )
    golden_invariant_key = generated_invariant_key
    passed = (
        not reasons
        and generated_contract["metadata_count_matches_stream"]
        and golden_contract["metadata_count_matches_stream"]
        and bool(generated_invariants[generated_invariant_key])
        and bool(golden_invariants[golden_invariant_key])
        and values_within_tolerance
    )

    return {
        "topic": topic,
        "expected_message_type": expected_message_type,
        "generated": {
            "contract": generated_contract,
            "invariants": generated_invariants,
        },
        "golden": {
            "contract": golden_contract,
            "invariants": golden_invariants,
        },
        "comparison": {
            "enabled": comparison_enabled,
            "disabled_or_incomplete_reasons": reasons,
            "count_match": (
                generated_contract["metadata_message_count"]
                == golden_contract["metadata_message_count"]
            ),
            "paired_message_count": paired_message_count,
            "comparable_message_count": comparable_message_count,
            "uncomparable_message_count": uncomparable_message_count,
            "numeric_comparison_complete": numeric_complete,
            "message_order_basis": (
                "chronological bag order; no nearest-neighbor timestamp matching"
            ),
            "bag_timestamp_difference": timestamp_result,
            "numeric_absolute_error_by_field": field_results,
            "frame_id_mismatch_count": (
                frame_id_mismatch_count if topic == VELOCITY_TOPIC else None
            ),
            "all_values_within_tolerance": values_within_tolerance,
        },
        "passed": passed,
    }


def validate_preprocessing_bags(
    generated_bag_dir: str | Path,
    golden_bag_dir: str | Path,
    *,
    artifacts_root: str | Path,
    diagnostics_path: str | Path,
    expected_velocity_frame_id: str = "base_link_frd",
    numeric_tolerance: float = 1e-7,
    zero_tolerance: float = 1e-12,
    timestamp_tolerance_ns: int = 0,
) -> dict[str, Any]:
    """Compare generated preprocessing topics against the teacher golden bag.

    Parameters
    ----------
    generated_bag_dir, golden_bag_dir:
        Existing rosbag2 directories.  They are opened read-only by
        :class:`rosbags.highlevel.AnyReader`.
    artifacts_root:
        Existing root that owns generated diagnostics.  It is supplied
        separately so the output containment check cannot be inferred from an
        untrusted output path.
    diagnostics_path:
        Explicit ``.json`` output path.  It must be below ``artifacts_root``
        and outside both bag directories.
    expected_velocity_frame_id:
        Required ``TwistStamped.header.frame_id``.
    numeric_tolerance:
        Absolute tolerance for floating-point message fields.  The default is
        deliberately above the observed 2.98e-8 round-trip difference in two
        legacy outlier-score values, while remaining negligible relative to
        pixel-flow and velocity measurement scales.
    zero_tolerance:
        Absolute tolerance for velocity ``linear.z`` and all angular fields.
    timestamp_tolerance_ns:
        Tolerance for all timestamp comparisons, in nanoseconds.

    Returns
    -------
    dict
        The exact diagnostics payload written to ``diagnostics_path``.

    Notes
    -----
    Per-field numeric comparison is intentionally enabled only when topic
    presence, type, and message counts match.  Messages are paired by their
    chronological bag order, not by a nearest-timestamp heuristic.
    """

    if not math.isfinite(numeric_tolerance) or numeric_tolerance < 0.0:
        raise ValueError("numeric_tolerance must be >= 0")
    if not math.isfinite(zero_tolerance) or zero_tolerance < 0.0:
        raise ValueError("zero_tolerance must be >= 0")
    if timestamp_tolerance_ns < 0:
        raise ValueError("timestamp_tolerance_ns must be >= 0")
    if not expected_velocity_frame_id:
        raise ValueError("expected_velocity_frame_id must not be empty")

    generated_path = _resolved(generated_bag_dir)
    golden_path = _resolved(golden_bag_dir)
    artifacts_path = _resolved(artifacts_root)
    output_path = _resolved(diagnostics_path)
    _validate_paths(
        generated_path,
        golden_path,
        artifacts_path,
        output_path,
    )

    AnyReader, typestore = _require_rosbags()
    with AnyReader(
        [generated_path], default_typestore=typestore
    ) as generated_reader, AnyReader(
        [golden_path], default_typestore=typestore
    ) as golden_reader:
        flow = _compare_topic(
            generated_reader,
            golden_reader,
            topic=FLOW_TOPIC,
            expected_message_type=FLOW_MESSAGE_TYPE,
            field_names=FLOW_FIELDS,
            expected_frame_id=expected_velocity_frame_id,
            numeric_tolerance=numeric_tolerance,
            zero_tolerance=zero_tolerance,
            timestamp_tolerance_ns=timestamp_tolerance_ns,
        )
        velocity = _compare_topic(
            generated_reader,
            golden_reader,
            topic=VELOCITY_TOPIC,
            expected_message_type=VELOCITY_MESSAGE_TYPE,
            field_names=VELOCITY_FIELDS,
            expected_frame_id=expected_velocity_frame_id,
            numeric_tolerance=numeric_tolerance,
            zero_tolerance=zero_tolerance,
            timestamp_tolerance_ns=timestamp_tolerance_ns,
        )

    payload: dict[str, Any] = {
        "schema_version": 1,
        "validator": "project_cv.preprocessing_validation.validate_preprocessing_bags",
        "read_only_bag_inputs": True,
        "paths": {
            "generated_bag_dir": str(generated_path),
            "golden_bag_dir": str(golden_path),
            "artifacts_root": str(artifacts_path),
            "diagnostics_path": str(output_path),
        },
        "tolerances": {
            "numeric_absolute": numeric_tolerance,
            "zero_field_absolute": zero_tolerance,
            "timestamp_ns": timestamp_tolerance_ns,
            "flow_t_sec_float64_roundtrip_ns": max(
                timestamp_tolerance_ns,
                _FLOW_T_SEC_ROUNDTRIP_TOLERANCE_NS,
            ),
        },
        "topics": {
            FLOW_TOPIC: flow,
            VELOCITY_TOPIC: velocity,
        },
        "overall": {
            "passed": bool(flow["passed"] and velocity["passed"]),
            "interpretation": (
                "Pass requires matching topic contracts and counts, valid "
                "per-message invariants, and message-order numeric/timestamp "
                "differences within the configured tolerances."
            ),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return payload


__all__ = ["validate_preprocessing_bags"]
