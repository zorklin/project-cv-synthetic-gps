#!/usr/bin/env bash

set -Eeuo pipefail

if [[ "${EUID}" -eq 0 ]]; then
    echo "Run this script as the regular WSL project user, not root." >&2
    exit 1
fi

if [[ "$#" -ne 1 ]]; then
    echo "Usage: $0 /mnt/c/path/to/project_cv" >&2
    exit 1
fi

source_root="$(realpath "$1")"
runtime_root="${PROJECT_CV_RUNTIME:-${HOME}/project_cv_runtime}"

raw_source="${source_root}/data/01_raw_k2r"
derived_source="${source_root}/data/02_derived_with_velocity"
calibration_source="${source_root}/calibration"

raw_runtime="${runtime_root}/bags/raw_k2r"
derived_runtime="${runtime_root}/bags/derived_velocity"
calibration_runtime="${runtime_root}/calibration"
artifacts_runtime="${runtime_root}/artifacts"

required_files=(
    "${raw_source}/K2R00005_20260607_194949_0.mcap"
    "${raw_source}/metadata.yaml"
    "${derived_source}/K2R00005_20260607_194949_with_velocity.mcap"
    "${derived_source}/metadata_velocity.yaml"
    "${calibration_source}/thermal_5_9x7_30mm_384x288_20260123_102051.yml"
)

for path in "${required_files[@]}"; do
    if [[ ! -f "${path}" ]]; then
        echo "Required file is missing: ${path}" >&2
        exit 1
    fi
done

mapfile -d '' rotation_files < <(
    find "${calibration_source}" -maxdepth 1 -type f \
        -name '*am_to_imu_rot_mtrx.yml' -print0
)
if [[ "${#rotation_files[@]}" -ne 1 ]]; then
    echo "Expected exactly one camera-to-IMU rotation file, found ${#rotation_files[@]}." >&2
    exit 1
fi
rotation_source="${rotation_files[0]}"

mkdir -p \
    "${raw_runtime}" \
    "${derived_runtime}" \
    "${calibration_runtime}" \
    "${artifacts_runtime}"

ln -sfn \
    "${raw_source}/K2R00005_20260607_194949_0.mcap" \
    "${raw_runtime}/K2R00005_20260607_194949_0.mcap"
ln -sfn \
    "${raw_source}/metadata.yaml" \
    "${raw_runtime}/metadata.yaml"

ln -sfn \
    "${derived_source}/K2R00005_20260607_194949_with_velocity.mcap" \
    "${derived_runtime}/K2R00005_20260607_194949_with_velocity.mcap"
install -m 0644 \
    "${derived_source}/metadata_velocity.yaml" \
    "${derived_runtime}/metadata.yaml"
install -m 0644 \
    "${derived_source}/metadata_velocity.yaml" \
    "${derived_runtime}/metadata_velocity.yaml"

install -m 0644 \
    "${calibration_source}/thermal_5_9x7_30mm_384x288_20260123_102051.yml" \
    "${calibration_runtime}/thermal_camera.yml"
install -m 0644 \
    "${rotation_source}" \
    "${calibration_runtime}/cam_to_imu_rot_mtrx.yml"

cat > "${runtime_root}/paths.env" <<EOF
export PROJECT_CV_SOURCE='${source_root}'
export PROJECT_CV_RUNTIME='${runtime_root}'
export PROJECT_CV_RAW_BAG='${raw_runtime}'
export PROJECT_CV_DERIVED_BAG='${derived_runtime}'
export PROJECT_CV_CALIBRATION='${calibration_runtime}'
export PROJECT_CV_ARTIFACTS='${artifacts_runtime}'
EOF

echo "Runtime prepared at ${runtime_root}"
echo "Raw bag: ${raw_runtime}"
echo "Derived bag: ${derived_runtime}"
echo "Artifacts: ${artifacts_runtime}"
