#!/usr/bin/env bash

set -Eeuo pipefail

if [[ "${EUID}" -ne 0 ]]; then
    echo "Run this script as root through wsl.exe -u root." >&2
    exit 1
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
requirements_file="${script_dir}/requirements.txt"
project_user="${PROJECT_USER:-fedor}"
project_home="$(getent passwd "${project_user}" | cut -d: -f6)"
venv_dir="${project_home}/.venvs/project_cv"
ros_setup="/opt/ros/humble/setup.bash"
kernel_wrapper="${project_home}/.local/bin/project-cv-kernel"
kernelspec_dir="${project_home}/.local/share/jupyter/kernels/project-cv-ros-humble"

if [[ -z "${project_home}" ]]; then
    echo "Project user '${project_user}' does not exist." >&2
    exit 1
fi

run_as_project_user() {
    runuser --user "${project_user}" -- env HOME="${project_home}" "$@"
}

source /etc/os-release
if [[ "${ID}" != "ubuntu" || "${VERSION_CODENAME}" != "jammy" ]]; then
    echo "Expected Ubuntu 22.04 Jammy, got ${PRETTY_NAME}." >&2
    exit 1
fi

apt-get update
apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    locales \
    software-properties-common

add-apt-repository universe -y

install -d -m 0755 /usr/share/keyrings
curl -fsSL \
    https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg

architecture="$(dpkg --print-architecture)"
printf 'deb [arch=%s signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu %s main\n' \
    "${architecture}" "${VERSION_CODENAME}" \
    > /etc/apt/sources.list.d/ros2.list

apt-get update
apt-get install -y \
    python3-dev \
    python3-numpy \
    python3-opencv \
    python3-pip \
    python3-venv \
    ros-humble-geometry-msgs \
    ros-humble-ros-base \
    ros-humble-rosbag2-py \
    ros-humble-rosbag2-storage-mcap \
    ros-humble-rosbag2-transport \
    ros-humble-sensor-msgs \
    ros-humble-std-msgs

run_as_project_user python3 -m venv --system-site-packages "${venv_dir}"
run_as_project_user "${venv_dir}/bin/python" -m pip install --upgrade pip setuptools wheel
run_as_project_user "${venv_dir}/bin/python" -m pip install --requirement "${requirements_file}"
run_as_project_user "${venv_dir}/bin/python" -m pip install --no-deps 'gtsam==4.2'

run_as_project_user "${venv_dir}/bin/python" -m ipykernel install --user \
    --name project-cv-ros-humble \
    --display-name 'Project CV (ROS 2 Humble)'

install -d -o "${project_user}" -g "${project_user}" \
    "$(dirname "${kernel_wrapper}")" "${kernelspec_dir}"

cat > "${kernel_wrapper}" <<EOF
#!/usr/bin/env bash
set -e
source ${ros_setup}
if [[ -f ${project_home}/project_cv_runtime/paths.env ]]; then
    source ${project_home}/project_cv_runtime/paths.env
fi
exec ${venv_dir}/bin/python -Xfrozen_modules=off -m ipykernel_launcher "\$@"
EOF
chmod 0755 "${kernel_wrapper}"
chown "${project_user}:${project_user}" "${kernel_wrapper}"

cat > "${kernelspec_dir}/kernel.json" <<EOF
{
  "argv": [
    "${kernel_wrapper}",
    "-f",
    "{connection_file}"
  ],
  "display_name": "Project CV (ROS 2 Humble)",
  "language": "python",
  "metadata": {
    "debugger": true
  }
}
EOF
chown "${project_user}:${project_user}" "${kernelspec_dir}/kernel.json"

ros_source_line="source ${ros_setup}"
grep -Fqx "${ros_source_line}" "${project_home}/.bashrc" \
    || printf '\n%s\n' "${ros_source_line}" >> "${project_home}/.bashrc"
chown "${project_user}:${project_user}" "${project_home}/.bashrc"

run_as_project_user bash -c \
    "source '${ros_setup}' && exec '${venv_dir}/bin/python' '${script_dir}/verify_environment.py'"

echo
echo "Environment is ready."
echo "Python: ${venv_dir}/bin/python"
echo "Kernel: Project CV (ROS 2 Humble)"
