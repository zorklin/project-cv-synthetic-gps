# Local runtime layout

The notebook was originally written for Google Colab. Local provisioning is
kept outside the notebook because a notebook cannot install or start the WSL
kernel that is required to run its own cells.

The setup is split into three layers:

1. `windows/enable_wsl_features.ps1` enables the Windows WSL prerequisites.
2. `ubuntu/create_project_user.sh` creates the local WSL development user.
3. `ubuntu/bootstrap_ros2_humble.sh` installs ROS 2 Humble, the Python virtual
   environment, Jupyter kernel, and pinned project dependencies.

The setup scripts are intended to be rerunnable. The notebook itself should
only validate the environment and should never install OS packages, download
multi-gigabyte bags, or delete data directories.

The Linux development user has no administrative privileges. The bootstrap is
run explicitly as WSL `root` for system packages, while the virtual environment
and Jupyter kernel are created under the unprivileged user's home directory.

Target runtime:

- WSL 2
- Ubuntu 22.04 (Jammy)
- Python 3.10
- ROS 2 Humble
- `rosbag2_py` with MCAP storage
- NumPy 1.26.4
- GTSAM 4.2 with `gtsam_unstable`

## Cursor connection

Cursor opened on Windows cannot directly discover a kernelspec installed inside
WSL. Start the Linux Jupyter server from PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\environment\windows\jupyter.ps1 start
```

Copy the displayed local URL, choose `Change Kernel` -> `Select Another
Kernel` -> `Existing Jupyter Server` in Cursor, and paste the URL. Then select
`Project CV (ROS 2 Humble)`.

The same launcher accepts `status` and `stop` instead of `start`.
