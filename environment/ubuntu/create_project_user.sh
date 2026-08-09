#!/usr/bin/env bash

set -Eeuo pipefail

if [[ "${EUID}" -ne 0 ]]; then
    echo "Run this script as root." >&2
    exit 1
fi

user_name="${1:-fedor}"

if ! id "${user_name}" >/dev/null 2>&1; then
    useradd --create-home --shell /bin/bash "${user_name}"
fi

cat > /etc/wsl.conf <<EOF
[user]
default=${user_name}
EOF

echo "Non-administrative WSL user '${user_name}' is ready and configured as the default user."
