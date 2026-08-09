#!/usr/bin/env bash

set -Eeuo pipefail

action="${1:-start}"
runtime_root="${HOME}/project_cv_runtime"
paths_file="${runtime_root}/paths.env"
python="${HOME}/.venvs/project_cv/bin/python"
pid_file="${runtime_root}/jupyter.pid"
log_file="${runtime_root}/jupyter.log"
token_file="${runtime_root}/jupyter.token"

if [[ ! -f "${paths_file}" ]]; then
    echo "Runtime paths are missing: ${paths_file}" >&2
    exit 1
fi
source "${paths_file}"

if [[ ! -x "${python}" ]]; then
    echo "Project Python is missing: ${python}" >&2
    exit 1
fi

server_is_running() {
    [[ -f "${pid_file}" ]] || return 1
    local pid
    pid="$(cat "${pid_file}")"
    [[ "${pid}" =~ ^[0-9]+$ ]] || return 1
    kill -0 "${pid}" 2>/dev/null || return 1
    tr '\0' ' ' < "/proc/${pid}/cmdline" | grep -Fq 'jupyterlab'
}

show_servers() {
    "${python}" -m jupyter server list
}

case "${action}" in
    start)
        if ! server_is_running; then
            rm -f "${pid_file}" "${log_file}"
            if [[ ! -s "${token_file}" ]]; then
                umask 077
                "${python}" -c 'import secrets; print(secrets.token_hex(32))' \
                    > "${token_file}"
            fi
            token="$(cat "${token_file}")"
            nohup "${python}" -m jupyterlab \
                --no-browser \
                --ServerApp.ip=127.0.0.1 \
                --ServerApp.port=8888 \
                --ServerApp.port_retries=10 \
                --ServerApp.allow_remote_access=False \
                --ServerApp.root_dir="${PROJECT_CV_SOURCE}" \
                --IdentityProvider.token="${token}" \
                >"${log_file}" 2>&1 </dev/null &
            echo "$!" > "${pid_file}"
        fi

        for _ in $(seq 1 30); do
            if show_servers 2>/dev/null | grep -Fq 'http://127.0.0.1:'; then
                break
            fi
            sleep 0.5
        done

        if ! server_is_running; then
            echo "Jupyter failed to start. Log:" >&2
            tail -n 50 "${log_file}" >&2
            exit 1
        fi

        echo "Jupyter server is running. Use this URL in Cursor:"
        show_servers
        ;;
    status)
        if server_is_running; then
            echo "Jupyter server is running."
            show_servers
        else
            echo "Jupyter server is stopped."
            exit 1
        fi
        ;;
    stop)
        if server_is_running; then
            pid="$(cat "${pid_file}")"
            kill "${pid}"
            for _ in $(seq 1 20); do
                kill -0 "${pid}" 2>/dev/null || break
                sleep 0.25
            done
            echo "Jupyter server stopped."
        else
            echo "Jupyter server was not running."
        fi
        rm -f "${pid_file}"
        ;;
    *)
        echo "Usage: $0 {start|status|stop}" >&2
        exit 2
        ;;
esac
