#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
USER_BIN="${HOME}/.local/bin"
mkdir -p "${USER_BIN}"
ln -sfn "${PROJECT_ROOT}/bin/vidpipe" "${USER_BIN}/vidpipe"

if [[ ":${PATH}:" != *":${USER_BIN}:"* ]]; then
    echo "Added ${USER_BIN}/vidpipe, but ${USER_BIN} is not in PATH. Add this to ~/.bashrc:"
    echo 'export PATH="$HOME/.local/bin:$PATH"'
else
    echo "Installed command: ${USER_BIN}/vidpipe"
fi

echo "Use it from an activated environment, then run: vidpipe --help"
