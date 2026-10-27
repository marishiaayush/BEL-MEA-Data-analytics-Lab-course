#!/usr/bin/env bash
# One-time setup for the HD-MEA practical (macOS / Linux).
set -e
cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv ..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "Building the environment ..."
uv sync

echo
echo "Done!  Launch the notebooks with:"
echo "    uv run jupyter lab notebooks"
