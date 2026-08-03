#!/usr/bin/env bash
# Rebuild this whole setup from a bare Prime Intellect CPU sandbox.
# The box is ephemeral (§0 of RUNBOOK.md) — this script is the recovery path.
#
#   bash setup_sandbox.sh        # ~5 min on 16 cores
set -euo pipefail

ENGINE_REPO=${ENGINE_REPO:-/root/NetHack-engine}
HUB_REPO=${HUB_REPO:-/root/NetHack-hub}
HUB_BRANCH=${HUB_BRANCH:-exp/cli-harness-eval}

echo "==> system build deps"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq \
  bison flex build-essential cmake pkg-config \
  libncurses-dev libbz2-dev zlib1g-dev \
  bubblewrap curl git

echo "==> uv + python 3.12"
if ! command -v uv >/dev/null; then curl -LsSf https://astral.sh/uv/install.sh | sh; fi
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.12

echo "==> hub on the pinned branch (main is missing the harness repairs)"
cd "$HUB_REPO"
git checkout "$HUB_BRANCH" 2>/dev/null || git checkout -b "$HUB_BRANCH" "origin/$HUB_BRANCH"

echo "==> fork submodule at the pinned SHA"
cd "$ENGINE_REPO"
git submodule update --init --recursive --depth 1
PINNED=$(git rev-parse HEAD:third_party/NetHack)
ACTUAL=$(git -C third_party/NetHack rev-parse HEAD)
[ "$PINNED" = "$ACTUAL" ] || { echo "submodule $ACTUAL != pinned $PINNED"; exit 1; }
echo "    submodule ok: $ACTUAL"

echo "==> venv (python 3.12, verifiers 0.2.1 stock)"
cd "$HUB_REPO"
uv venv --python 3.12 .venv-cli-eval
# shellcheck disable=SC1091
source .venv-cli-eval/bin/activate
uv pip install pybind11 numpy pytest pytest-asyncio

echo "==> build libnethack.so against the 3.12 interpreter"
SRC="$ENGINE_REPO/third_party/NetHack/src"
rm -rf "$SRC/build"
cmake -S "$SRC" -B "$SRC/build" -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DPYTHON_EXECUTABLE="$(command -v python)"
cmake --build "$SRC/build" --target nethack -j"${JOBS:-16}"

echo "==> python packages (local engine, NOT the git URLs)"
uv pip install -e "$ENGINE_REPO/nethack_core" -e "$ENGINE_REPO/nethack_interface"
uv pip install -e environments/nethack --no-deps
uv pip install "verifiers>=0.2.1,<0.3" "gymnasium>=0.29" "numpy>=1.24" \
               "datasets>=2.14" "pillow>=10"
uv pip install -e harnesses/nethack-prime-agent

echo "==> prime CLI (inference)"
uv tool install prime

echo "==> node (claude_code / prime_agent arms)"
if ! command -v node >/dev/null; then
  curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
  apt-get install -y -qq nodejs
fi

echo "==> prime-agent scaffold (arm 3) -- NOT on npm, Prime's own installer"
if ! command -v prime-agent >/dev/null; then
  curl -fsSL https://pub-728493de92a943e2a9b2d17b4719f318.r2.dev/install.sh | sh
fi
prime-agent --version   # must match `version` in configs/prime_agent.toml (0.3.3)

echo "==> preflight"
export ENG="$ENGINE_REPO"
export PYTHONPATH="$HUB_REPO/tools/pycompat:$ENG:$HUB_REPO:$HUB_REPO/environments/nethack"
python tools/cli_harness_eval/engine_provenance.py --check

cat <<EOF

Done. Activate with:

  export PATH="\$HOME/.local/bin:\$PATH"
  source $HUB_REPO/.venv-cli-eval/bin/activate
  cd $HUB_REPO
  export ENG=$ENGINE_REPO
  export PYTHONPATH="\$PWD/tools/pycompat:\$ENG:\$PWD:\$PWD/environments/nethack"

All three arms are installed. Still needed before a real run:

  prime login --headless --plain
  key=\$(python -c "import json;print(json.load(open('\$HOME/.prime/config.json'))['api_key'])")
  export PI_API_KEY="\$key" PRIME_API_KEY="\$key"

See RUNBOOK.md §8.
EOF
