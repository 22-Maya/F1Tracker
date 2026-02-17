#!/usr/bin/env bash
# Helper to clone the upstream f1-race-replay into replay/upstream
# Usage: bash replay/setup_replay.sh
set -euo pipefail
REPO_URL="https://github.com/IAmTomShaw/f1-race-replay.git"
TARGET_DIR="replay/upstream"

if [ -d "$TARGET_DIR" ]; then
  echo "Target $TARGET_DIR already exists. If you want a fresh copy, remove it first."
  exit 1
fi

echo "Cloning $REPO_URL into $TARGET_DIR..."
git clone "$REPO_URL" "$TARGET_DIR"

echo "Done. Next steps:"
echo "1) Create and activate a Python 3.11+ virtualenv in the project root (recommended):"
echo "   python3 -m venv .venv && source .venv/bin/activate"
echo "2) Install dependencies for the replay viewer:"
echo "   pip install -r replay/upstream/requirements.txt"
echo "3) (Optional) If you want the viewer to use your project's cache folder, run:"
echo "   mkdir -p cache && ln -s \$PWD/cache replay/upstream/.fastf1-cache || true"
echo "4) Run a replay viewer (example):"
echo "   python replay/upstream/main.py --viewer --year 2025 --round 12"

echo "If you want, I can run these steps for you or scaffold a tighter integration into the web UI."
