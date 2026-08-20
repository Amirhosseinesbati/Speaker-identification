#!/bin/bash
# ============================================================================
# setup_vast_onstart.sh — tiny Vast.ai onstart launcher
#
# The full bootstrap (setup_vast.sh) is ~18 KB — bigger than the 16 KB limit
# vast.ai's create-instance API puts on the onstart script, so sending it
# directly fails with "Invalid args: len(args) > 16384". This launcher (a few
# hundred bytes) clones the repo from GIT_REPO_URL / GIT_BRANCH — which
# deploy.py already passes as env vars — and runs the committed setup_vast.sh.
# Works for the CLI/Streamlit flow AND when pasted manually into the vast.ai
# website UI (pass GIT_REPO_URL + GIT_BRANCH as env vars there too).
# ============================================================================

set -e

echo "🚀 setup_vast_onstart: cloning ${GIT_REPO_URL} (${GIT_BRANCH:-main})..."

# The pytorch base image ships git; install it only on minimal images.
if ! command -v git >/dev/null 2>&1; then
    apt-get update -qq && apt-get install -y -qq git 2>/dev/null || true
fi

rm -rf /workspace/project
git clone -b "${GIT_BRANCH:-main}" "${GIT_REPO_URL}" /workspace/project
cd /workspace/project

exec bash setup_vast.sh