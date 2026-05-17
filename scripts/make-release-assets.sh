#!/usr/bin/env bash
# Create release asset archives for wiki and any extra folders.
# Usage: ./scripts/make-release-assets.sh [VERSION]
# Output: model-tools-wiki-{VERSION}.zip in the current directory.
set -euo pipefail

VERSION="${1:-$(python3 -c "import model_tools; print(model_tools.__version__)" 2>/dev/null || true)}"
VERSION="${VERSION:-unknown}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "Creating release assets for model-tools v${VERSION}..."

# Wiki
if [ -d "${REPO_ROOT}/wiki" ]; then
    cd "${REPO_ROOT}"
    zip -r "${REPO_ROOT}/model-tools-wiki-${VERSION}.zip" wiki/
    echo "  ✓ model-tools-wiki-${VERSION}.zip"
fi

echo "Done. Upload these to the GitHub release."
