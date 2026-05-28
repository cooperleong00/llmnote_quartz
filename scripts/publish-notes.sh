#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
COMMIT_MESSAGE="${1:-Update content snapshot}"

cd "$REPO_ROOT"

"$SCRIPT_DIR/snapshot-content.sh"

git add content

if git diff --cached --quiet -- content; then
  echo "No content changes to commit."
else
  git commit -m "$COMMIT_MESSAGE"
fi

git push
