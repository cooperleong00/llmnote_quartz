#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

SOURCE_DIR="${QUARTZ_CONTENT_SOURCE:-$HOME/Obsidian/LLM}"
CONTENT_DIR="${QUARTZ_CONTENT_DIR:-content}"

if [[ "$SOURCE_DIR" != /* ]]; then
  SOURCE_DIR="$REPO_ROOT/$SOURCE_DIR"
fi

if [[ "$CONTENT_DIR" != /* ]]; then
  CONTENT_DIR="$REPO_ROOT/$CONTENT_DIR"
fi

if [[ ! -d "$SOURCE_DIR" ]]; then
  echo "Content source does not exist: $SOURCE_DIR" >&2
  exit 1
fi

if ! command -v rsync >/dev/null 2>&1; then
  echo "rsync is required to create the content snapshot." >&2
  exit 1
fi

if [[ -L "$CONTENT_DIR" ]]; then
  rm "$CONTENT_DIR"
elif [[ -e "$CONTENT_DIR" && ! -d "$CONTENT_DIR" ]]; then
  echo "Content destination exists but is not a directory: $CONTENT_DIR" >&2
  exit 1
fi

mkdir -p "$CONTENT_DIR"

SOURCE_REAL="$(cd "$SOURCE_DIR" && pwd -P)"
CONTENT_REAL="$(cd "$CONTENT_DIR" && pwd -P)"

if [[ "$SOURCE_REAL" == "$CONTENT_REAL" ]]; then
  echo "Content source and destination resolve to the same directory." >&2
  exit 1
fi

rsync -a --delete --delete-excluded \
  --exclude=".*/" \
  --exclude=".*" \
  --exclude=".git/" \
  --exclude=".obsidian/" \
  --exclude=".trash/" \
  --exclude=".DS_Store" \
  --exclude="AGENTS.md" \
  --exclude="CLAUDE.md" \
  --exclude="pyproject.toml" \
  --exclude="skills-lock.json" \
  --exclude="uv.lock" \
  --exclude="private/" \
  --exclude="scripts/" \
  --exclude="templates/" \
  "$SOURCE_DIR"/ "$CONTENT_DIR"/

file_count="$(find "$CONTENT_DIR" -type f | wc -l | tr -d ' ')"
echo "Snapshot written to ${CONTENT_DIR#$REPO_ROOT/} from $SOURCE_DIR ($file_count files)."
