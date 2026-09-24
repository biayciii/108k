#!/usr/bin/env bash
# .agents/hooks/install.sh
# .git/hooks/ không được git theo dõi -> phải chạy lại script này mỗi khi clone
# repo ở máy mới, để cài post-commit hook.

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
SRC="$REPO_ROOT/.agents/hooks/post-commit.sh"
DEST="$REPO_ROOT/.git/hooks/post-commit"

cp "$SRC" "$DEST"
chmod +x "$DEST"

echo "Đã cài $DEST từ $SRC"
