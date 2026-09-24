#!/usr/bin/env bash
# .agents/hooks/post-commit.sh
# Auto-installed into .git/hooks/post-commit by .agents/hooks/install.sh
# Parses the latest commit message as `[TaskID] <wip|done|blocked>: <mô tả>`,
# flips the matching row's Status in plan.csv, and appends a line to
# .agents/action-history.md. KHÔNG chỉnh sửa file này để ghi tay — chỉ đọc.

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
PLAN_CSV="$REPO_ROOT/plan.csv"
ACTION_HISTORY="$REPO_ROOT/.agents/action-history.md"

commit_msg="$(git log -1 --pretty=%B)"
commit_hash="$(git rev-parse --short HEAD)"
timestamp="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

# Parse `[TaskID] <wip|done|blocked>: <mô tả>`
if [[ "$commit_msg" =~ ^\[([A-Za-z0-9_-]+)\][[:space:]]+(wip|done|blocked):[[:space:]]*(.*)$ ]]; then
  task_id="${BASH_REMATCH[1]}"
  action="${BASH_REMATCH[2]}"
else
  exit 0
fi

if [[ ! -f "$PLAN_CSV" ]]; then
  echo "[post-commit] Cảnh báo: không tìm thấy $PLAN_CSV" >&2
  exit 0
fi

python3 "$REPO_ROOT/.agents/hooks/_post_commit.py" \
  "$PLAN_CSV" "$ACTION_HISTORY" "$task_id" "$action" "$timestamp" "$commit_hash"
