#!/usr/bin/env python3
"""Helper for post-commit.sh — flips plan.csv Status and appends action-history.md.

Not meant to be run by hand; post-commit.sh invokes this with fixed args.
"""
import csv
import subprocess
import sys
from pathlib import Path

STATUS_COL = "Status"
DOD_COL = "DoD (check)"
TASKID_COL = "TaskID"


def dod_passes(dod: str, repo_root: Path) -> bool:
    dod = dod.strip()
    if not dod:
        return False
    candidate = repo_root / dod
    if candidate.exists():
        return True
    result = subprocess.run(dod, shell=True, cwd=repo_root)
    return result.returncode == 0


def main() -> None:
    plan_csv, action_history, task_id, action, timestamp, commit_hash = sys.argv[1:7]
    plan_path = Path(plan_csv)
    repo_root = plan_path.parent

    with plan_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        fieldnames = rows[0].keys() if rows else None

    if fieldnames is None:
        print(f"[post-commit] Cảnh báo: {plan_csv} rỗng hoặc không đọc được.", file=sys.stderr)
        return

    target = None
    for row in rows:
        if row.get(TASKID_COL, "").strip() == task_id:
            target = row
            break

    if target is None:
        print(f"[post-commit] Cảnh báo: không tìm thấy TaskID '{task_id}' trong plan.csv — không sửa gì.", file=sys.stderr)
        return

    new_status = None
    if action == "wip":
        if target[STATUS_COL].strip() == "Not started":
            target[STATUS_COL] = "In progress"
        new_status = target[STATUS_COL]
    elif action == "done":
        dod = target.get(DOD_COL, "")
        if dod.strip() and dod_passes(dod, repo_root):
            target[STATUS_COL] = "Done"
        else:
            reason = "DoD (check) trống" if not dod.strip() else "DoD (check) fail"
            print(f"[post-commit] '{task_id}' giữ nguyên '{target[STATUS_COL]}' — {reason}, không tự đánh Done.", file=sys.stderr)
        new_status = target[STATUS_COL]
    elif action == "blocked":
        target[STATUS_COL] = "Blocked"
        new_status = target[STATUS_COL]

    with plan_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    history_path = Path(action_history)
    with history_path.open("a", encoding="utf-8") as f:
        f.write(f"| {timestamp} | {commit_hash} | {task_id} | {new_status} |\n")


if __name__ == "__main__":
    main()
