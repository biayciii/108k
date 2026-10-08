# Working process
Gotchas and working flows learned the hard way (setup, environment, review habits). Append only.

Format:
## YYYY-MM-DD — short title
What happens, what to do instead.

## Standup checks
Read by `/plan-check` when it is installed. Add 3 to 6 checks that fit this project as bullet lines starting with `- ` below this paragraph. One check per line: a read-only command to run or a file to read, and what counts as a failure. `/plan-check` runs them and prints only the ones that fail.
Examples (not active, copy the ones you want): the gate passes on the default branch; a task is `done` but its `review` is still `pending`; a task is in progress while a task in its `depends` is not done; the tracker and `plan.csv` disagree (only with an external tracker); every number in the report cites a run-id that is a chosen result in `decisions-log.md` (research projects).

## 2026-10-08 — plan.csv format KHÔNG theo template agent-gov 0.9.0

Repo này giữ header `plan.csv` tiếng Việt (`TaskID, Task, ..., Status, ...`) và quy ước commit
`[TaskID] <wip|done|blocked>: ...` vì `.agents/hooks/_post_commit.py` phụ thuộc chính xác các tên
cột `TaskID`/`Status`. Lệnh `/done` của template 0.9.0 (yêu cầu cột `status,mr,review,...`)
đã bị gỡ khỏi `.opencode/commands/` — KHÔNG dùng được với plan.csv hiện tại. Muốn dùng `/done`
phải migrate plan.csv sang header tiếng Anh trước (khi đó phải viết lại hook).
Gate command để trống: repo nghiên cứu chưa có lint/test chuẩn.
