# Learnings
Append only. Every error fixed becomes an entry so nobody hits it twice.

Format:
## YYYY-MM-DD — short title
- Symptom:
- Root cause:
- Fix:
- Lesson:

> Copy nguyên từ `.agents/record.md` (mục 3) ngày 2026-10-08.

## 3. Bài học

- **Lỗi**: `yaml.safe_dump()` (PyYAML `SafeDumper`) không tự nhận diện `Config` (dict
  subclass tự viết trong `Train/src/common/utils/config.py`) là `dict` — representer của
  PyYAML tra theo `type(data)` chính xác, không theo `isinstance`, nên dump thất bại với
  `RepresenterError: cannot represent an object` ngay khi config có dict con lồng nhau (mọi
  config thật đều có, vd `training:`, `data:`).
  **Cách sửa**: thêm hàm `_to_plain()` chuyển đệ quy `Config`/dict lồng nhau về `dict`/`list`
  thuần trước khi `yaml.safe_dump`.
  **Rút ra**: bất kỳ subclass nào của `dict`/`list` (kể cả tiện lợi khi viết code) đều cần tự
  convert về type gốc trước khi đưa qua thư viện serialize ngoài (yaml/json) — không giả định
  thư viện tự xử lý qua `isinstance`. Phát hiện được nhờ chủ động smoke-test
  `save_resolved_config()` với config lồng nhau thật, không chỉ test với dict phẳng.
