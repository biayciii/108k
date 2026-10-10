"""Re-export định nghĩa model DETR từ `uet-thyroid-detection-main/src/detr` — KHÔNG copy-paste
(nguyên tắc thiết kế D7 trong `.agents/record.md`, `Train/init.md` mục 0/2).

CHƯA wiring thật. Phát hiện khi scaffold: `uet-thyroid-detection-main/src/` VÀ `Train/src/` đều
là package tên `src` — nếu import kiểu `from src.detr.models.detr import DETR` sau khi
`Train/src` đã được Python nạp làm module `src`, Python sẽ tìm nhầm vào `Train/src.detr` (rỗng)
thay vì package `src` thật trong `uet-thyroid-detection-main`, gây import sai âm thầm chứ
không báo lỗi rõ ràng. Cần giải quyết TRƯỚC khi viết code import thật ở đây, ví dụ:
  (a) đổi tên package `Train/src` thành tên khác (vd `train_lib`) để hết đụng tên, hoặc
  (b) dùng `importlib.util.spec_from_file_location` nạp thẳng theo đường dẫn file, không qua
      tên module `src` chung.
Chưa tự chọn phương án — ghi lại làm câu hỏi treo trong `Train/init.md` mục 4.
"""
