# Open questions
Add when a question opens. Change its state when it moves. Move to "Closed" with the answer; do not delete.

## Open
- YYYY-MM-DD — question — owner

## Closed
- YYYY-MM-DD — question — answer (link to decision/ADR)

> Copy nguyên từ `.agents/record.md` (mục 4) ngày 2026-10-08.

## 4. Câu hỏi treo

Trích từ hướng nghiên cứu tương lai/hạn chế mà các bài báo tự nêu (chưa rõ đã được giải
quyết ở phiên bản code hiện tại hay chưa):

- (D2) Có nên huấn luyện riêng một mô hình chỉ trên tập ca khó đã cân bằng lại để cải thiện
  hiệu năng nhóm khó (ANT-HN-POST), thay vì dùng chung mô hình cho cả ca dễ/khó?
- (D4) Có nên dự đoán mức độ tăng sáng (brightness level) phù hợp theo từng ảnh thay vì cố
  định số mức cho toàn bộ dataset?
- (D4) Cần tinh chỉnh sâu hơn hyperparameter của DETR (num_queries, số layer encoder/decoder,
  class_cost) — chưa rõ đã thử nghiệm có hệ thống hay chưa.
- (D5) Cần thử nghiệm lâm sàng (clinical validation) trên tập dữ liệu lớn hơn trước khi áp
  dụng mô hình đề xuất liều xạ trị vào thực tế điều trị.

**Câu hỏi treo vận hành cũ — ĐÃ TRẢ LỜI bởi D7 (2026-09-28), giữ lại để truy vết:**
- ~~Trong `src/{detr,faster_rcnn,yolov7}/`, phần nào thực sự dùng chung... đủ để đưa vào
  `shared/`?~~ → Không đưa vào `shared/` lồng trong uet nữa; toàn bộ code chung mới nằm ở
  `Train/src/common/`, xem `Train/init.md`.
- ~~"Hướng nghiên cứu gốc không dùng nữa" cụ thể là hướng nào?~~ → Không có hướng nào bị "loại
  bỏ"/di chuyển; `uet-thyroid-detection-main` giữ nguyên 100% làm legacy, chỉ đơn giản là không
  dùng để phát triển tiếp.
- ~~Tên thư mục đích cho code cũ?~~ → Không cần thư mục đích — không di chuyển gì cả (D7).

**Câu hỏi treo mới phát sinh khi scaffold `Train/` (D7, chặn phần còn lại của T8):**
- `uet-thyroid-detection-main/src/` và `Train/src/` đều là package tên `src` — import kiểu
  `from src.detr...` trong `Train/src/models/*/__init__.py` có nguy cơ resolve nhầm vào
  `Train/src` rỗng thay vì package thật bên `uet-thyroid-detection-main`. Đổi tên package
  `Train/src` (vd `train_lib`), hay dùng `importlib.util.spec_from_file_location` nạp thẳng
  theo đường dẫn file? Xem `Train/init.md` mục 4 để biết chi tiết + trade-off 2 phương án.

**Câu hỏi treo mới phát sinh từ Data & Training-Protocol Audit (D6, chặn T1–T7 trong plan.csv):**
- Patient ID dùng để GroupKFold (T1) nên lấy từ cột `name` (dễ trùng do lỗi chính tả/viết hoa
  khác nhau) hay cần một khoá định danh bệnh nhân ổn định hơn (vd DICOM PatientID nếu có trong
  metadata gốc, chưa kiểm tra)?
- Với Faster-RCNN không có cấu hình huấn luyện gốc (T5): tái tạo bằng cách suy luận ngược từ
  kiến trúc checkpoint `.ckpt` hiện có, hay chấp nhận train lại từ đầu với cấu hình mới và bỏ
  qua việc tái lập chính xác baseline cũ?
- Quy đổi "epoch tương đương" giữa centralized và FL (T6) nên tính theo tổng số ảnh đã nhìn thấy
  (gradient steps), hay theo số round × local-epoch cố định bất kể kích thước client?
