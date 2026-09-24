# AGENTS.md — 108 Thyroid

**Project:** 108 Thyroid — nghiên cứu xây dựng pipeline phát hiện mô tuyến giáp còn sót lại
sau phẫu thuật cắt tuyến giáp từ ảnh xạ hình SPECT. So sánh 3 kiến trúc phát hiện đối tượng
(Faster R-CNN, DETR, YOLOv7) trên cùng bộ dữ liệu DICOM đã tiền xử lý để chọn mô hình
tối ưu; kết quả detection được dùng để tính chỉ số RSI hỗ trợ chẩn đoán tồn dư mô giáp.

Đây là luật hiện hành — đọc file này đầu mỗi phiên.

## Việc đầu mỗi phiên

Kiểm tra đủ:
- `.agents/AGENTS.md` (file này)
- `.agents/record.md`
- `.agents/action-history.md`
- `plan.csv` ở root repo
- Git repo hợp lệ (`git status` chạy được)

Thiếu file/thư mục nào → tạo lại từ template tương ứng, in ra rõ ràng đã tạo gì. Không tự
suy diễn nội dung mới cho các file bị thiếu — nếu không chắc nội dung gốc, hỏi trước.

## Frozen paths

CHƯA CÓ — hỏi ngay khi phát sinh dữ liệu thô/checkpoint/kết quả cần bảo vệ khỏi bị ghi đè,
đừng tự đoán tên thư mục trước khi nó tồn tại.

(Gợi ý các ứng viên khi phát sinh: `Data/` gốc chưa qua xử lý, checkpoint model cuối cùng
dùng để báo cáo, kết quả/số liệu đã trích dẫn vào bản thảo bài báo.)

## Status trong plan.csv

Agent KHÔNG được tự đánh dấu Status = `Done`. Chỉ được đẩy tới `In progress` hoặc `Blocked`
qua commit. `Done` chỉ do post-commit hook gán (khi cột `DoD (check)` của dòng đó pass) hoặc
do người xác nhận tay.

## Quy ước commit (bắt buộc)

```
[TaskID] <wip|done|blocked>: <mô tả ngắn>
```

## Trách nhiệm record.md

Cập nhật `.agents/record.md` là trách nhiệm chủ động của agent — không phải việc của hook.
Có quyết định mới / bài học mới / câu hỏi treo mới hay được trả lời → ghi ngay, không đợi
được nhắc.

## Bảo mật dữ liệu

Đây là dữ liệu y tế (ảnh SPECT/DICOM bệnh nhân). Không dán dữ liệu nhạy cảm của project vào
công cụ AI bên ngoài không rõ chính sách lưu trữ.
