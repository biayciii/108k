"""Re-export định nghĩa model YOLOv7 từ `uet-thyroid-detection-main/src/yolov7` — KHÔNG
copy-paste. Cùng vấn đề packaging (2 package tên `src` trùng nhau) như
`src/models/detr/__init__.py` — xem docstring ở đó.

Thêm 1 vấn đề riêng của YOLOv7: code gốc dùng `sys.path.insert(0, './src/yolov7')` trong
`app.py` rồi import trực tiếp `from models.experimental import attempt_load` (không qua
package `src`) — một hack path riêng, khác hẳn cách DETR/Faster-RCNN được import. Nguyên tắc
thiết kế #3 (Train/init.md) muốn bỏ hack này, nhưng chưa tự viết lại vì cần đảm bảo không phá
việc load checkpoint `yolov7.pt` hiện có (kiến trúc model phải khớp bit-for-bit).
"""
