"""mAP@0.5 (thyroid) / mAP@0.3 (shoulder, quy ước D3 — xem `.agents/record.md` mục 2) — CHƯA
tự implement lại từ đầu ở đây.

Ý định thiết kế: wrap lại đúng hàm mAP đã có sẵn và đã dùng để ra số liệu D1-D5, tránh lệch
công thức so với các paper gốc:
- YOLOv7: `uet-thyroid-detection-main/src/yolov7/utils/metrics.py` (`ap_per_class`)
- DETR: COCOeval chuẩn COCO (pycocotools), xem `uet-thyroid-detection-main/src/detr/datasets`
- Faster-RCNN: chưa có eval script nào sẵn (nhất quán với việc thiếu train.py — proposal.md 2.4)

CHƯA implement — cần chốt 1 công thức mAP DÙNG CHUNG cho cả 3 kiến trúc trước (hiện D1-D5 tự
tính theo cách khác nhau giữa các paper), nếu không baseline centralized mới (T7 trong
plan.csv) sẽ lại không so sánh công bằng được giữa 3 model — đúng lỗi audit đã phát hiện ở D6.
"""
from __future__ import annotations


def compute_map(*args, **kwargs):
    raise NotImplementedError(
        "Chưa implement — cần chốt 1 công thức mAP dùng chung cho cả 3 kiến trúc trước (xem "
        "docstring module này). Việc này nằm ngoài phạm vi scaffold Train/init.md ban đầu."
    )
