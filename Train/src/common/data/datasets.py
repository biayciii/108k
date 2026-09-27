"""Load ảnh + nhãn theo 1 trong 3 định dạng (`detr_data` COCO JSON, `faster-rcnn-data` CSV,
`yolov7-data` YOLO txt trong thư mục `Data/` ở root repo), thống nhất qua một interface duy
nhất để `src/engine/trainer.py` dùng chung bất kể kiến trúc nào.

CHƯA implement — cố ý chặn lại ở đây thay vì viết loader đọc thẳng `Data/faster-rcnn-data`
hiện tại, vì theo proposal.md mục 2.3/5.1: CSV đó còn chứa tên thật + năm sinh bệnh nhân
(PII), và 3 định dạng đang dùng 3 split lệch nhau (patient-level leakage). Phải chờ T1-T3
(split canonical + pseudonymize) xong trước, để không loader nào vô tình đọc lại dữ liệu rò
rỉ/PII cũ vào pipeline mới.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def build_dataset(data_root: str | Path, ids: list[str], cfg: Any):
    raise NotImplementedError(
        "Chưa implement — chờ split canonical (T1/T2) + pseudonymize CSV (T3) xong trước. "
        "Xem Train/init.md mục 1 (src/common/data/datasets.py) và proposal.md mục 5.1/2.3."
    )
