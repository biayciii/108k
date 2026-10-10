"""RSI (chỉ số hấp thu dùng chẩn đoán tồn dư mô giáp) — port nguyên công thức từ
`app.py::cal_avg_uptake` + `app.py::predict()`: RSI = avg_uptake(thyroid_box) -
avg_uptake(shoulder_box). Xem D3 trong `.agents/record.md` mục 2.

Đây là phần đã có công thức rõ ràng (không phụ thuộc split/cấu hình training chưa chốt) nên
viết thật ngay ở bước scaffold, khác với `common/metrics/detection.py` (mAP) vẫn còn treo.
"""
from __future__ import annotations

import numpy as np


def truncate_bbox(bbox: list[float], h: int, w: int) -> list[float]:
    bbox = [max(0.0, e) for e in bbox]
    bbox[0] = min(bbox[0], w - 1)
    bbox[2] = min(bbox[2], w - 1)
    bbox[1] = min(bbox[1], h - 1)
    bbox[3] = min(bbox[3], h - 1)
    return bbox


def average_uptake(img: np.ndarray, bbox: list[float]) -> float:
    """Trung bình log-uptake trong vùng elip nội tiếp bbox — nguyên logic
    `app.py::cal_avg_uptake`."""
    bbox = truncate_bbox(bbox, *img.shape)
    a = (bbox[2] - bbox[0]) / 2
    b = (bbox[3] - bbox[1]) / 2
    c_x = (bbox[0] + bbox[2]) / 2
    c_y = (bbox[1] + bbox[3]) / 2
    yy, xx = np.mgrid[0 : img.shape[0], 0 : img.shape[1]]
    mask = ((xx - c_x) ** 2 / (a**2) + (yy - c_y) ** 2 / (b**2)) <= 1
    return float(np.log(img[mask].sum()))


def compute_rsi(
    img: np.ndarray,
    thyroid_box: list[float] | None,
    shoulder_box: list[float] | None,
) -> float:
    if thyroid_box is None or shoulder_box is None:
        return -1.0
    return average_uptake(img, thyroid_box) - average_uptake(img, shoulder_box)
