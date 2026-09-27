"""Tiền xử lý ảnh SPECT — port lại nguyên logic từ
`uet-thyroid-detection-main/app.py` (`increase_count`, `create_imbatch`, `irrelevant_cutting`)
và pipeline mô tả trong `README.md` (root) mục 3. Hành vi giữ NGUYÊN so với bản gốc, chỉ tách
ra thành hàm dùng lại được giữa 3 kiến trúc thay vì nằm rải rác trong `app.py`.

KHÔNG cần chờ T1-T7 — đây là phần tiền xử lý ảnh thuần tuý, không phụ thuộc split/PII/cấu hình
huấn luyện, nên viết thật ngay ở bước scaffold này thay vì để NotImplementedError.
"""
from __future__ import annotations

import numpy as np
from scipy import signal


def crop_whole_body(img: np.ndarray, keep_rows: int = 512) -> np.ndarray:
    """Cắt nửa trên ảnh whole-body scan 1024x256 -> 512x256, tập trung vùng cổ/ngực.
    README.md (root) mục 3 Bước 1."""
    return img[:keep_rows]


def percentile_normalize(img: np.ndarray, p_min: float = 1.0, p_max: float = 99.5) -> np.ndarray:
    """Percentile windowing + min-max scale về uint8 [0, 255]. README.md (root) mục 3 Bước 2."""
    lo = np.percentile(img, p_min)
    hi = np.percentile(img, p_max)
    clipped = np.clip(img, lo, hi)
    normalized = (clipped - lo) / (hi - lo + 1e-8) * 255.0
    return normalized.astype(np.uint8)


def increase_brightness(img: np.ndarray, factor: float = 1.0) -> np.ndarray:
    """Tăng cường độ sáng bằng convolution kernel cố định — port nguyên logic từ
    `app.py::increase_count`, KHÔNG đổi hành vi so với bản gốc."""
    if factor <= 0:
        return img
    kernel = (
        np.array(
            [
                [1, 1, 1, 1, 1],
                [1, 2, 2, 2, 1],
                [1, 2, 4, 2, 1],
                [1, 2, 2, 2, 1],
                [1, 1, 1, 1, 1],
            ]
        )
        * factor
    )
    out = signal.convolve2d(img, kernel, boundary="symm", mode="same")
    return np.clip(out, 0, 255)


def build_brightness_stack(img: np.ndarray, brightness_levels: int) -> np.ndarray:
    """Xếp chồng nhiều mức tăng sáng thành các kênh HxWxN — port từ `app.py::create_imbatch`.

    Nguyên tắc thiết kế #6 (Train/init.md): `brightness_levels` PHẢI lấy từ
    `config.data.brightness_level` (control-variable cố định, chốt ở T6) — KHÔNG hardcode
    khác nhau giữa model như `app.py` gốc (`frcnn=2, detr=4, yolov7=3`).
    """
    factors = [2**i for i in range(brightness_levels - 1)]
    factors.insert(0, 0)
    stack = np.vstack([increase_brightness(img, f) for f in factors]).reshape(
        brightness_levels, *img.shape
    )
    return stack.transpose(1, 2, 0).astype(np.uint8)
