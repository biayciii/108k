"""Đọc split canonical — kết quả T1/T2 trong plan.csv (patient-level GroupKFold, MỘT split
dùng chung cho cả 3 định dạng dữ liệu thay vì 3 split độc lập như hiện tại). Xem proposal.md
mục 5.1 và câu hỏi treo trong .agents/record.md mục 4 (khoá định danh bệnh nhân dùng để
GroupKFold — cột `name` hay DICOM PatientID — chưa quyết).

CHƯA implement: T1/T2 chưa chạy nên chưa có file split canonical thật để biết đúng schema
(JSON hay CSV, khoá bằng image ID hay DICOM UID). Định nghĩa `CanonicalSplit` bên dưới là
interface DỰ KIẾN cho phần còn lại của Train/ dùng — đổi lại khi T1/T2 chốt schema thật.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class CanonicalSplit:
    train: list[str]
    val: list[str]
    test: list[str]


def load_canonical_split(path: str | Path) -> CanonicalSplit:
    raise NotImplementedError(
        "Chưa có file split canonical (chờ T1/T2 trong plan.csv chạy xong). Khi có, cập nhật "
        "hàm này để đọc đúng schema thật mà T1/T2 sinh ra — xem proposal.md mục 5.1."
    )
