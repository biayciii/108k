# Train/init.md — Thiết kế khung code huấn luyện (chuẩn hoá, FL-ready)

> Tài liệu thiết kế — CHƯA phải code đã chạy được. Ghi rõ cây thư mục đề xuất và khung
> `train.py` tham số hoá để bàn trước khi hiện thực hoá. Bối cảnh đầy đủ: `proposal.md` mục
> 5.1/5.2/5.3, quyết định D1–D5 trong `.agents/record.md`, task T8–T13 trong `plan.csv`.

## 0. Vì sao có `Train/` riêng, không sửa trực tiếp `uet-thyroid-detection-main/src`

- `uet-thyroid-detection-main/` giữ nguyên làm tham chiếu tái lập kết quả D1–D5 đã công bố —
  không sửa tại chỗ để không phá khả năng đối chiếu ngược.
- `Train/` là nơi triển khai **cấu hình huấn luyện đã chuẩn hoá** (control-variables dùng
  chung, proposal.md 5.2) và sau này là **FL simulation** (5.3) — logic dữ liệu/huấn luyện
  được viết lại theo audit đã phát hiện (patient-level split, cấu hình tường minh, không
  hardcode path), còn **định nghĩa model** (kiến trúc DETR/Faster-RCNN/YOLOv7 tự thân) vẫn tái
  dùng từ code gốc bằng import — không copy-paste, không viết lại từ đầu.
- **Lưu ý cần xác nhận**: `plan.csv` hiện ghi T8 là "Tạo `shared/` trong
  `uet-thyroid-detection-main`". Tài liệu này đổi hướng sang một thư mục `Train/` ở root repo
  thay vì lồng bên trong `uet-thyroid-detection-main`. Chưa tự sửa lại mô tả T8/T9 trong
  `plan.csv` — cần bạn xác nhận trước khi đổi.

## 1. Cây thư mục đề xuất

```
Train/
├── init.md                    # tài liệu này
├── configs/
│   ├── base.yaml               # control-variables dùng chung (proposal.md 5.2): brightness
│   │                           # level, image size, augmentation, epoch/round tương đương
│   ├── detr.yaml                # override riêng DETR, kế thừa base.yaml
│   ├── faster_rcnn.yaml
│   └── yolov7.yaml
├── src/
│   ├── __init__.py
│   ├── common/                 # CHỈ đặt ở đây nếu thật sự không đặc thù kiến trúc nào
│   │   ├── data/
│   │   │   ├── split.py         # đọc/ghi split canonical patient-level (T1/T2, plan.csv)
│   │   │   ├── datasets.py       # load ảnh + nhãn, thống nhất interface cho cả 3 định dạng
│   │   │   └── transforms.py      # crop, percentile windowing, brightness augmentation
│   │   │                         # (tái cấu trúc từ increase_count/create_imbatch trong app.py)
│   │   ├── metrics/
│   │   │   ├── detection.py       # mAP@0.5 (thyroid), mAP@0.3 (shoulder) — quy ước D3
│   │   │   └── diagnosis.py        # RSI, Precision/Recall/F1/Accuracy
│   │   └── utils/
│   │       ├── seed.py
│   │       ├── config.py          # load YAML + override CLI, LUÔN ghi resolved config ra
│   │       │                     # run dir (khắc phục lỗi thiếu cấu hình Faster-RCNN, 2.4)
│   │       └── checkpoint.py
│   ├── models/                  # đặc thù riêng từng kiến trúc — KHÔNG dùng chung
│   │   ├── detr/                 # import từ uet-thyroid-detection-main/src/detr, không copy
│   │   ├── faster_rcnn/          # import từ uet-thyroid-detection-main/src/faster_rcnn
│   │   └── yolov7/               # import từ uet-thyroid-detection-main/src/yolov7 (đóng gói
│   │                             # thành package thật, bỏ sys.path.insert hack hiện có)
│   └── engine/
│       ├── trainer.py            # interface chung Trainer.fit()/.evaluate(); mỗi kiến trúc
│       │                        # implement 1 subclass — để train.py không if/else lan man
│       └── federated.py          # FedAvg/FedProx (T11/T12) — làm SAU, chưa cần khi tạo khung
├── scripts/
│   ├── train.py                  # entrypoint duy nhất, tham số hoá — xem mục 3
│   └── evaluate.py
└── runs/                         # output checkpoint/log mỗi lần chạy — gitignore, KHÔNG
                                  # commit trọng số vào git
```

## 2. Nguyên tắc thiết kế (rút từ Data & Training-Protocol Audit, proposal.md mục 2)

1. **Một entrypoint duy nhất** cho cả 3 kiến trúc — chọn qua `--arch {detr,faster_rcnn,yolov7}`,
   không phải 3 script train riêng biệt không đồng bộ như hiện tại.
2. **Không hardcode đường dẫn tuyệt đối** kiểu `/media/vinh/.../checkpoints/epoch_011.ckpt`
   như trong `app.py` hiện tại — mọi đường dẫn đi qua config hoặc CLI argument.
3. **Không dùng `sys.path.insert` hack** như `app.py` (`sys.path.insert(0, './src/yolov7')`) —
   đóng gói `models/yolov7` thành package import được bình thường.
4. **Luôn ghi lại full resolved config** (base + override kiến trúc + override CLI) ra run dir
   khi train — khắc phục đúng lỗi "Faster-RCNN không có cấu hình huấn luyện nào được lưu lại"
   nêu ở proposal.md mục 2.4.
5. **Dùng đúng 1 split canonical** (kết quả T1/T2 trong `plan.csv`) cho cả 3 kiến trúc —
   `train.py` chỉ đọc split có sẵn, không tự tạo split riêng theo từng lần chạy.
6. **Control-variables cố định trong config**, không để mỗi model tự chọn brightness level
   riêng như hiện tại (`frcnn=2, detr=4, yolov7=3` trong `app.py`) — đúng yêu cầu 5.2.
7. **Seed cố định + log seed** trong resolved config, để một lần chạy có thể tái lập lại.

## 3. Khung `train.py` tham số hoá (minh hoạ thiết kế, chưa phải bản cuối)

```python
# Train/scripts/train.py
import argparse
from pathlib import Path

from src.common.utils.config import load_config, save_resolved_config
from src.common.utils.seed import set_seed
from src.common.data.split import load_canonical_split
from src.common.data.datasets import build_dataset
from src.engine.trainer import build_trainer  # registry: {"detr": DetrTrainer, ...}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", required=True, choices=["detr", "faster_rcnn", "yolov7"])
    p.add_argument("--config", type=Path, default=None,
                   help="Override YAML riêng kiến trúc; mặc định configs/<arch>.yaml")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--split-file", type=Path, required=True,
                   help="Split canonical từ T1/T2 (plan.csv), dùng chung cho cả 3 kiến trúc")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=None, help="Override config nếu có")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--brightness-level", type=int, default=None,
                   help="Control-variable dùng chung — KHÔNG để mặc định khác nhau giữa model")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", type=Path, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    base_cfg = load_config("configs/base.yaml")
    arch_cfg = load_config(args.config or f"configs/{args.arch}.yaml")
    cfg = base_cfg.merge(arch_cfg).override_from_cli(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_resolved_config(cfg, args.output_dir / "resolved_config.yaml")  # nguyên tắc #4

    split = load_canonical_split(args.split_file)  # nguyên tắc #5
    train_set = build_dataset(args.data_root, split.train, cfg)
    val_set = build_dataset(args.data_root, split.val, cfg)

    trainer = build_trainer(args.arch, cfg, resume=args.resume)  # nguyên tắc #1
    trainer.fit(train_set, val_set, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
```

```
# Ví dụ chạy:
python scripts/train.py --arch detr \
  --data-root ../Data/detr_data \
  --split-file ../Data/canonical_split.json \
  --output-dir runs/detr_run01 \
  --brightness-level 3
```

## 4. Trạng thái hiện tại (cập nhật 2026-09-28, sau D7)

Đã xác nhận: `uet-thyroid-detection-main/` là legacy, không đụng đến (xem D7,
`.agents/record.md` mục 2). Scaffold `Train/` đã dựng thật — không còn là thiết kế suông:

- Đã tạo đủ cây thư mục mục 1, đã smoke-test qua (`config.py` merge/override/save,
  `transforms.py` + `metrics/diagnosis.py` chạy với dữ liệu giả lập, `scripts/train.py` chạy
  hết tới đúng điểm dừng dự kiến — `NotImplementedError` ở `load_canonical_split`).
- `configs/{yolov7,detr}.yaml` đã điền **giá trị thật** port từ cấu hình/argparse hiện có
  trong `uet-thyroid-detection-main` (không phải giá trị bịa) — coi như hoàn thành phần
  "trích xuất tường minh" của T4. `configs/faster_rcnn.yaml` vẫn để `null` đúng theo phát hiện
  audit (không có cấu hình nào được lưu lại — T5 chưa làm).
- `plan.csv` T8 đã có DoD (check) kiểm được; T9 được thu hẹp lại và có DoD riêng — xem D7.

**Phát hiện mới khi scaffold — CHƯA giải quyết, chặn bước wiring model thật (T8 phần còn lại)**:
`uet-thyroid-detection-main/src/` và `Train/src/` đều là package tên `src`. Nếu
`src/models/{detr,faster_rcnn,yolov7}/__init__.py` import kiểu `from src.detr...` sau khi
`Train/src` đã được nạp làm module `src`, Python sẽ resolve nhầm vào `Train/src` (rỗng) thay vì
package `src` thật trong `uet-thyroid-detection-main` — lỗi import sai **âm thầm**, không báo
rõ ràng. Hai phương án chưa chọn:
  (a) đổi tên package `Train/src` (vd `train_lib`) để hết đụng tên — ảnh hưởng mọi import nội
      bộ đã viết trong scaffold này;
  (b) dùng `importlib.util.spec_from_file_location` nạp thẳng theo đường dẫn file, không qua
      tên module `src` chung — không cần đổi tên nhưng verbose hơn ở mỗi chỗ import model.
Cần xác nhận trước khi viết code thật trong `src/models/*/__init__.py`.

## 5. Việc CHƯA làm (chờ xác nhận/tuần tự theo plan.csv)

- Chưa giải quyết xung đột tên package `src` nêu ở mục 4 — chặn wiring model thật.
- Chưa implement `federated.py` (FedAvg/FedProx) — đó là T11/T12, phụ thuộc T8 xong trước và
  phụ thuộc T10 (partition N-client).
- Chưa implement `common/data/split.py` và `common/data/datasets.py` thật — chờ T1-T3.
- Chưa implement `common/metrics/detection.py` (mAP) — cần chốt 1 công thức dùng chung cho cả
  3 kiến trúc trước (hiện D1-D5 tự tính khác nhau giữa các paper).
- Đã tự viết YAML load/merge nhỏ gọn thay vì OmegaConf/Hydra (mục 4 cũ) — giữ nguyên lựa chọn
  này, đã đủ dùng cho merge 2 cấp base+arch hiện tại.
