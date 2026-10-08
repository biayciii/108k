# AGENTS.md

> Migration agent-gov 0.9.0 (2026-10-08): `plan.csv` giữ format tiếng Việt + post-commit hook
> (xem `.agents/working-process.md`). Quyết định mới ghi vào `.agents/wiki/decisions-log.md`
> (D1–D8 đã copy nguyên từ `.agents/record.md`). `.agents/record.md` giữ làm archive.

## Profile

- What: Phát hiện mô tuyến giáp còn sót lại sau cắt tuyến giáp từ ảnh SPECT (DETR/Faster R-CNN/YOLOv7)
- Goal: Hướng Federated Learning, hoặc Domain Adaptation nếu đủ dữ liệu đa viện; chọn kiến trúc detector tối ưu và kết quả tái lập được
- Stack: Python (PyTorch), nghiên cứu · Task id prefix: `T` · GitHub, branch `main`
- Research project: yes · Gate command: (trống) · External tracker: none
- Roles: xem `.agents/roles.md`
- Python: `python3`


Nghiên cứu phát hiện mô tuyến giáp sót lại sau cắt tuyến giáp từ ảnh SPECT (DETR / Faster R-CNN / YOLOv7), định hướng Federated Learning. Luật vận hành chi tiết nằm ở `.agents/AGENTS.md` — đọc file đó đầu mỗi phiên; file này chỉ ghi những gì agent dễ sai/bỏ sót.

## Việc đầu phiên (tóm tắt)

Đọc `.agents/AGENTS.md` → `.agents/record.md` → `.agents/action-history.md` → `plan.csv` → `git status`. Thiếu file thì hỏi trước, KHÔNG tự bịa nội dung.

## Bản đồ thư mục — điều cần biết mà KHÔNG suy ra được từ tên file

| Path | Thực trạng |
|---|---|
| `uet-thyroid-detection-main/` | **Legacy, READ-ONLY** (D7). Không sửa/di chuyển; chỉ import model defs từ đó. Mọi file trong đó không được xuất hiện trong `git diff`. |
| `Train/` | Scaffold huấn luyện mới (T8, In progress). `scripts/train.py` mới chạy tới `NotImplementedError` ở `split.py` — **chưa train thật được**. `configs/faster_rcnn.yaml` để `null` đúng audit T5 (Faster-RCNN không có config gốc, không thể tái lập). |
| `Train/src` vs `uet-thyroid-detection-main/src` | Cả hai đều là package tên `src` → import `from src.detr...` trong `Train/src/models/*/__init__.py` có nguy cơ resolve nhầm âm thầm. CHƯA được quyết (init.md mục 4): đổi tên package hay `importlib` nạp theo đường dẫn. **Đừng viết wiring model thật trước khi vấn đề này được chốt.** |
| `Data/` | 3 định dạng đã convert: `detr_data/` (COCO JSON), `faster-rcnn-data/` (CSV, **đang chứa PII plaintext — chờ T3**), `yolov7-data/` (txt + cache). Split canonical T1/T2 **chưa làm** — DETR dùng split khác hẳn 2 model kia, 46% bệnh nhân bị leakage. **Không dùng split hiện tại để so sánh baseline mới.** |
| `Experiments/Re-create/Result/` | Kết quả train/eval đã commit (xem mục dưới). `Train/runs/` là nơi output mới (gitignored). |
| `plan.csv` | SSOT tiến độ. Agent tuyệt đối KHÔNG tự ghi `Status=Done` (chỉ hook/người dùng). Commit theo `[TaskID] <wip|done|blocked>: ...`. |

## Đọc kết quả train (`Experiments/Re-create/Result/`) và giải thích

Mỗi run là 1 thư mục `Experiments/Re-create/Result/<model>-<run>/`. Cấu trúc artifact khác nhau theo model:

- **DETR** `Experiments/Re-create/Result/detr-spect-*/`: `config.json` (argparse đầy đủ: lr, epochs=100, backbone, `brightness_levels=4`, split path), `log.jsonl` (1 dòng JSON/epoch: `train_loss`, `train_loss_ce/bbox/giou`, `train_lr`...), `baseline.json` (loss của tập val mỗi epoch — đọc để biết overfitting), `best.pth`/`last.pth`/`eval.pth`.
- **Faster R-CNN** `Experiments/Re-create/Result/frcnn-*/`: `config.json` (chỉ có `data_dir`, `epochs`, `batch_size`, `device` — **không có lr/augmentation**, đúng audit T5), `log.jsonl`, `checkpoint.pth`.
- **YOLOv7** `Experiments/Re-create/Result/yolov7-spect-*/`: `results.txt` (bảng/epoch: P, R, mAP@0.5, mAP@0.5:0.95; 2 epoch/dòng), `opt.yaml` + `hyp.yaml` + `data.yaml` (full hyperparams), `PR_curve.png`/`F1_curve.png`/`confusion_matrix.png`, `weights/{best,last,init,epoch_000}.pt`, `terminal.log`.
- **RR-HCL-SVM** `Experiments/Re-create/Result/test-hcl-svm/`: `metrics.json` (có `assumptions` + `exact_paper_reproduction: false` — **đọc kỹ trước khi trích dẫn số**), `config.json`, `predictions.csv` (`model,scan_id,group_id,split,y_true,status,score,y_pred`; split dùng `train_oof` = out-of-fold), `excluded_scans.csv`, `extraction_status.csv`, `manifest.csv`, `features.npz`.

Lưu ý khi giải thích:
- Mọi path tuyệt đối trong `config.json` (`/mnt/nvme2/users/utbt_sv1/...`) là đường dẫn trên máy train cũ — **không tồn tại local**, chỉ dùng để truy vết.
- DETR chọn `brighness_levels=4` (cũng cố ý sai chính tả như code gốc), Faster-RCNN=2, YOLOv7=3 trong `app.py` — baseline cũ KHÔNG control-variable, không so sánh trực tiếp với FL sau này (T6/T7).
- Metric quy ước D3: Thyroid mAP@0.5, Shoulder mAP@0.3.

## Bảo mật

Dữ liệu y tế SPECT/DICOM thật. Không dán PII hay dữ liệu bệnh nhân vào công cụ AI bên ngoài. CSV trong `Data/faster-rcnn-data/annotations/` hiện có tên + năm sinh plaintext (T3).
