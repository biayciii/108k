# Outline — Federated Learning cho Phát hiện Mô Tuyến giáp Còn sót trên Ảnh SPECT

> Dàn ý heading-only, khớp 1-1 với `proposal.md`. Mỗi mục ghi 1 dòng nội dung dự kiến + nguồn
> tham chiếu. Dùng file này khi mở rộng `proposal.md` thành bản thảo bài báo/luận văn đầy đủ.

## 1. Bối cảnh & Động lực
- Nội dung dự kiến: tóm tắt chuỗi D1→D5, trích nguyên văn 2 giới hạn D1 tự nêu (dữ liệu độc
  quyền + domain shift theo quần thể), nêu quyết định "FL trước, DA sau".
- Nguồn: `.agents/record.md` mục 2 (D1–D5).

## 2. Phát hiện từ Data & Training-Protocol Audit
### 2.1. Tổng số mẫu thực chất (471 ảnh) vs con số các paper báo cáo (559 / 396)
- Nguồn: kết quả đối chiếu `Data/{detr_data,faster-rcnn-data,yolov7-data}` trong phiên audit.
### 2.2. Ba định dạng dữ liệu không dùng chung một split
- Nguồn: bảng so khớp ID theo split, phiên audit.
### 2.3. Patient-level leakage (100/216 bệnh nhân)
- Nguồn: cột `name` trong `Data/faster-rcnn-data/annotations/*.csv`, phiên audit.
### 2.4. Dữ liệu định danh trực tiếp còn tồn tại (tên, năm sinh)
- Nguồn: `Data/faster-rcnn-data/annotations/*.csv`; đối chiếu `.agents/AGENTS.md` mục bảo mật.
### 2.5. Cấu hình huấn luyện không đồng nhất giữa 3 kiến trúc
- Nguồn: `uet-thyroid-detection-main/{runs/train/*/opt.yaml, hyp.scratch.p5.yaml}`,
  `src/detr/main.py` (argparse), thiếu `configs/`+`train.py` cho `src/faster_rcnn/`.

## 3. Vấn đề nghiên cứu
### 3.1. RQ1 — FL vs centralized trên dữ liệu mô phỏng multi-client (giai đoạn này)
### 3.2. RQ2 — Domain Adaptation khi mở rộng đa viện thật (để ngỏ, giai đoạn sau)
- Nguồn: quyết định chốt với người dùng trong phiên lập kế hoạch.

## 4. Related Work
### 4.1. Federated Learning cho ảnh y tế
- FL cho ảnh siêu âm tuyến giáp đa viện; FedProx (heterogeneity).
### 4.2. Federated Learning + Domain Shift (chuẩn bị cho RQ2)
- FedBN (ưu tiên — đã tự đánh giá "sát nhất"), LC-Fed, FACMIC.
### 4.3. Bảo mật bổ sung (hướng xa hơn RQ1/RQ2)
- CryptoNets, Homomorphic Encryption cho FL/ViT.
- Nguồn: `Docs/Ref.xlsx`.

## 5. Phương pháp đề xuất (giai đoạn này)
### 5.1. Data Audit & Re-split
- Patient-level GroupKFold, hợp nhất split canonical, pseudonymize CSV.
### 5.2. Chuẩn hoá cấu hình huấn luyện
- Ghi lại cấu hình YOLOv7/DETR hiện có; tái tạo cấu hình Faster-RCNN còn thiếu; chốt bộ
  control-variables dùng chung (brightness level, image size, augmentation, epoch/round).
### 5.3. Federated simulation
- Partition N-client giả lập (không cắt đôi bệnh nhân); FedAvg baseline; FedProx; so sánh
  FL vs centralized trên cùng split + cấu hình.

## 6. Domain Adaptation — hướng mở rộng tương lai (chưa triển khai)
- Điều kiện kích hoạt RQ2 (dữ liệu đa viện thật / mô phỏng domain shift hợp lệ).
- Thứ tự ưu tiên thử nghiệm dự kiến: FedBN → LC-Fed → FACMIC.

## 7. Kế hoạch đánh giá
- Metric detection: mAP@0.5 (thyroid), mAP@0.3 (shoulder) — theo quy ước D3.
- Metric chẩn đoán: Precision/Recall/F1/Accuracy qua RSI.

## 8. Rủi ro & giới hạn
- Cỡ mẫu nhỏ khi chia client; đây là mô phỏng chưa phải đa viện thật; baseline centralized có
  thể đổi so với số đã công bố sau khi chuẩn hoá dữ liệu + cấu hình.

## Phụ lục
- Bảng ánh xạ D1–D5 ↔ vai trò trong đề cương.
- Tham chiếu đầy đủ: `.agents/record.md` (mục 2), `Docs/Ref.xlsx`.
