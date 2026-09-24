# Đề cương nghiên cứu: Federated Learning cho Phát hiện Mô Tuyến giáp Còn sót trên Ảnh SPECT

## 1. Bối cảnh & Động lực

Chuỗi 5 nghiên cứu gốc của nhóm (xem `original paper/`, tóm tắt quyết định D1–D5 trong
`.agents/record.md`) đã đi qua bốn giai đoạn liên tiếp:

1. **Thu thập & chuẩn hoá dữ liệu** — D1, *New thyroid scintigraphy datasets: Construction and
   benchmark assessment* (2023): xây hai bộ dữ liệu chuẩn hoá từ 559 ảnh SPECT của Bệnh viện
   TWQĐ 108 (2020–2021), benchmark bằng transfer learning trên nhiều CNN pretrained.
2. **Detection** — D2 (*A deep learning method using SPECT images...*, NICS 2022, phân loại
   bằng CNN fine-tune) và D3 (*Utilizing DETR model on SPECT image...*, SSP 2023, chuyển sang
   object detection bằng DETR/Faster-RCNN/YOLOv7 + định nghĩa chỉ số RSI).
3. **Tăng tính giải thích** — D4, *RR-HCL-SVM: A Two-Stage Framework...* (IJIST 2024): kết hợp
   RSI với 83 đặc trưng radiomics, giảm chiều bằng clustering, phân loại bằng SVM.
4. **Ứng dụng lâm sàng** — D5, *Ablation dosage recommendation...* (2024): Decision Tree (C4.5)
   khuyến nghị liều I-131 từ RSI + dữ liệu lâm sàng, ưu tiên explainability.

Cả 5 nghiên cứu đều dùng dữ liệu từ **một bệnh viện duy nhất**. Paper D1 tự thừa nhận hai giới
hạn cốt lõi chưa từng được giải quyết trong toàn bộ chuỗi nghiên cứu:

> *"sharing this dataset with other researchers has been limited"* — dữ liệu độc quyền, khó chia
> sẻ liên viện.
>
> *"each population group presents unique pathological characteristics on SPECT"* — mỗi quần
> thể/dân số có đặc trưng bệnh lý riêng trên ảnh SPECT.

Hai phát biểu này chính là động lực cho hướng nghiên cứu tiếp theo: **Federated Learning (FL)**
để nhiều bệnh viện có thể cùng huấn luyện mô hình mà không cần tập trung hoá dữ liệu nhạy cảm, và
**Domain Adaptation (DA)** để xử lý lệch phân phối giữa các viện/máy quét khi FL được mở rộng ra
đa viện thật.

**Quyết định phạm vi**: triển khai **FL trước, DA sau** — vì hiện chỉ có dữ liệu một viện nên
domain shift thật giữa các viện chưa thể kiểm chứng; DA sẽ chờ đến khi có dữ liệu đa viện thật
hoặc có cách mô phỏng domain shift hợp lệ.

## 2. Phát hiện từ Data & Training-Protocol Audit

Trước khi thiết kế thí nghiệm FL, nhóm đã đối chiếu trực tiếp `Data/{detr_data,faster-rcnn-data,
yolov7-data}` (so khớp theo DICOM UID) và cấu hình huấn luyện của 3 kiến trúc. Bốn phát hiện dưới
đây bắt buộc phải xử lý trước, vì nếu không thì bất kỳ so sánh centralized-vs-FL nào sau này cũng
đã sai từ gốc:

### 2.1. Tổng số mẫu thực chất khác với con số các paper báo cáo

| Nguồn | Train | Val | Test | Tổng |
|---|---|---|---|---|
| Faster-RCNN (`Data/faster-rcnn-data`) | 329 | 94 | 47 | 470 |
| YOLOv7 (`Data/yolov7-data`) | 329 | 94 | 47 | 470 |
| DETR (`Data/detr_data`) | 330 | 94 | 47 | 471 |
| **Union thực tế (không trùng lặp)** | | | | **471 ảnh DICOM duy nhất** |

Con số này khác với 559 ảnh mà paper D1 báo cáo đã dùng, và khác với 396 ảnh (278/79/39) mà
paper D3 báo cáo cho thí nghiệm DETR — tức là bộ dữ liệu hiện có trong `Data/` **không phải bản
dùng để tạo ra các con số đã công bố**, và không thể tái lập chính xác kết quả của D1/D3 từ dữ
liệu hiện tại.

### 2.2. Ba định dạng dữ liệu không dùng chung một cách chia train/val/test

Faster-RCNN và YOLOv7 dùng đúng một pool 470 ảnh với **split giống hệt nhau** (khớp 100% theo
từng ID). DETR dùng một pool khác (471 ảnh) với **split hoàn toàn độc lập**: chỉ 7/47 ảnh trong
test set của DETR cũng nằm trong test set Faster-RCNN/YOLOv7; 29/47 ảnh "test" của DETR thực ra
là ảnh mà Faster-RCNN/YOLOv7 đã dùng để **train**. Do đó, các con số so sánh mAP/F1 giữa 3 kiến
trúc trong paper D3 (căn cứ để chọn DETR làm detector chính) **không được đo trên cùng một test
set** — đây là một confound thực sự trong nghiên cứu gốc.

### 2.3. Rò rỉ dữ liệu ở mức bệnh nhân (patient-level leakage)

Cột `name` trong CSV Faster-RCNN cho thấy 470 ảnh thuộc về 216 bệnh nhân (~2,2 ảnh/bệnh nhân).
Trong đó **100/216 bệnh nhân (46%)** có ảnh xuất hiện ở nhiều hơn một tập (train + val, hoặc
train + test, hoặc cả ba). Đây là lỗi patient-level leakage kinh điển trong ML y tế: mô hình có
thể học đặc điểm riêng của bệnh nhân lúc train rồi được đánh giá lại trên đúng bệnh nhân đó ở
test, làm thổi phồng các chỉ số hiệu năng đã báo cáo.


### 2.4. Cấu hình huấn luyện giữa 3 kiến trúc không đồng nhất

- YOLOv7 có `opt.yaml`/`hyp.scratch.p5.yaml` ghi rõ (300 epoch, batch 8, ảnh 640×640, toàn bộ
  augmentation hình học — rotate/scale/shear/mosaic/mixup — bị tắt).
- DETR chỉ có giá trị mặc định trong `argparse` (`main.py`: epochs=300, batch_size=2), không có
  file cấu hình được lưu lại của lần train tạo ra checkpoint đang dùng trong `app.py`.
- **Faster-RCNN hoàn toàn không có cấu hình huấn luyện nào được lưu lại** — không có thư mục
  `configs/` Hydra, không có `train.py` nào trong `src/faster_rcnn/`. Epoch/batch/learning-rate
  đã tạo ra checkpoint `.ckpt` hiện có không thể xác minh lại.
- Mỗi kiến trúc còn được train riêng ở nhiều "brightness level" (1–5, xem bảng Model logs trong
  README gốc) và chọn checkpoint tốt nhất khác nhau cho từng model (`frcnn=2, detr=4, yolov7=3`
  trong `app.py`) — đây là một biến không được kiểm soát thống nhất giữa các model.

→ Nếu không khoá lại cả dữ liệu lẫn cấu hình huấn luyện, mọi khác biệt hiệu năng đo được giữa
centralized và FL sau này đều có thể chỉ là do những confound trên, không phải do bản thân FL.

## 3. Vấn đề nghiên cứu

- **(chính, giai đoạn này)**: Khi mô phỏng phân mảnh dữ liệu 471 ảnh thành nhiều client giả
  lập, liệu Federated Learning (FedAvg/FedProx) có giữ được hiệu năng detection (mAP@0.5) và
  chẩn đoán qua RSI (F1/Sensitivity/Specificity) tương đương với huấn luyện tập trung
  (centralized), trên cùng một split dữ liệu và cùng một cấu hình huấn luyện đã chuẩn hoá?
- **(để ngỏ cho giai đoạn sau)**: Khi có thêm dữ liệu và một phương án mô phỏng domain shift được kiểm chứng hợp lệ, kỹ thuật Domain Adaptation nào (chuẩn hoá cục bộ kiểu
  FedBN, hiệu chỉnh cục bộ kiểu LC-Fed, hay domain generalization kiểu FACMIC) phù hợp nhất để
  giữ hiệu năng khi mở rộng FL ra nhiều bệnh viện có đặc trưng ảnh khác nhau?

RQ2 không được triển khai trong giai đoạn này — chỉ được nêu ra và chuẩn bị sẵn ứng viên phương
pháp (mục 6).

## 4. Related Work

**(i) Federated Learning cho ảnh y tế**
- *Federated Learning for Thyroid Ultrasound Image Analysis to Protect Personal Information*
  — FL thật trên dữ liệu tuyến giáp đa viện (8,4k ảnh, 5 kiến trúc CNN), chứng minh FL đạt hiệu
  năng gần với centralized dù tốn thời gian train hơn.
- *Federated Optimization in Heterogeneous Networks* (FedProx) — thêm proximal term xử lý
  non-IID (statistical heterogeneity) và system heterogeneity so với FedAvg gốc.

**(ii) Federated Learning + Domain Shift** (chuẩn bị cho RQ2, chưa triển khai)
- *FedBN: Federated Learning on Non-IID Features via Local Batch Normalization* — chỉ chia sẻ
  trọng số Conv/Linear, giữ riêng thống kê BatchNorm cục bộ từng client. Nhóm đã tự đánh giá đây
  là phương pháp **"sát nhất về phương pháp & loại dữ liệu SPECT"** trong Ref.xlsx.
- *Personalizing Federated Medical Image Segmentation via Local Calibration* (LC-Fed) — tách
  nhánh biểu diễn dùng chung khỏi nhánh hiệu chỉnh cục bộ riêng từng client.
- *FACMIC: Federated Adaptative CLIP Model for Medical Image Classification* — federated
  domain generalization, kiểm tra trên 1 client đóng vai trò unseen target domain.

**(iii) Bảo mật bổ sung** (hướng xa hơn, không phải phạm vi giai đoạn này)
- *CryptoNets*, *Privacy-Preserving Federated Vision Transformer Learning... Homomorphic
  Encryption* — mã hoá đồng cấu cho suy luận/gradient, tăng chi phí tính toán đáng kể, chưa cần
  thiết ở quy mô proof-of-concept hiện tại.

## 5. Phương pháp đề xuất 

### 5.1. Data Audit & Re-split
- Patient-level GroupKFold trên 471 ảnh (nhóm theo bệnh nhân, không theo ảnh) để loại bỏ hoàn
  toàn leakage mô tả ở mục 2.3.
- Hợp nhất thành **một split canonical duy nhất**, áp dụng lại cho cả 3 định dạng dữ liệu
  (`detr_data`, `faster-rcnn-data`, `yolov7-data`) thay vì 3 split độc lập như hiện tại.
- Loại bỏ/pseudonymize cột `name` và năm sinh khỏi mọi file CSV.

### 5.2. Chuẩn hoá cấu hình huấn luyện (training protocol)
- Trích xuất và ghi lại tường minh cấu hình hiện có của YOLOv7 (`opt.yaml`, `hyp.scratch.p5.yaml`)
  và DETR (`argparse` trong `main.py`).
- Tái tạo cấu hình huấn luyện còn thiếu cho Faster-RCNN.
- Chốt **một bộ control-variables** dùng chung cho cả 3 kiến trúc và cho mọi thí nghiệm
  centralized/FL sau này: brightness level cố định (không để mỗi model tự chọn mức riêng), cùng
  kích thước ảnh đầu vào, cùng chính sách augmentation trong phạm vi kiến trúc cho phép, cùng số
  epoch tương đương (quy đổi hợp lý giữa epoch centralized và local-epoch × round của FL).
- Lưu thành một file cấu hình duy nhất cho mỗi kiến trúc, làm nguồn tham chiếu bắt buộc cho mọi
  lần chạy thí nghiệm.

### 5.3. Federated simulation
- Partition 471 ảnh (đã patient-level split) thành N client giả lập — ví dụ theo lô thời gian
  chụp (`study_datetime`) hoặc chia ngẫu nhiên có kiểm soát mức độ non-IID nhẹ. Ràng buộc bắt
  buộc: **không cắt đôi cùng một bệnh nhân giữa hai client**.
- Cài đặt FedAvg làm baseline bắt buộc cho cả 3 kiến trúc detection.
- Cài đặt FedProx, so sánh với FedAvg dưới điều kiện non-IID do partition tạo ra.
- Đánh giá FL vs centralized (đã sửa leakage, cùng cấu hình huấn luyện) trên cùng bộ metric.

## 6. Domain Adaptation  (chưa triển khai)

Chờ một trong hai điều kiện: (a) có thêm dữ liệu SPECT thật, hoặc (b)
có phương án mô phỏng domain shift được kiểm chứng là hợp lệ trên dữ liệu hiện có (ví dụ: phân
biệt theo lô máy quét/thời kỳ hiệu chuẩn nếu metadata DICOM cho phép). Ứng viên phương pháp đã
review sẵn trong Ref.xlsx (mục 4) sẽ được đánh giá lại khi RQ2 được kích hoạt: FedBN (chi phí
thấp nhất, ưu tiên thử trước), LC-Fed (khi cần cá nhân hoá sâu hơn), FACMIC (khi cần khả năng
tổng quát hoá sang viện hoàn toàn mới chưa tham gia huấn luyện).

## 7. Kế hoạch đánh giá

Giữ nguyên các metric đã dùng trong chuỗi nghiên cứu gốc để đảm bảo có thể so sánh xuyên suốt:
- **Detection**: mAP@0.5 (thyroid), mAP@0.3 (shoulder, theo quy ước D3).
- **Chẩn đoán qua RSI**: Precision, Recall, F1, Accuracy — so sánh centralized vs FedAvg vs
  FedProx trên cùng split canonical (mục 5.1) và cùng cấu hình (mục 5.2).

## 8. Rủi ro & giới hạn

- Cỡ mẫu vốn đã nhỏ (471 ảnh / 216 bệnh nhân); chia nhỏ tiếp cho nhiều client mô phỏng sẽ làm
  từng client có rất ít dữ liệu — cần báo cáo khoảng tin cậy, không chỉ điểm số trung bình.
- Đây là **mô phỏng multi-client trên dữ liệu một viện**, chưa phải triển khai đa viện thật —
  kết quả chứng minh tính khả thi hạ tầng, chưa phải bằng chứng lâm sàng.
- Việc chuẩn hoá cấu hình huấn luyện có thể làm thay đổi (tăng hoặc giảm) hiệu năng centralized
  baseline so với con số đã công bố trong D1–D5 — đây là điều được chủ đích chấp nhận để đảm bảo
  so sánh công bằng, cần nêu rõ trong phần Discussion khi công bố kết quả.

---

## Phụ lục: Ánh xạ tài liệu tham khảo

| Ký hiệu | Paper | Vai trò trong đề cương |
|---|---|---|
| D1 | New thyroid scintigraphy datasets... (2023) | Nguồn dữ liệu gốc + động lực FL/DA |
| D2 | A deep learning method using SPECT images... (NICS 2022) | Bối cảnh detection giai đoạn 1 |
| D3 | Utilizing DETR model on SPECT image... (SSP 2023) | Nguồn 3 kiến trúc detection + RSI hiện có |
| D4 | RR-HCL-SVM: A Two-Stage Framework... (IJIST 2024) | Nguồn hướng "tính giải thích" |
| D5 | Ablation dosage recommendation... (2024) | Nguồn hướng "ứng dụng" |

Chi tiết Context–Decision–Rejected alternatives–Consequences của D1–D5: xem `.agents/record.md`
mục 2. Danh sách đầy đủ tài liệu FL/DA đã review: `Docs/Ref.xlsx`.
