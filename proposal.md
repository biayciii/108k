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

## 6. Domain Adaptation — chiến lược mô phỏng khi dữ liệu bổ sung hạn chế

### 6.0. Kiểm tra thực tế: dữ liệu hiện có KHÔNG có sẵn biến thiên thiết bị

Đã quét trực tiếp header DICOM (không suy đoán) trên mẫu ngẫu nhiên từ 471 ảnh hiện có: **100%
cùng một máy `GE Infinia` + trạm xử lý `Xeleris 3.1108`**. Tức là "domain shift do khác máy
quét/phần mềm" **không có sẵn** trong dữ liệu hiện tại — nếu muốn dùng lý do này phải tự tạo
proxy, không thể lấy miễn phí từ dữ liệu đang có.

### 6.1. Bối cảnh ràng buộc thực tế

Khả năng xin thêm dữ liệu thật từ bệnh viện khác là thấp; kịch bản khả thi nhất là xin thêm được
một lượng khiêm tốn (đủ nâng tổng dữ liệu lên khoảng 1.000 mẫu) rồi tự thiết kế mô phỏng domain
shift trên nền dữ liệu này. Ba phương án được xếp hạng theo độ tin cậy khoa học, có thể phối hợp:

**Bậc 1 (đáng tin nhất — chỉ cần một lượng nhỏ dữ liệu thật khác nguồn)**: nếu phần dữ liệu xin
thêm đến từ một viện khác (dù chỉ vài chục ca), **không trộn vào pool train chung** — dành riêng
làm tập test "unseen domain". Chỉ cần đủ để đo được "generalization gap" (mô hình global tụt bao
nhiêu khi gặp domain chưa từng thấy), không cần đủ lớn để học — đúng thiết kế kiểu FACMIC.

**Bậc 2 (nếu dữ liệu thêm vẫn cùng một viện — kịch bản nhiều khả năng xảy ra hơn)**: dùng
**covariate shift có căn cứ lâm sàng thật**, không phải nhiễu giả lập:
- `gap` = `study_datetime − treatment_datetime` (thời gian từ lúc uống I-131 đến lúc chụp) — đại
  lượng này làm thay đổi thật sự phân phối cường độ đếm phóng xạ do phân rã + đào thải sinh học.
  Chia client giả lập theo dải `gap` khác nhau tạo ra non-IID có ý nghĩa vật lý, không phải random
  split gắn mác "domain shift".
- Nếu dữ liệu mở rộng có thêm biến phân nhóm bệnh lý (loại ung thư, giai đoạn) — chia theo đó
  cũng là shift thật về mặt bệnh lý, đúng tinh thần paper D1 tự nêu ("mỗi quần thể có đặc trưng
  bệnh lý riêng").

**Bậc 3 (chỉ dùng làm ablation phụ, không phải luận điểm chính)**: tái dùng pipeline
`increase_count`/brightness-level đã có sẵn trong code (`src/{detr,faster_rcnn,yolov7}`) — gán
mỗi client giả lập một mức brightness cố định khác nhau để giả lập "độ nhạy máy quét khác nhau".
Đây là proxy nhân tạo, chỉ dùng để kiểm tra cơ chế của phương pháp DA hoạt động đúng (stress-test),
không dùng làm bằng chứng chính cho luận điểm domain shift.

### 6.2. Thiết kế thí nghiệm đề xuất

Kết hợp 2 lớp: (a) non-IID "thật" theo `gap`/protocol làm nền cho phần FL chính ở mục 5.3 — giải
quyết luôn điểm yếu "client giả lập bị chia IID giả tạo"; (b) nếu xin được dù chỉ một nhóm nhỏ ca
từ viện khác, dành riêng làm test set domain lạ, không gộp vào 1.000 mẫu chia train/test chung —
giá trị nằm ở việc **không train trên nó**, không nằm ở số lượng.

### 6.3. Ứng viên phương pháp DA

Chờ kích hoạt RQ2 khi có kết quả từ 6.1/6.2. Ứng viên đã review sẵn trong Ref.xlsx (mục 4): FedBN
(chi phí thấp nhất, ưu tiên thử trước), LC-Fed (khi cần cá nhân hoá sâu hơn), FACMIC (khi cần khả
năng tổng quát hoá sang viện hoàn toàn mới chưa tham gia huấn luyện).

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

## 9. Định hướng nâng cao chất lượng công bố (rank tạp chí/hội nghị)

Vì FedAvg/FedProx tự thân không còn là novelty thuật toán, trần rank công bố được nâng bằng cách
đóng khung nghiên cứu theo bài toán nghiệp vụ/thực tế thay vì thuần thuật toán. Bốn hướng đã chốt
dùng, mỗi hướng kèm điều kiện/khảo sát cần hoàn tất trước khi đưa vào bản thảo:

### 9.1. Đóng khung theo ràng buộc pháp lý/vận hành thật (đã khảo sát)

Đã tra cứu trực tiếp văn bản pháp luật hiện hành của Việt Nam (không suy đoán), dùng làm căn cứ
pháp lý chính thức cho việc dữ liệu SPECT/DICOM bệnh nhân không được rời khỏi bệnh viện dưới dạng
tập trung hoá — đây chính là lý do tồn tại thật của FL, không phải giả định cho có:

- **Luật Khám bệnh, chữa bệnh 2023 (Luật số 15/2023/QH15), Điều 69**: người hành nghề và cơ sở
  khám chữa bệnh có nghĩa vụ giữ bí mật tình trạng bệnh, thông tin người bệnh cung cấp và hồ sơ
  bệnh án, trừ khi người bệnh đồng ý chia sẻ hoặc thuộc trường hợp luật định (khoản 3, 4 Điều 69);
  hồ sơ bệnh án phải có cơ chế bảo mật và kiểm soát truy cập.
- **Nghị định 13/2023/NĐ-CP** (hiệu lực từ 01/7/2023) — văn bản đầu tiên của Việt Nam quy định
  toàn diện về bảo vệ dữ liệu cá nhân: xếp tình trạng sức khoẻ/hồ sơ bệnh án vào nhóm **dữ liệu cá
  nhân nhạy cảm**; yêu cầu đánh giá tác động khi thu thập/xử lý (Điều 24) và triển khai biện pháp
  bảo vệ, quy chế nội bộ, phân định trách nhiệm rõ ràng (Điều 27, 28).

→ Hai căn cứ này thay thế hoàn toàn cách viết motivation chung chung kiểu "privacy is important"
bằng nghĩa vụ pháp lý cụ thể mà 108 Military Central Hospital đang phải tuân thủ.

### 9.2. Đóng góp benchmark/reproducibility (đã chốt dùng)

Giữ nguyên như mục 2 — biến toàn bộ phần Data & Training-Protocol Audit (leakage, split lệch giữa
3 định dạng, thiếu cấu hình Faster-RCNN) thành một đóng góp độc lập có thể trích dẫn, tiếp nối
đúng tinh thần dataset/benchmark paper của D1.

### 9.3. Vòng validation lâm sàng thật (đã chốt dùng — cần chẩn đoán thật)

So sánh kết quả centralized/FL với chẩn đoán của bác sĩ thật trên cùng ca (theo đúng cách D4 đã
làm với 2 bác sĩ). **Điều kiện cần**: đây là yêu cầu **chẩn đoán thật** từ bác sĩ chuyên khoa —
chưa có sẵn trong phạm vi hiện tại của repo, cần chủ động thu xếp với 108 Military Central
Hospital trước khi đưa vào kế hoạch thí nghiệm chính thức.

### 9.4. Đóng khung theo hạ tầng thực tế tuyến dưới (đã chốt dùng — cần khảo sát phần cứng)

Đo chi phí communication/compute thực tế của FedAvg/FedProx trong điều kiện phần cứng khiêm tốn,
đúng thực trạng các viện tuyến dưới. **Điều kiện cần**: cần khảo sát phần cứng thực tế tại các
viện mục tiêu — hiện tại **giả định tạm thời chỉ có CPU nhiều nhân, không có GPU**, cho đến khi
khảo sát xong và có số liệu thật để thay thế giả định này.

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

**Căn cứ pháp lý (mục 9.1)**:
- Luật Khám bệnh, chữa bệnh 2023 (Luật số 15/2023/QH15): https://xaydungchinhsach.chinhphu.vn/toan-van-luat-15-2023-qh15-kham-benh-chua-benh-119231127164453959.htm
- Nghị định 13/2023/NĐ-CP về bảo vệ dữ liệu cá nhân: https://xaydungchinhsach.chinhphu.vn/toan-van-nghi-dinh-13-2023-nd-cp-bao-ve-du-lieu-ca-nhan-119230516104357809.htm
