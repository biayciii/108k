# record.md — 108 Thyroid

File duy nhất agent chủ động cập nhật, đọc đầu tiên mỗi phiên. Không xóa entry cũ — đổi ý
về một quyết định thì thêm entry mới, ghi rõ "thay thế entry #N".

## 1. Bản đồ

| File/thư mục | Vai trò | Ai cập nhật |
|---|---|---|
| `README.md` (root) | Tài liệu kỹ thuật: tiền xử lý ảnh SPECT/DICOM + định dạng nhãn cho DETR/Faster-RCNN/YOLOv7 | Người dùng |
| `original paper/` | 5 bài báo gốc (PDF) của nhóm nghiên cứu — nguồn quyết định/bài học ở mục 2 | Người dùng |
| `Docs/Ref.xlsx` | Bảng theo dõi tài liệu tham khảo (lấy từ Google Sheet `108 K/Docs/Ref.xlsx` trên Drive) — tóm tắt + note cho 5 paper gốc và ~19 paper/git-repo khác đang được rà soát cho hướng nghiên cứu tiếp theo (privacy-preserving FL, MAE/representation learning, synthetic data, domain adaptation cho ảnh tuyến giáp) | Người dùng |
| `uet-thyroid-detection-main/` | Code gốc: `app.py` (demo Gradio 3 model), `src/{detr,faster_rcnn,yolov7}/`, `data/`, `assets/`, `draft.ipynb`, `logistic_model.pkl`, `yolov7.pt` | Agent (đang chờ reorg — xem plan.csv T1/T2) |
| `Data/` | Dữ liệu đã convert sẵn theo 3 định dạng: `detr_data/` (COCO JSON), `faster-rcnn-data/` (CSV), `yolov7-data/` (YOLO txt + hyp/cfg) | Người dùng |
| `plan.csv` | SSOT tiến độ (root repo) — viết lại theo roadmap D6 (GĐ0 Data & Training-Protocol Audit → GĐ1 Federated Learning → GĐ2 Domain Adaptation để ngỏ) | Agent (qua post-commit hook) |
| `proposal.md` (root repo) | Đề cương học thuật hướng Federated Learning (D6) | Agent, review bởi người dùng |
| `outline.md` (root repo) | Dàn ý chi tiết khớp 1-1 với `proposal.md` | Agent, review bởi người dùng |
| `.agents/AGENTS.md` | Luật hiện hành | Agent |
| `.agents/hooks/` | `post-commit.sh` + `_post_commit.py` (logic) + `install.sh` | Agent |

## 2. Quyết định

Khung bắt buộc: **Context — Decision — Rejected alternatives — Consequences**. 5 entry dưới
đây được trích từ 5 bài báo gốc trong `original paper/` (đúng nội dung bài báo, không suy
diễn thêm).

### D1 — New thyroid scintigraphy datasets: Construction and benchmark assessment (2023)
- **Context**: Không có bộ dữ liệu SPECT tuyến giáp chuẩn, công khai để so sánh các phương pháp CADx; dữ liệu hiện có (Yinxiang Guo et al., 446 ca) không được chia sẻ.
- **Decision**: Xây 2 bộ dữ liệu chuẩn từ 559 ảnh của Bệnh viện TWQĐ 108 (2020-2021): Full Body Dataset và Neck Region Dataset (crop vùng cổ); benchmark bằng transfer learning trên các CNN pretrained (VGG16, Inception-v3, ResNet, Xception, MobileNet, NASNetMobile, EfficientNet).
- **Rejected alternatives**: Không đề cập loại bỏ kiến trúc cụ thể — mục tiêu là benchmark khách quan toàn bộ các model trên.
- **Consequences**: Ảnh crop vùng cổ cho kết quả tốt hơn ảnh toàn thân ở hầu hết chỉ số (VGG16/Xception đạt Acc 0.955, F1 0.961 trên tập neck vs Acc~0.89 trên tập full-body) — vì vùng cổ loại bỏ nhiễu hấp thu ở bụng. Có bias nhẹ về nhãn "radioiodine uptake" do mất cân bằng lớp.

### D2 — A deep learning method using SPECT images to diagnose remaining thyroid tissue post-thyroidectomy (NICS 2022)
- **Context**: Chẩn đoán dựa kinh nghiệm bác sĩ đọc ảnh SPECT thang xám không nhất quán; nghiên cứu trước (Guo, ResNet18) không xét ảnh hưởng chất lượng ảnh.
- **Decision**: Fine-tune CNN pretrained (ResNet50, VGG16, GoogLeNet, DenseNet, Inception-ResNet-V2, EfficientNet) trên ảnh 2 kênh ANT+POST đã crop + cân bằng histogram; dùng focal loss vì mất cân bằng dữ liệu; tách riêng tập ca dễ (ANT-POST) và ca khó (ANT-HN-POST).
- **Rejected alternatives**: VGG16 bị loại vì kết quả rất kém (SEN=0, AUC=0.5). ResNet50 được chọn làm đề xuất chính vì AUC cao và ổn định nhất trên cả 2 tập.
- **Consequences**: Tập dễ: Acc 87%, AUC 0.93. Tập khó: Acc 74%, AUC 0.79 — giảm mạnh do ảnh tương phản kém, mô giáp mờ khó định vị.

### D3 — Utilizing DETR model on SPECT image to assess remaining thyroid tissues (SSP 2023)
- **Context**: CNN-based detector (Faster-RCNN, YOLOv7) hạn chế do pooling nhiều lớp làm mất tương quan không gian tổng thể, chỉ giữ đặc trưng cục bộ.
- **Decision**: Dùng DETR (Transformer-based) để detect bbox vùng giáp/vai; định nghĩa chỉ số RSI (tỷ lệ hấp thu vùng cổ/vai) từ bbox dự đoán, phân loại còn/hết mô giáp bằng logistic regression; tiền xử lý gồm crop đầu-cổ + tăng cường đa mức độ sáng (chọn 3 mức, đánh đổi accuracy/tốc độ).
- **Rejected alternatives**: Loại Faster-RCNN và YOLOv7 khỏi lựa chọn detector chính vì mAP@0.5 thấp hơn (hạn chế nắm bắt quan hệ không gian).
- **Consequences**: Chẩn đoán qua RSI đạt Precision 0.86, Recall 0.95, Acc 0.96, F1 0.91 — vượt baseline ResNet-50 trước đó (F1 0.88). DETR: Thyroid mAP@0.5=0.73, Shoulder mAP@0.3=0.604, tốt hơn 2 model kia.

### D4 — RR-HCL-SVM: A Two-Stage Framework for Assessing Remaining Thyroid Tissue (IJIST 2024)
- **Context**: Phương pháp trước (Guo, ResNet-18 + HE + GrabCut) không giữ được độ chính xác trên dữ liệu nhiễu/thực tế hơn; tuyến nước bọt dễ gây nhầm với vùng cổ.
- **Decision**: Framework 2 giai đoạn — (1) DETR pretrained detect bbox cổ/vai; (2) tính RSI + 83 đặc trưng radiomics (texture/shape/intensity) từ ROI, giảm chiều bằng clustering (Spectral/Hierarchical dựa trên Pearson correlation, ngưỡng 0.9) trước khi phân loại bằng SVM/Logistic Regression.
- **Rejected alternatives**: GrabCut / dùng ảnh gốc trực tiếp cho kết quả dưới tối ưu. So sánh RSI-only (F1 0.93) và RR-no-clustering (F1 0.96) — cả hai bị vượt bởi SVM RR + Hierarchical clustering (F1 0.97), được chọn làm cấu hình cuối.
- **Consequences**: AUC 0.995, F1 0.97, SEN 0.96, SPEC 0.98; giảm >80% (có cấu hình 90%) số chiều đặc trưng mà không giảm đáng kể độ chính xác. So với 2 bác sĩ thật: mô hình vượt trội ở F1 lớp Nonresidual (0.89) nhưng thua 1 bác sĩ về Recall lớp Residual.

### D5 — Ablation dosage recommendation for thyroid cancer treatment using machine learning (2024)
- **Context**: Chọn liều I-131 (50/75/100 mCi) sau phẫu thuật hiện dựa kinh nghiệm bác sĩ, dễ quá liều/thiếu liều.
- **Decision**: Dùng Decision Tree (C4.5) trên 42 đặc trưng lâm sàng (T/M/N, nguy cơ tái phát, kích thước khối u, Tg, TSH, ATg, RSI...) đã chuẩn hóa, ưu tiên explainability cho quyết định y khoa hơn là hiệu năng thuần túy.
- **Rejected alternatives**: Nearest Neighbors, Linear SVM, MLP, AdaBoost đều có F1 thấp hơn DT-C4.5 (đặc biệt mức 75MCI: DT 0.74 vs AdaBoost 0.44, KNN 0.39). Random Forest không được chọn vì đánh đổi khả năng diễn giải lấy hiệu năng.
- **Consequences**: Acc tổng thể 85%; F1 = 0.82/0.74/0.84 cho 50/75/100 MCI (mức 75MCI kém hơn do ít mẫu). Permutation importance cho thấy chỉ vài đặc trưng (TumorSize, MedRisk, Pregnant, Tg, HighRisk) thực sự ảnh hưởng dù dùng tới 42 đặc trưng đầu vào.

### D6 — Pivot sang Federated Learning trước, Domain Adaptation sau (quyết định nhóm, không phải trích từ paper)
- **Context**: Dữ liệu SPECT tuyến giáp là dữ liệu y tế nhạy cảm, hiện chỉ có từ 1 bệnh viện
  (108 Military Central Hospital). Paper D1 tự nêu 2 giới hạn chưa giải quyết: dữ liệu độc quyền
  khó chia sẻ liên viện, và mỗi quần thể có đặc trưng bệnh lý riêng trên SPECT (domain shift).
  Audit trực tiếp `Data/{detr_data,faster-rcnn-data,yolov7-data}` trong phiên này còn phát hiện
  thêm: tổng thực chất chỉ 471 ảnh DICOM (không phải 559 như D1 hay 396 như D3 báo cáo); DETR
  dùng split khác hẳn Faster-RCNN/YOLOv7 (chỉ 7/47 ảnh test trùng nhau); 100/216 bệnh nhân (46%)
  bị patient-level leakage giữa các split; CSV Faster-RCNN chứa tên thật + năm sinh bệnh nhân
  plaintext; cấu hình huấn luyện không đồng nhất giữa 3 kiến trúc (Faster-RCNN không có
  configs//train.py nào được lưu lại).
- **Decision**: Triển khai Federated Learning (FL) trước bằng cách mô phỏng multi-client trên
  dữ liệu 1 viện hiện có (chưa có dữ liệu đa viện thật); Domain Adaptation (DA) để sau, chờ dữ
  liệu đa viện thật hoặc phương án mô phỏng domain shift được kiểm chứng hợp lệ. Trước khi làm
  FL, bắt buộc phải: (1) patient-level re-split + hợp nhất 1 split canonical dùng chung cho cả 3
  định dạng, (2) pseudonymize dữ liệu định danh, (3) chuẩn hoá & ghi lại cấu hình huấn luyện
  (control-variables) dùng chung cho cả 3 kiến trúc — để tách được "khác biệt do FL" khỏi
  "khác biệt do dữ liệu/cấu hình train không đồng nhất". FL áp dụng cho cả 3 kiến trúc
  (Faster-RCNN/DETR/YOLOv7) để giữ tính so sánh của nghiên cứu gốc; baseline bắt buộc là FedAvg,
  có so sánh thêm FedProx.
- **Rejected alternatives**: Làm DA trước FL — bị loại vì domain shift thật giữa các viện chưa
  thể kiểm chứng khi chỉ có dữ liệu 1 viện. Chỉ FL cho 1 kiến trúc (DETR, theo D3 đã chọn là
  detector tốt nhất) — bị loại để giữ khả năng so sánh 3 kiến trúc như nghiên cứu gốc. Giữ
  nguyên split/cấu hình cũ và làm FL ngay — bị loại vì baseline centralized để so sánh sẽ sai từ
  gốc do leakage + cấu hình không đồng nhất đã phát hiện.
- **Consequences**: Thêm hẳn 1 giai đoạn GĐ0 (Data & Training-Protocol Audit, 7 task) chặn trước
  GĐ1 (Federated Learning, 6 task) trong `plan.csv`; baseline centralized sẽ phải đo lại từ đầu
  (không dùng được số liệu D1–D5 cũ để so sánh); `proposal.md`/`outline.md` được viết lại theo
  roadmap này (xem `proposal.md`, `outline.md` ở root repo).

**Addendum D6 (phiên sau, ngày 2026-09-24)** — bổ sung vào `proposal.md` mục 6 và mục 9 mới:
- **Kiểm tra thật**: quét header DICOM trên mẫu ngẫu nhiên từ 471 ảnh hiện có → 100% cùng máy
  `GE Infinia` + trạm `Xeleris 3.1108`. Kết luận: không có sẵn domain shift do khác thiết bị
  trong dữ liệu hiện tại; phải tự tạo proxy nếu muốn dùng luận điểm domain shift.
- **Quyết định (chưa triển khai, ghi lại làm chiến lược cho GĐ2)**: xếp hạng 3 phương án mô
  phỏng domain shift theo độ tin cậy — Bậc 1 (dữ liệu thật từ viện khác, dù ít, dùng làm test set
  "unseen domain" chứ không train), Bậc 2 (covariate shift thật từ `gap = study_datetime −
  treatment_datetime` hoặc phân nhóm bệnh lý — ưu tiên dùng cho cả phần FL ở GĐ1 để non-IID có
  căn cứ thật, không chỉ để dành cho DA), Bậc 3 (brightness-level nhân tạo, chỉ dùng ablation).
- **Quyết định**: chốt 4 hướng nâng rank công bố cho nhánh FL-only (khi chưa có DA) — (1) đóng
  khung theo căn cứ pháp lý thật: Luật Khám bệnh, chữa bệnh 2023 (Luật 15/2023/QH15) Điều 69 +
  Nghị định 13/2023/NĐ-CP (đã tra cứu, xem link trong `proposal.md` Phụ lục); (2) đóng góp
  benchmark/reproducibility từ chính phần Data & Training-Protocol Audit; (3) vòng validation với
  chẩn đoán bác sĩ thật — **điều kiện treo**: cần chủ động thu xếp với 108 Military Central
  Hospital, chưa có sẵn; (4) đóng khung theo hạ tầng tuyến dưới thật — **điều kiện treo**: cần
  khảo sát phần cứng thật, tạm giả định chỉ có CPU nhiều nhân, không GPU, cho đến khi khảo sát.
- **Consequences**: `proposal.md` được bổ sung mục 6 (viết lại) và mục 9 (mới); đã tạo bản Google
  Doc mới nhất tại `https://docs.google.com/document/d/10sFv-GFE6xRMFoKkpfgY8-1iLeJUJZFWgDgJvtTFhkc`
  (không ghi đè được doc gốc do giới hạn tool Google Drive — xem `record.md` không lưu; chi tiết
  giới hạn tool nằm trong lịch sử hội thoại phiên này). Còn tồn 1 bản nháp cũ
  (`1BFm5b_o7nFuksZ3yJGOFZmHvGAfTuP1feDuDQ2v-QIo`) và doc gốc người dùng gửi — cần người dùng tự
  dọn trùng lặp trên Drive.

## 3. Bài học

(agent bổ sung khi gặp lỗi thật trong lúc chạy: lỗi → cách sửa → rút ra)

## 4. Câu hỏi treo

Trích từ hướng nghiên cứu tương lai/hạn chế mà các bài báo tự nêu (chưa rõ đã được giải
quyết ở phiên bản code hiện tại hay chưa):

- (D2) Có nên huấn luyện riêng một mô hình chỉ trên tập ca khó đã cân bằng lại để cải thiện
  hiệu năng nhóm khó (ANT-HN-POST), thay vì dùng chung mô hình cho cả ca dễ/khó?
- (D4) Có nên dự đoán mức độ tăng sáng (brightness level) phù hợp theo từng ảnh thay vì cố
  định số mức cho toàn bộ dataset?
- (D4) Cần tinh chỉnh sâu hơn hyperparameter của DETR (num_queries, số layer encoder/decoder,
  class_cost) — chưa rõ đã thử nghiệm có hệ thống hay chưa.
- (D5) Cần thử nghiệm lâm sàng (clinical validation) trên tập dữ liệu lớn hơn trước khi áp
  dụng mô hình đề xuất liều xạ trị vào thực tế điều trị.

**Câu hỏi treo vận hành (chưa phải nghiên cứu, nay là T8/T9 trong plan.csv — đã gộp vào GĐ1
Federated Learning thay vì đứng riêng như T1/T2 cũ):**
- Trong `src/{detr,faster_rcnn,yolov7}/`, phần nào thực sự dùng chung (data loading, augmentation,
  training loop) đủ để đưa vào `shared/`, và phần nào đặc thù riêng từng kiến trúc nên giữ
  nguyên tại chỗ?
- "Hướng nghiên cứu gốc không dùng nữa" cụ thể là hướng nào trong 3 model (DETR/Faster-RCNN/
  YOLOv7), hay là toàn bộ cách tiếp cận object-detection nói chung (so với một hướng mới)?
- Tên thư mục đích cho code cũ: dùng chung `original paper/` (hiện đang chứa 5 PDF) hay tách
  riêng một thư mục mới để không trộn code với tài liệu tham khảo?

**Câu hỏi treo mới phát sinh từ Data & Training-Protocol Audit (D6, chặn T1–T7 trong plan.csv):**
- Patient ID dùng để GroupKFold (T1) nên lấy từ cột `name` (dễ trùng do lỗi chính tả/viết hoa
  khác nhau) hay cần một khoá định danh bệnh nhân ổn định hơn (vd DICOM PatientID nếu có trong
  metadata gốc, chưa kiểm tra)?
- Với Faster-RCNN không có cấu hình huấn luyện gốc (T5): tái tạo bằng cách suy luận ngược từ
  kiến trúc checkpoint `.ckpt` hiện có, hay chấp nhận train lại từ đầu với cấu hình mới và bỏ
  qua việc tái lập chính xác baseline cũ?
- Quy đổi "epoch tương đương" giữa centralized và FL (T6) nên tính theo tổng số ảnh đã nhìn thấy
  (gradient steps), hay theo số round × local-epoch cố định bất kể kích thước client?
