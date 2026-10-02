# record.md — 108 Thyroid

File duy nhất agent chủ động cập nhật, đọc đầu tiên mỗi phiên. Không xóa entry cũ — đổi ý
về một quyết định thì thêm entry mới, ghi rõ "thay thế entry #N".

## 1. Bản đồ

| File/thư mục | Vai trò | Ai cập nhật |
|---|---|---|
| `README.md` (root) | Tài liệu kỹ thuật: tiền xử lý ảnh SPECT/DICOM + định dạng nhãn cho DETR/Faster-RCNN/YOLOv7 | Người dùng |
| `original paper/` | 5 bài báo gốc (PDF) của nhóm nghiên cứu — nguồn quyết định/bài học ở mục 2 | Người dùng |
| `Docs/Ref.xlsx` | Bảng theo dõi tài liệu tham khảo (lấy từ Google Sheet `108 K/Docs/Ref.xlsx` trên Drive) — tóm tắt + note cho 5 paper gốc và ~19 paper/git-repo khác đang được rà soát cho hướng nghiên cứu tiếp theo (privacy-preserving FL, MAE/representation learning, synthetic data, domain adaptation cho ảnh tuyến giáp) | Người dùng |
| `uet-thyroid-detection-main/` | Code gốc: `app.py` (demo Gradio 3 model), `src/{detr,faster_rcnn,yolov7}/`, `data/`, `assets/`, `draft.ipynb`, `logistic_model.pkl`, `yolov7.pt` — **legacy, không sửa/di chuyển** (xem D7) | Người dùng (chỉ đọc/import từ đây, không sửa) |
| `Train/` | Cấu trúc code huấn luyện mới, tối ưu hơn — thiết kế + scaffold thật trong `Train/init.md` (D7). `configs/`, `src/common/{data,metrics,utils}`, `src/engine/trainer.py`, `scripts/train.py` đã có code chạy được (đã smoke-test); `src/models/*` còn chặn bởi xung đột tên package (mục 4 `Train/init.md`) | Agent (plan.csv T8, chưa Done — chờ model wiring) |
| `Data/` | Dữ liệu đã convert sẵn theo 3 định dạng: `detr_data/` (COCO JSON), `faster-rcnn-data/` (CSV), `yolov7-data/` (YOLO txt + hyp/cfg) | Người dùng |
| `Experiments/Re-create/Result/` | Kết quả train/eval đã commit — dịch chuyển từ `Result/` cũ ngày 2026-10-02 (xem D8): 3 baseline runs (DETR/Faster-RCNN/YOLOv7) + `test-hcl-svm` (tái hiện D4, `exact_paper_reproduction: false`). `Train/runs/` vẫn là nơi output mới (gitignored) | Agent |
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

### D7 — `uet-thyroid-detection-main/` là legacy, không đụng đến; cấu trúc train mới nằm ở `Train/` (root repo)
- **Context**: Mục 4 (Câu hỏi treo vận hành) từng để ngỏ ranh giới `shared/` vs code đặc thù và
  tên thư mục đích cho code hướng gốc không dùng nữa. Trong phiên dựng `Train/init.md`
  (2026-09-28), người dùng chốt quyết định trực tiếp thay vì tiếp tục để ngỏ.
- **Decision**: `uet-thyroid-detection-main/` giữ nguyên toàn bộ, coi là **legacy/tham chiếu** —
  không sửa, không di chuyển bất kỳ file nào ra khỏi đó. Toàn bộ code huấn luyện mới (data
  loading đã chuẩn hoá theo audit D6, config, sau này là FL simulation) được xây trong `Train/`
  ở root repo — xem cây thư mục + nguyên tắc thiết kế trong `Train/init.md`. Định nghĩa model
  (DETR/Faster-RCNN/YOLOv7) được **import lại** từ `uet-thyroid-detection-main/src`, không
  copy-paste, không sửa tại chỗ.
- **Rejected alternatives**: (a) Tạo `shared/` lồng bên trong `uet-thyroid-detection-main` (kế
  hoạch T8 cũ) — bị loại vì muốn giữ thư mục này hoàn toàn nguyên vẹn làm mốc tái lập D1–D5. (b)
  Di chuyển code hướng gốc không dùng nữa sang `original paper/` hoặc một thư mục tách riêng (kế
  hoạch T9 cũ) — bị loại: không cần di chuyển gì cả, sự tách biệt đạt được tự nhiên bằng cách
  không đụng vào `uet-thyroid-detection-main` và chỉ thêm code mới ở `Train/`.
- **Consequences**: `plan.csv` T8/T9 được viết lại cho khớp quyết định này (T8 → dựng scaffold
  `Train/` theo `Train/init.md`; T9 → thu hẹp thành xác nhận `Train/` không copy/sửa ngược vào
  `uet-thyroid-detection-main`, không còn ý nghĩa "di chuyển code" ban đầu). Câu hỏi treo vận
  hành cũ ở mục 4 coi như đã được trả lời bởi entry này.

### D8 — Chuyển kết quả train/eval sang `Experiments/Re-create/Result/` (2026-10-02)
- **Context**: Pull ngày 30/09–01/10 thêm 2 commit kết quả (`faab0d3` DETR/FRCNN/YOLOv7, `384ceed` test-hcl-svm) nhưng không có code train mới — code train mới vẫn ở `Train/` (commit `1a61fc3`). Các thư mục kết quả nằm lẫn với code/docs ở root, khó tách biệt khi đọc repo.
- **Decision**: Tạo `Experiments/` làm thư mục cha cho các nhánh thực nghiệm; di chuyển nguyên `Result/` → `Experiments/Re-create/Result/` bằng `git mv` (giữ lịch sử, LFS pointer `*.pth`/`*.pt` không đổi). `Train/` giữ nguyên ở root. Cập nhật tất cả tham chiếu đường dẫn trong `AGENTS.md` root. Tên "Re-create" phản ánh bộ kết quả tái lập baseline các paper D1–D5.
- **Rejected alternatives**: Tạo `Experiments/` mà xóa `Result/` cũ thay vì `git mv` — bị loại vì mất lịch sử blame/diff. Chuyển cả `Train/` vào `Experiments/` — bị loại vì `Train/` là scaffold đang phát triển, không phải kết quả thực nghiệm hoàn tất.
- **Consequences**: `AGENTS.md` và `record.md` dùng đường dẫn `Experiments/Re-create/Result/`; `git log --follow` vẫn truy vết được file cũ trong `Result/`; mọi script/path tuyệt đối trong `config.json` của các run cũ (`/mnt/nvme2/...`) vốn đã không trỏ local nên không bị ảnh hưởng.

## 3. Bài học

- **Lỗi**: `yaml.safe_dump()` (PyYAML `SafeDumper`) không tự nhận diện `Config` (dict
  subclass tự viết trong `Train/src/common/utils/config.py`) là `dict` — representer của
  PyYAML tra theo `type(data)` chính xác, không theo `isinstance`, nên dump thất bại với
  `RepresenterError: cannot represent an object` ngay khi config có dict con lồng nhau (mọi
  config thật đều có, vd `training:`, `data:`).
  **Cách sửa**: thêm hàm `_to_plain()` chuyển đệ quy `Config`/dict lồng nhau về `dict`/`list`
  thuần trước khi `yaml.safe_dump`.
  **Rút ra**: bất kỳ subclass nào của `dict`/`list` (kể cả tiện lợi khi viết code) đều cần tự
  convert về type gốc trước khi đưa qua thư viện serialize ngoài (yaml/json) — không giả định
  thư viện tự xử lý qua `isinstance`. Phát hiện được nhờ chủ động smoke-test
  `save_resolved_config()` với config lồng nhau thật, không chỉ test với dict phẳng.

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

**Câu hỏi treo vận hành cũ — ĐÃ TRẢ LỜI bởi D7 (2026-09-28), giữ lại để truy vết:**
- ~~Trong `src/{detr,faster_rcnn,yolov7}/`, phần nào thực sự dùng chung... đủ để đưa vào
  `shared/`?~~ → Không đưa vào `shared/` lồng trong uet nữa; toàn bộ code chung mới nằm ở
  `Train/src/common/`, xem `Train/init.md`.
- ~~"Hướng nghiên cứu gốc không dùng nữa" cụ thể là hướng nào?~~ → Không có hướng nào bị "loại
  bỏ"/di chuyển; `uet-thyroid-detection-main` giữ nguyên 100% làm legacy, chỉ đơn giản là không
  dùng để phát triển tiếp.
- ~~Tên thư mục đích cho code cũ?~~ → Không cần thư mục đích — không di chuyển gì cả (D7).

**Câu hỏi treo mới phát sinh khi scaffold `Train/` (D7, chặn phần còn lại của T8):**
- `uet-thyroid-detection-main/src/` và `Train/src/` đều là package tên `src` — import kiểu
  `from src.detr...` trong `Train/src/models/*/__init__.py` có nguy cơ resolve nhầm vào
  `Train/src` rỗng thay vì package thật bên `uet-thyroid-detection-main`. Đổi tên package
  `Train/src` (vd `train_lib`), hay dùng `importlib.util.spec_from_file_location` nạp thẳng
  theo đường dẫn file? Xem `Train/init.md` mục 4 để biết chi tiết + trade-off 2 phương án.

**Câu hỏi treo mới phát sinh từ Data & Training-Protocol Audit (D6, chặn T1–T7 trong plan.csv):**
- Patient ID dùng để GroupKFold (T1) nên lấy từ cột `name` (dễ trùng do lỗi chính tả/viết hoa
  khác nhau) hay cần một khoá định danh bệnh nhân ổn định hơn (vd DICOM PatientID nếu có trong
  metadata gốc, chưa kiểm tra)?
- Với Faster-RCNN không có cấu hình huấn luyện gốc (T5): tái tạo bằng cách suy luận ngược từ
  kiến trúc checkpoint `.ckpt` hiện có, hay chấp nhận train lại từ đầu với cấu hình mới và bỏ
  qua việc tái lập chính xác baseline cũ?
- Quy đổi "epoch tương đương" giữa centralized và FL (T6) nên tính theo tổng số ảnh đã nhìn thấy
  (gradient steps), hay theo số round × local-epoch cố định bất kể kích thước client?
