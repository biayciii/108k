# Tài liệu Kỹ thuật: Tiền xử lý Dữ liệu & Huấn luyện Mô hình Phát hiện Tổn thương Tuyến giáp

Tài liệu mô tả chi tiết về tập dữ liệu ảnh xạ hình gốc, quy trình tiền xử lý ảnh y tế DICOM và cơ chế chuyển đổi định dạng nhãn cho 3 kiến trúc mô hình: **Faster R-CNN**, **DETR** và **YOLOv7**.

---

## 1. Tổng quan Dữ liệu Gốc (Dataset Overview)

* **Loại dữ liệu:** Ảnh xạ hình tuyến giáp và toàn thân (Thyroid Scintigraphy / SPECT) lưu dưới định dạng chuẩn y khoa DICOM (`.dcm`).
* **Tổng số ca chụp gốc:** Khoảng **470 – 471 ảnh `.dcm`**.
* **Kích thước ảnh:** Gồm ảnh $256 \times 256$, $512 \times 512$ (chụp tập trung vùng cổ/ngực) và $1024 \times 256$ (ảnh quét toàn thân).
* **Độ sâu màu (Bit depth):** $12\text{-bit}$ hoặc $16\text{-bit}$ thể hiện cường độ bức xạ ion hóa.
* **Các lớp mục tiêu (Classes):**
  1. `thyroid`: Vùng mô tuyến giáp bắt giữ chất phóng xạ.
  2. `shoulder`: Vùng khớp vai (làm mốc đối chứng nền bức xạ).
* **Dữ liệu lâm sàng tích hợp trong bảng nhãn:** Bổ sung các chỉ số y khoa như `residual_state` (đánh giá còn sót mô giáp), `TSH`, `TG`, và `gap` (khoảng cách thời gian từ điều trị đến chụp).

---

## 2. Phân chia Tập Dữ liệu (Dataset Split)

Dữ liệu được phân chia theo tỷ lệ chuẩn Machine Learning để đảm bảo tính khách quan và chống rò rỉ dữ liệu (Data Leakage):

* **Tập huấn luyện (Train set):** $\sim 70\%$ ($\sim 330$ ảnh) — Sử dụng để cập nhật trọng số mạng.
* **Tập kiểm định (Validation set):** $\sim 15\%$ ($\sim 70$ ảnh) — Dùng để kiểm tra Overfitting, tinh chỉnh siêu tham số và Early Stopping.
* **Tập kiểm thử (Test set):** $\sim 15\%$ ($\sim 70$ ảnh) — Đánh giá độc lập chất lượng mô hình sau cùng ($mAP@0.5$, $mAP@0.5:0.95$).

---

## 3. Quy trình Tiền xử lý Ảnh (Image Preprocessing Pipeline)

Quá trình chuyển đổi từ ma trận pixel DICOM sang ma trận số thực tương thích với mạng nơ-ron gồm 4 bước kỹ thuật chính:

### Bước 1: Cắt vùng quan tâm (Cropping)
Đối với các ảnh chụp quét toàn thân dài ($1024 \times 256$), tiến hành cắt lấy $512$ pixel nửa trên ($512 \times 256$) để tập trung vào giải phẫu vùng cổ và ngực, loại bỏ phần thân dưới không chứa đối tượng.

### Bước 2: Cân bằng dải tương phản (Intensity Windowing)
* Áp dụng Percentile Clipping để loại bỏ nhiễu nền và các điểm chói xạ giả (lấy ngưỡng $1\%$ và $99.5\%$):
  $$p_{min} = \text{Percentile}(I, 1), \quad p_{max} = \text{Percentile}(I, 99.5)$$
* Chuẩn hóa Min-Max Scaling đưa dải mức xám về thang $8\text{-bit}$ ($[0, 255]$) và ép kiểu sang `uint8`:
  $$I_{\text{scaled}} = \frac{\text{clip}(I, p_{min}, p_{max}) - p_{min}}{p_{max} - p_{min}} \times 255$$

### Bước 3: Chuẩn hóa không gian màu & Kích thước
* Nhân bản ma trận xám $1$ kênh thành ảnh $3$ kênh (RGB) để tương thích với các kiến trúc Backbone tiền huấn luyện (Pre-trained weights ImageNet).
* Đưa ảnh về kích thước chuẩn đầu vào ($512 \times 512$).

---

## 4. Định dạng Nhãn cho 3 Mô hình

Mỗi kiến trúc yêu cầu cấu trúc và quy chuẩn tọa độ Bounding Box riêng biệt từ nhãn gốc $[x_{min}, y_{min}, w, h]$:

### 1. Faster R-CNN
* **Cấu trúc lưu trữ:** Bảng `.csv` (`train.csv`, `val.csv`, `test.csv`).
* **Định dạng tọa độ:** Tọa độ pixel tuyệt đối $[x_{min}, y_{min}, x_{max}, y_{max}]$:
  $$x_{max} = x_{min} + w, \quad y_{max} = y_{min} + h$$
* **Quy ước nhãn:** Lớp `0` dành cho Background, `1: thyroid`, `2: shoulder`.

### 2. DETR (DEtection TRansformer)
* **Cấu trúc lưu trữ:** Chuẩn COCO JSON (`train.json`, `val.json`, `test.json`).
* **Định dạng tọa độ:** Tọa độ pixel tuyệt đối $[x_{min}, y_{min}, w, h]$ kèm diện tích `area = w * h`.
* **Quy ước nhãn:** `category_id` bắt đầu từ `1` (`1: thyroid`, `2: shoulder`), lớp `0` dành cho Background.

### 3. YOLOv7
* **Cấu trúc lưu trữ:** File nhãn `.txt` riêng biệt cho từng ảnh, file cấu hình `thyroid_data.yaml` và các file tiền biên dịch cache (`train.cache`, `val.cache`, `test.cache`).
* **Định dạng tọa độ:** Chuẩn hóa về khoảng $[0, 1]$ theo tọa độ tâm và kích thước:
  $$x_c = \frac{x_{min} + w/2}{W}, \quad y_c = \frac{y_{min} + h/2}{H}, \quad w_{norm} = \frac{w}{W}, \quad h_{norm} = \frac{h}{H}$$
  Định dạng dòng ghi: `<class_id> <x_c> <y_c> <w_norm> <h_norm>`.
* **Quy ước nhãn:** Đánh số bắt đầu từ `0` (`0: thyroid`, `1: shoulder`).

---

## 5. Bảng So sánh Tổng hợp giữa 3 Mô hình

| Tiêu chí | Faster R-CNN | DETR | YOLOv7 |
| :--- | :--- | :--- | :--- |
| **Kiến trúc chính** | Two-stage (RPN + RoI Head) | Transformer Encoder-Decoder | One-stage Anchor-based |
| **Định dạng nhãn** | Bảng CSV $[x_1, y_1, x_2, y_2]$ | COCO JSON $[x, y, w, h]$ | File TXT phân tán $[x_c, y_c, w, h]$ |
| **Cơ chế nạp dữ liệu** | `torch.utils.data.DataLoader` | PyTorch COCO Dataset API | YOLO Dataset Loader qua `.cache` |
| **Tăng cường dữ liệu** | Random Flip, Scaling | Multi-scale Resize ($480 \rightarrow 800$), Crop | Mosaic 4-ảnh, MixUp, HSV jitter |
| **Đặc thù đầu ra** | Box Coordinates + Softmax Class Score | 100 Object Queries + Bipartite Matching Loss | Feature Pyramids Grid Predictions |
