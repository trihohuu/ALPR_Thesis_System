# Automated License Plate Recognition (ALPR) System

Tài liệu này mô tả quy trình xây dựng, phát triển và triển khai hệ thống nhận diện biển số xe tự động từ đầu đến cuối (End-to-End AI Lifecycle).

---

## 1. Phân tích yêu cầu và xác định bài toán AI

### Mục tiêu
Xây dựng hệ thống tự động phát hiện, theo dõi và trích xuất thông tin ký tự trên biển số xe từ các nguồn dữ liệu video thời gian thực (Camera giám sát, Webcam) hoặc các tệp video lưu trữ.

### Loại bài toán
Hệ thống là sự kết hợp của ba bài toán thị giác máy tính cốt lõi:
1.  **Object Detection (Phát hiện đối tượng):** Xác định vị trí khung hình chữ nhật (bounding box) chứa biển số xe trong ảnh toàn cảnh.
2.  **Object Tracking (Theo dõi đối tượng):** Gán định danh (ID) duy nhất cho mỗi xe để tránh việc xử lý trùng lặp trên các frame liên tiếp.
3.  **Optical Character Recognition (OCR):** Nhận dạng và chuyển đổi hình ảnh ký tự trên biển số thành văn bản số hóa.

### Dữ liệu đầu vào và đầu ra
* **Input:** Luồng video (RTSP/Webcam) hoặc ảnh tĩnh/video file.
* **Output:** Chuỗi ký tự biển số xe, hình ảnh trích xuất của biển số, và độ tin cậy (confidence score) của dự đoán.

---

## 2. Thu thập và tiền xử lý dữ liệu

### Nguồn dữ liệu
* Sử dụng tập dữ liệu mở từ **Roboflow** với số lượng khoảng **10.000 ảnh** biển số xe đa dạng về điều kiện ánh sáng, góc chụp và loại xe.
* Dữ liệu bao gồm cả biển số vuông (2 dòng) và biển số dài (1 dòng).

### Tiền xử lý (Preprocessing)
Trước khi đưa vào mô hình, dữ liệu được xử lý qua các bước:
* **Gán nhãn chuẩn hóa:** Định dạng lại annotation phù hợp với kiến trúc model phát hiện vật thể.
* **Căn chỉnh hình học (Perspective Transform):** Khi cắt ảnh biển số từ xe, hệ thống tự động nắn chỉnh các biển số bị nghiêng/xéo về dạng phẳng (Frontal View) để tăng độ chính xác cho OCR.
* **Xử lý nhiễu:** Áp dụng bộ lọc Gaussian và chuyển đổi thang xám (Grayscale) để làm nổi bật đường nét ký tự.

---

## 3. Thiết kế và lựa chọn mô hình AI

### Lý do áp dụng giải pháp AI (Deep Learning)
Các phương pháp xử lý ảnh truyền thống (như phát hiện biên, contour) hoạt động kém hiệu quả trong môi trường thực tế (ánh sáng thay đổi, biển số bị mờ, che khuất). Deep Learning cho phép mô hình tự học các đặc trưng phức tạp, mang lại độ chính xác cao và khả năng tổng quát hóa tốt hơn.

### Kiến trúc mô hình (Two-Stage Pipeline)
Hệ thống sử dụng kiến trúc 2 giai đoạn (Cascade):

1.  **Giai đoạn 1: Plate Detector (Sử dụng YOLOv8)**
    * **Ưu điểm:** Tốc độ suy luận cực nhanh (Real-time), cân bằng tốt giữa tốc độ và độ chính xác (mAP).
    * **Nhược điểm:** Có thể gặp khó khăn với các đối tượng quá nhỏ nếu không được tinh chỉnh tham số anchor phù hợp.

2.  **Giai đoạn 2: Text Recognizer (Sử dụng PaddleOCR)**
    * **Ưu điểm:** Hỗ trợ nhận diện đa ngôn ngữ, hoạt động tốt trên các văn bản nghiêng hoặc có phông chữ phức tạp. Nhẹ hơn các mô hình Transformer lớn nhưng chính xác hơn Tesseract truyền thống.
    * **Cơ chế sửa lỗi (Post-processing):** Tích hợp các luật Heuristic để sửa các lỗi nhầm lẫn phổ biến (ví dụ: nhầm số `0` thành chữ `O`, số `8` thành chữ `B`) dựa trên quy chuẩn định dạng biển số.

---

## 4. Cài đặt, huấn luyện và triển khai hệ thống AI

### Huấn luyện (Training)
* Mô hình phát hiện vật thể được Fine-tune trên GPU với kỹ thuật **Transfer Learning** từ trọng số tiền huấn luyện (Pre-trained weights), giúp mô hình hội tụ nhanh hơn và đạt độ chính xác cao dù dữ liệu huấn luyện có hạn.

### Tích hợp ứng dụng (Application Integration)
Hệ thống được đóng gói theo kiến trúc Microservices:
* **Backend:** Xây dựng API Service để chịu tải việc xử lý AI nặng nề. API nhận ảnh đầu vào và trả về kết quả JSON.
* **Frontend:** Giao diện Dashboard tương tác cho phép người dùng xem luồng video trực tiếp, quản lý danh sách biển số đã nhận diện và xem lại lịch sử.
* **Tracking Module:** Tích hợp thuật toán IoU (Intersection over Union) để theo dõi hành trình của biển số, chọn ra khung hình rõ nét nhất để thực hiện OCR, giúp tối ưu tài nguyên hệ thống (không cần OCR trên mọi frame).

---

## 5. Đánh giá và giám sát

### Đánh giá hiệu năng
* Hệ thống liên tục theo dõi độ tin cậy (Confidence Score) của cả bước phát hiện và bước nhận diện ký tự.
* Các kết quả có độ tin cậy thấp (dưới ngưỡng threshold) sẽ tự động bị lọc bỏ để tránh dương tính giả (False Positives).

### Giám sát vận hành
* Hệ thống hiển thị trực quan các Bounding Box và nhãn dự đoán ngay trên video thời gian thực để người vận hành dễ dàng kiểm tra.
* Cơ chế log lỗi và cảnh báo được tích hợp để phát hiện các sự cố như mất kết nối camera hoặc lỗi tải model.

---

## 6. Hướng dẫn chạy (How to run)

### Bước 1: Cài đặt thư viện: 
```bash
git clone https://github.com/trihohuu/ALPR_Thesis_System.git
cd ALPR_Thesis_System
pip install -r requirements.txt
```

### Bước 2: Khởi động BackEnd
Mở terminal tại thư mục gốc của dự án và chạy lệnh:

```bash
python api/main.py
```

### Bước 3: Khởi động Streamlit
Mở thêm 1 terminal khác tại thư mục gốc của dự án và chạy lệnh:

```bash
streamlit run web_app/app.py
```

## 7. Credit
- Phạm Hồ Hữu Trí - 24521841 (24521841@gm.uit.edu.vn)
- Phan Minh Trí - 24521843 (24521843@gm.uit.edu.vn)
