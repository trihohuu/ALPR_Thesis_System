# Vietnamese License Plate Recognition System (VLPR)

🚗 **Hệ thống nhận diện biển số xe máy/ô tô Việt Nam End-to-End với khả năng xử lý thời gian thực (Real-time).**

## Giới thiệu

Dự án này xây dựng một hệ thống giám sát và nhận diện biển số xe tự động. Hệ thống có khả năng xử lý đa nguồn vào (Hình ảnh, Video, Webcam, RTSP Stream từ mạng nội bộ) và trả về kết quả biển số dạng văn bản với độ chính xác cao nhờ sử dụng **PaddleOCR**.

Dự án được triển khai giao diện web tương tác bằng **Streamlit**, tích hợp module xử lý luồng video bất đồng bộ để đảm bảo hiệu năng trên các thiết bị cấu hình tầm trung.

## Kiến trúc Hệ thống (Pipeline)

Hệ thống hoạt động theo mô hình Multi-stage Pipeline:

1. **Input Layer**: Hỗ trợ upload file hoặc lấy luồng trực tiếp từ Camera IP thông qua giao thức RTSP (Server MediaMTX).
2. **Detection Layer**: Sử dụng model Deep Learning (YOLO) để phát hiện vị trí biển số xe trong khung hình.
3. **Preprocessing Layer**: Cắt (Crop) và xử lý biến đổi góc nhìn (Perspective Transform) để đưa biển số về dạng phẳng.
4. **Recognition Layer**: Sử dụng PaddleOCR để trích xuất ký tự từ ảnh biển số đã xử lý.
5. **Application Layer**: Giao diện Streamlit hiển thị video và kết quả thời gian thực.

## Tính năng chính

- ✅ **Đa dạng đầu vào**: Hỗ trợ Image, Video file, và Livestreaming (RTSP).
- ✅ **Real-time Processing**: Sử dụng kỹ thuật Multi-threading để đọc luồng video, giảm độ trễ (Latency).
- ✅ **Xử lý ảnh thông minh**: Tự động căn chỉnh biển số bị nghiêng trước khi đưa vào OCR.
- ✅ **Hỗ trợ tiếng Việt**: Tối ưu hóa cho format biển số xe Việt Nam (2 dòng, 1 dòng).
- ✅ **Giao diện thân thiện**: Web App trực quan, dễ sử dụng cho demo.

## Hướng phát triển (Future Work)

- [ ] Tích hợp Database (SQLite/MySQL) để lưu lịch sử ra vào.
- [ ] Thêm tính năng Tracking (DeepSort) để đếm lưu lượng xe.
- [ ] Tối ưu hóa model Detection (Quantization) để chạy trên thiết bị nhúng (Jetson Nano).

## Yêu cầu hệ thống

- Python 3.8+
- (Khuyến dùng) GPU NVIDIA + CUDA để có FPS cao nhất.
- MediaMTX (nếu muốn chạy tính năng Livestream mạng nội bộ).

## Cài đặt & Hướng dẫn sử dụng

### 1. Yêu cầu hệ thống
- Python 3.8 trở lên
- GPU NVIDIA + CUDA (khuyến khích để đạt FPS cao)

### 2. Cài đặt thư viện

```bash
git clone https://github.com/username/project-name.git
cd project-name
pip install -r requirements.txt
```

### 3. Cấu hình MediaMTX (Cho tính năng Live RTSP)

1. Tải và giải nén MediaMTX
2. Chạy file thực thi **mediamtx.exe** (Windows) hoặc **./mediamtx**
3. Đảm bảo Camera/Điện thoại và Máy tính cùng mạng LAN.
Lưu ý: Trong file mediamtx.yml thay đổi dòng protocols từ [rtsp, udp, tcp] thành [tcp] để chất lượng stream tốt hơn

### 4. Chạy ứng dụng

```bash
streamlit run app.py
```

### 5. Tác giả
- Phạm Hồ Hữu Trí - 24521841 - Khoa Học Máy Tính
- Email: edricalbert2006@gmail.com
- Phan Minh Trí - 24521843 - Khoa học Máy tính
- Email: 24521843@gm.uit.edu.vn