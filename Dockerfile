# 1. Chọn môi trường Python 3.9
FROM python:3.9-slim

# 2. Thiết lập thư mục làm việc
WORKDIR /app

# 3. Cài đặt các thư viện hệ thống
# ĐÃ SỬA: Thay libgl1-mesa-glx bằng libgl1
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy file requirements và cài đặt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy toàn bộ mã nguồn
COPY . .

# 6. Mở port 8501
EXPOSE 8501

# 7. Lệnh chạy ứng dụng
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]