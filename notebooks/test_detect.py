from ultralytics import YOLO
import cv2
import os

# Load model vừa train xong
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
model = YOLO(os.path.join(CURRENT_DIR, '..', 'models', 'yolo', 'weights', 'best.pt'))

# Đường dẫn ảnh test (Bạn kiếm đại 1 ảnh xe cộ trên mạng bỏ vào)
img_path = os.path.join(CURRENT_DIR, '..', 'assets', 'test_image.jpg') 

# Dự đoán
results = model(img_path)

for result in results:
    result.show()  # Nó sẽ bật cửa sổ hiện ảnh đã vẽ khung biển số
    result.save(filename="result.jpg") # Lưu ảnh ra đĩa