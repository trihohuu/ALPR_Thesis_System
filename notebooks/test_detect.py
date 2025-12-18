from ultralytics import YOLO
import cv2
import os

# Load model vừa train xong
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
model = YOLO(os.path.join(CURRENT_DIR, '..', 'models', 'yolo', 'weights', 'best.pt'))

# Đường dẫn ảnh test (Bạn kiếm đại 1 ảnh xe cộ trên mạng bỏ vào)
#img_path = os.path.join(CURRENT_DIR, '..', 'assets', 'test_image.jpg') 
video_path = os.path.join(CURRENT_DIR, '..', 'assets', 'test_video.mp4')
custom_output_dir = '/mnt/d/github/alpr_thesis_system/output'
results = model.predict(source=video_path, stream = True, save=True, conf=0.5, project = custom_output_dir, name = 'result_video', exist_ok = True)

for r in results:
    pass  # Không cần làm gì trong này cả, chỉ cần lặp để code chạy

print("Đã xong! RAM an toàn.")