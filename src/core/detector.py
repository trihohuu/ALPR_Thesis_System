from ultralytics import YOLO
import cv2
import numpy as np

class PlateDetector:
    def __init__(self, model_path, conf_threshold=0.5):
        self.model = YOLO(model_path)
        self.conf = conf_threshold

    def detect(self, image):
        """
        Phát hiện biển số trong ảnh
        :param image: Ảnh đầu vào (numpy array từ cv2.imread)
        :return: List các dictionary, mỗi dict chứa thông tin 1 biển số
                 [{'box': [x1,y1,x2,y2], 'plate_img': numpy_img, 'conf': float}]
        """
        results = self.model(image, conf=self.conf, verbose=False)
        detections = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Lấy tọa độ
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                
                # Cắt ảnh biển số (Crop)
                # Lưu ý: cần kiểm tra biên để không cắt lố ra ngoài ảnh
                h_img, w_img = image.shape[:2]
                
                # Thêm phần padding
                padding = 10 # Mở rộng mỗi bên 10 pixel
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(w_img, x2 + padding)
                y2 = min(h_img, y2 + padding)
                plate_crop = image[y1:y2, x1:x2]
                
                detections.append({
                    'box': [x1, y1, x2, y2],
                    'plate_img': plate_crop,
                    'conf': conf
                })
        
        return detections