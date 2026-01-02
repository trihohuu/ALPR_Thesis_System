from ultralytics import YOLO
import cv2
import numpy as np

class PlateDetector:
    def __init__(self, model_path, conf_threshold=0.32):
        self.model = YOLO(model_path)
        self.conf = conf_threshold

    def detect(self, image):
        results = self.model(image, conf=self.conf, verbose=False)
        detections = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())

                h_img, w_img = image.shape[:2]
                
                # Add paddings 
                padding = 10 
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(w_img, x2 + padding)
                y2 = min(h_img, y2 + padding)
                plate_crop = image[y1:y2, x1:x2]

                h_crop, w_crop = plate_crop.shape[:2]
                if h_crop < 64: 
                    scale = 64 / h_crop
                    plate_crop = cv2.resize(plate_crop, (int(w_crop * scale), 64), interpolation=cv2.INTER_CUBIC)
                
                detections.append({
                    'box': [x1, y1, x2, y2],
                    'plate_img': plate_crop,
                    'conf': conf
                })
        
        return detections