import cv2
import numpy as np

def preprocess_plate(plate_img):
    if plate_img is None or plate_img.size == 0:
        return plate_img

    # 2. Phóng to ảnh (Upscale)
    h, w = plate_img.shape[:2]
    scale_factor = 1  # Giảm xuống 2 là đủ, 3 có thể làm vỡ hạt quá mức
    resized = cv2.resize(plate_img, (w * scale_factor, h * scale_factor), interpolation=cv2.INTER_CUBIC)

    # 3. Chuyển xám và khử nhiễu
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 4. Nhị phân hóa (Tuỳ chọn: Giúp chữ đen/nền trắng tách biệt hẳn)
    # _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return blur

def draw_results(image, detections):
    """
    Hàm vẽ kết quả lên ảnh gốc
    """
    img_copy = image.copy()
    for item in detections:
        x1, y1, x2, y2 = item['box']
        text = item.get('text', '')
        score = item.get('ocr_conf', 0.0)

        # Vẽ khung
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Vẽ nền chữ cho dễ đọc
        label = f"{text} ({score:.2f})"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(img_copy, (x1, y1 - 30), (x1 + w, y1), (0, 255, 0), -1)
        
        # Viết chữ
        cv2.putText(img_copy, label, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    return img_copy