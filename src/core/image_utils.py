import cv2
import numpy as np

def preprocess_plate(plate_img):
    """
    Xử lý ảnh biển số trước khi đưa vào OCR
    """
    if plate_img is None or plate_img.size == 0:
        return plate_img

    # 1. Phóng to ảnh (Upscale)
    # OCR hoạt động tốt hơn với ảnh to, đặc biệt nếu biển số ở xa
    h, w = plate_img.shape[:2]
    scale_factor = 2 # Phóng to gấp đôi
    resized = cv2.resize(plate_img, (w * scale_factor, h * scale_factor), interpolation=cv2.INTER_CUBIC)

    # 2. Chuyển sang ảnh xám (Grayscale)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # 3. Khử nhiễu nhẹ (Gaussian Blur)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 4. Tăng tương phản/Cân bằng sáng (Histogram Equalization)
    # Giúp biển số rõ hơn trong điều kiện chói hoặc tối
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    # enhanced = clahe.apply(blur)
    
    # Hoặc dùng Threshold (nhị phân hóa) nếu muốn chữ đen nền trắng tuyệt đối
    # _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Tạm thời trả về ảnh xám đã khử nhiễu là đủ tốt cho PaddleOCR
    # Nếu muốn dùng ảnh màu thì trả về 'resized'
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