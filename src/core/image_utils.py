import cv2
import numpy as np

def order_points(pts):
    """
    Sắp xếp 4 điểm theo thứ tự: 
    Top-Left, Top-Right, Bottom-Right, Bottom-Left
    """
    rect = np.zeros((4, 2), dtype="float32")
    
    # Tổng: TL có tổng nhỏ nhất, BR có tổng lớn nhất
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # Hiệu: TR có hiệu nhỏ nhất (y-x), BL có hiệu lớn nhất
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def perspective_transform(image):
    """
    Tự động tìm 4 góc của biển số và "duỗi thẳng" (Warp Perspective)
    """
    if image is None or image.size == 0:
        return image

    # 1. Chuyển Gray & Tìm cạnh
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Blur nhẹ để giảm nhiễu
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Dùng Canny để bắt cạnh hoặc Threshold
    edged = cv2.Canny(gray, 75, 200)

    # 2. Tìm Contours (đường bao)
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5] # Lấy 5 contour to nhất

    screenCnt = None
    
    # 3. Lặp qua các contour để tìm hình tứ giác (4 đỉnh)
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # Nếu xấp xỉ ra 4 điểm -> khả năng cao là biển số
        if len(approx) == 4:
            screenCnt = approx
            break

    # Nếu không tìm thấy tứ giác nào hợp lý, trả về ảnh gốc (crop padding)
    if screenCnt is None:
        return image

    # 4. Tính toán ma trận biến đổi
    pts = screenCnt.reshape(4, 2)
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Tính chiều rộng mới (Max của cạnh trên và cạnh dưới)
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # Tính chiều cao mới
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Tập hợp điểm đích (Destination points)
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # Lấy ma trận biến đổi và áp dụng
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

def preprocess_plate(plate_img):
    """
    Quy trình xử lý ảnh biển số đầy đủ:
    1. Căn chỉnh góc (Perspective Transform)
    2. Chuyển xám, khử nhiễu
    """
    if plate_img is None or plate_img.size == 0:
        return plate_img

    # BƯỚC 1: Cố gắng duỗi thẳng ảnh
    # Lưu ý: Nếu ảnh quá mờ hoặc tối, hàm này có thể trả về ảnh gốc
    aligned_img = perspective_transform(plate_img)

    # BƯỚC 2: Xử lý màu sắc để OCR tốt hơn
    if len(aligned_img.shape) == 3: # Nếu vẫn là ảnh màu
        gray = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = aligned_img
        
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Có thể thêm Threshold nếu cần, nhưng PaddleOCR chịu được Grayscale tốt
    return blur

def draw_results(image, detections):
    """
    Giữ nguyên hàm vẽ
    """
    img_copy = image.copy()
    for item in detections:
        x1, y1, x2, y2 = item['box']
        text = item.get('text', '')
        score = item.get('ocr_conf', 0.0)

        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{text} ({score:.2f})"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(img_copy, (x1, y1 - 30), (x1 + w, y1), (0, 255, 0), -1)
        cv2.putText(img_copy, label, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return img_copy