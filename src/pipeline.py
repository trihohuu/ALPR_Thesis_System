import cv2
import os
import time

# Import các module core
from core.detector import PlateDetector
from core.ocr_paddle import LicensePlateOCR
from core.image_utils import preprocess_plate, draw_results

class ALPRPipeline:
    def __init__(self, yolo_path, use_gpu=True):
        print("Đang khởi tạo hệ thống ALPR...")
        
        # 1. Khởi tạo Detector (YOLO)
        if not os.path.exists(yolo_path):
            raise FileNotFoundError(f"Không tìm thấy model YOLO tại: {yolo_path}")
        self.detector = PlateDetector(model_path=yolo_path)
        
        # 2. Khởi tạo OCR (Paddle)
        self.ocr = LicensePlateOCR(use_gpu=use_gpu)
        
        print("Hệ thống đã sẵn sàng!")

    def run(self, source_path, show=True, save_path=None):
        """
        Chạy quy trình trên 1 ảnh hoặc video
        :param source_path: Đường dẫn ảnh
        """
        # Đọc ảnh
        frame = cv2.imread(source_path)
        if frame is None:
            print(f" Không thể đọc ảnh: {source_path}")
            return

        start_time = time.time()

        # BƯỚC 1: Detect biển số
        detections = self.detector.detect(frame)
        print(f"🔍 Phát hiện {len(detections)} biển số.")

        # BƯỚC 2: Loop qua từng biển số để OCR
        final_results = []
        for item in detections:
            plate_img = item['plate_img']
            
            # Tiền xử lý ảnh (Phóng to, làm rõ)
            processed_plate = preprocess_plate(plate_img)
            
            # OCR
            text, conf = self.ocr.predict(processed_plate)
            
            # Lưu kết quả lại vào dict
            item['text'] = text
            item['ocr_conf'] = conf
            final_results.append(item)
            
            print(f"   -> Biển số: {text} (Conf: {conf:.2f})")

        end_time = time.time()
        fps = 1 / (end_time - start_time)
        print(f"⏱️ Thời gian xử lý: {end_time - start_time:.4f}s")

        # BƯỚC 3: Vẽ và Hiển thị
        result_image = draw_results(frame, final_results)

        if save_path:
            cv2.imwrite(save_path, result_image)
            print(f"💾 Đã lưu kết quả tại: {save_path}")

        if show:
            # Resize hiển thị nếu ảnh quá to
            h, w = result_image.shape[:2]
            if w > 1200:
                result_image = cv2.resize(result_image, (1200, int(1200*h/w)))
            
            cv2.imshow("ALPR Result", result_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

# ==========================================
# PHẦN TEST NHANH (ENTRY POINT)
# ==========================================
if __name__ == "__main__":
    # Cấu hình đường dẫn
    # Sửa lại đường dẫn model YOLO của bạn cho đúng
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(CURRENT_DIR, '..', 'models', 'yolo', 'weights', 'best.pt')
    
    # Ảnh test
    IMAGE_PATH = os.path.join(CURRENT_DIR, '..', 'assets', 'test_image.jpg')

    try:
        # Khởi tạo pipeline
        app = ALPRPipeline(yolo_path=MODEL_PATH, use_gpu=False)
        
        # Chạy
        app.run(IMAGE_PATH, save_path=os.path.join(CURRENT_DIR, '..', 'output', 'final_result.jpg'))
        
    except Exception as e:
        print(f"Có lỗi xảy ra: {e}")