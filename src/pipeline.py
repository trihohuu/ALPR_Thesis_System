import cv2
import os
import time
from core.detector import PlateDetector
from core.ocr_paddle import LicensePlateOCR
from core.image_utils import preprocess_plate, draw_results

class ALPRPipeline:
    def __init__(self, yolo_path, use_gpu=True):
        print("🚀 Đang khởi tạo hệ thống ALPR...")
        
        # 1. Khởi tạo Detector (YOLO)
        if not os.path.exists(yolo_path):
            raise FileNotFoundError(f"Không tìm thấy model YOLO tại: {yolo_path}")
        self.detector = PlateDetector(model_path=yolo_path)
        
        # 2. Khởi tạo OCR (Paddle)
        self.ocr = LicensePlateOCR(use_gpu=use_gpu)
        
        print("✅ Hệ thống đã sẵn sàng!")

    def _process_frame(self, frame):
        """
        Hàm xử lý nội bộ cho 1 khung hình (Dùng chung cho cả Ảnh và Video)
        Input: Ảnh gốc
        Output: Ảnh đã vẽ box + thông tin biển số
        """
        # B1: Detect
        detections = self.detector.detect(frame)
        
        # B2: OCR từng biển số
        final_results = []
        for item in detections:
            plate_img = item['plate_img']
            
            # Tiền xử lý (Padding, Resize,...) - Như đã fix ở bước trước
            processed_plate = preprocess_plate(plate_img)
            
            # OCR
            text, conf = self.ocr.predict(processed_plate)
            
            item['text'] = text
            item['ocr_conf'] = conf
            final_results.append(item)
            
            # Chỉ in log nếu độ tin cậy cao để đỡ spam terminal khi chạy video
            if conf > 0.5:
                print(f"   -> Biển số: {text} (Conf: {conf:.2f})")

        # B3: Vẽ kết quả
        result_frame = draw_results(frame, final_results)
        return result_frame

    def run(self, source_path, show=True, save_path=None):
        """
        Tự động nhận diện Ảnh hoặc Video để xử lý
        """
        if not os.path.exists(source_path):
            print(f"❌ Không tìm thấy file: {source_path}")
            return

        # Kiểm tra đuôi file để biết là ảnh hay video
        ext = os.path.splitext(source_path)[1].lower()
        video_exts = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']

        if ext in video_exts:
            self._process_video(source_path, show, save_path)
        else:
            self._process_image(source_path, show, save_path)

    def _process_image(self, img_path, show, save_path):
        print(f"\n🖼️ Đang xử lý ảnh: {img_path}")
        frame = cv2.imread(img_path)
        if frame is None:
            print("Lỗi đọc ảnh!")
            return

        start = time.time()
        processed_frame = self._process_frame(frame)
        print(f"⏱️ Thời gian: {time.time() - start:.4f}s")

        if save_path:
            cv2.imwrite(save_path, processed_frame)
            print(f"💾 Đã lưu tại: {save_path}")

        if show:
            self._show_result(processed_frame, wait_duration=0) # 0 = Đợi bấm phím

    def _process_video(self, video_path, show, save_path):
        print(f"\n🎥 Đang xử lý video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        
        # Lấy thông số video
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        writer = None
        if save_path:
            # Tạo video writer (MP4)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            print(f"Frame {frame_idx}/{total_frames}...", end='\r') # In đè dòng để đỡ spam
            
            # Xử lý frame
            processed_frame = self._process_frame(frame)
            
            # Lưu video
            if writer:
                writer.write(processed_frame)
            
            # Hiển thị
            if show:
                # waitKey(1) để video tự chạy, nhấn 'q' để thoát
                if self._show_result(processed_frame, wait_duration=1) == ord('q'):
                    print("\nĐã dừng bởi người dùng.")
                    break

        cap.release()
        if writer:
            writer.release()
        print(f"\n✅ Hoàn tất! Video đã lưu tại: {save_path}")

    def _show_result(self, image, wait_duration=0):
        # Resize nếu ảnh quá to để hiển thị vừa màn hình
        h, w = image.shape[:2]
        if w > 1200:
            scale = 1200 / w
            image = cv2.resize(image, (1200, int(h * scale)))
        
        cv2.imshow("ALPR System (Teddy)", image)
        return cv2.waitKey(wait_duration)

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(CURRENT_DIR, '..', 'models', 'yolo', 'weights', 'best.pt')
    
    # --- TEST VIDEO HOẶC ẢNH TẠI ĐÂY ---
    # Bạn thay đổi tên file ở đây là được
    # INPUT_FILE = os.path.join(CURRENT_DIR, '..', 'assets', 'test_image5.jpg') 
    INPUT_FILE = os.path.join(CURRENT_DIR, '..', 'assets', 'ducky.mp4') # Ví dụ test video

    # Tên file đầu ra tự động
    filename = os.path.basename(INPUT_FILE)
    OUTPUT_FILE = os.path.join(CURRENT_DIR, '..', 'output', 'result_' + filename)

    try:
        # Nhớ set use_gpu=False nếu đang chạy CPU để tránh lỗi Segfault
        app = ALPRPipeline(yolo_path=MODEL_PATH, use_gpu=False)
        
        # Chạy pipeline
        app.run(INPUT_FILE, save_path=OUTPUT_FILE, show=False)
        
    except Exception as e:
        print(f"❌ Lỗi Fatal: {e}")