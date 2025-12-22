import cv2
import os
import time
from core.detector import PlateDetector
from core.ocr_paddle import LicensePlateOCR
from core.tracker import Tracker
from core.image_utils import preprocess_plate, draw_results

class ALPRPipeline:
    def __init__(self, yolo_path, use_gpu=True):
        print("Đang khởi tạo hệ thống ALPR...")

        if not os.path.exists(yolo_path):
            raise FileNotFoundError(f"Không tìm thấy model YOLO!!")
        self.detector = PlateDetector(model_path=yolo_path)
        self.ocr = LicensePlateOCR(use_gpu=use_gpu)
        self.tracker = Tracker(iou_threshold=0.5)
        self.plates = []
        self.crop_dir = os.path.join(CURRENT_DIR, '..', 'output', 'plates')
        os.makedirs(self.crop_dir, exist_ok=True)

        self.save_ids = set()
        
        print("Hệ thống đã sẵn sàng!")

    def _process_frame(self, frame, run_ocr=True):
        # B1: Detect
        detections = self.detector.detect(frame)
        
        # B2: OCR (Logic cũ)
        if run_ocr:
            for item in detections:
                plate_img = item['plate_img']
                processed_plate = preprocess_plate(plate_img)
                text, conf = self.ocr.predict(processed_plate)
                
                item['text'] = text
                item['ocr_conf'] = conf
                
                # Filter rác: Nếu conf thấp quá thì coi như chưa đọc được
                if conf < 0.5:
                    item['text'] = "" 
        else:
            # Nếu không chạy OCR, text mặc định là rỗng
            for item in detections:
                item['text'] = ""
                item['ocr_conf'] = 0.0

        # B3: TRACKING (Logic Mới)
        # Tracker sẽ tự động điền text từ quá khứ vào nếu hiện tại text bị rỗng
        final_results = self.tracker.update(detections)

        # B4: Vẽ kết quả
        result_frame = draw_results(frame, final_results)
        return result_frame

    def save_final_results(self):
        """
        Hàm này gọi khi kết thúc video để lưu các biển số tốt nhất
        """
        print(f"\n--- Đang lưu các biển số đã nhận diện vào {self.crop_dir} ---")
        count = 0
        
        # [SỬA Ở ĐÂY] Lặp qua all_tracks thay vì tracks
        # self.tracker.all_tracks là dict {id: track_data}, nên cần .values()
        all_vehicles = list(self.tracker.all_tracks.values())
        
        for track in all_vehicles:
            # Lấy thông tin tốt nhất
            plate_txt = track.get('best_text', '')
            plate_img = track.get('best_img', None)
            
            # Chỉ lưu nếu có text và có ảnh
            if plate_txt and plate_img is not None and len(plate_txt) > 3:
                safe_name = "".join([c for c in plate_txt if c.isalnum()])
                filename = f"{safe_name}.jpg"
                save_path = os.path.join(self.crop_dir, filename)
                
                try:
                    cv2.imwrite(save_path, plate_img)
                    # print(f"Đã lưu: {filename}") # Comment bớt cho đỡ spam
                    count += 1
                except Exception as e:
                    print(f"Lỗi lưu file {filename}: {e}")
                    
        print(f"Tổng cộng đã lưu: {count} biển số.")
    
    def run(self, source_path, show=True, save_path=None):
        if not os.path.exists(source_path):
            print(f"Không tìm thấy file")
            return

        ext = os.path.splitext(source_path)[1].lower()
        video_exts = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']

        if ext in video_exts:
            self._process_video(source_path, show, save_path)
        else:
            self._process_image(source_path, show, save_path)

    def _process_image(self, img_path, save_path):
        # Ảnh tĩnh thì luôn luôn chạy OCR
        print(f"\nĐang xử lý ảnh: {img_path}")
        frame = cv2.imread(img_path)
        if frame is None:
            return

        start = time.time()
        processed_frame = self._process_frame(frame, run_ocr=True)
        print(f"Thời gian: {time.time() - start:.4f}s")

        if save_path:
            cv2.imwrite(save_path, processed_frame)
            print(f"Đã lưu tại: {save_path}")

    def _process_video(self, video_path, show, save_path):
        print(f"\nĐang xử lý video: {video_path}")
        cap = cv2.VideoCapture(video_path)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        ROTATE_CODE = None # cv2.ROTATE_180 nếu video bị ngược

        writer = None
        if save_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

        frame_idx = 0
        
        # [CẤU HÌNH] Chạy OCR mỗi bao nhiêu frame?
        # Ví dụ: 5 nghĩa là Frame 0, 5, 10... mới đọc chữ. Frame 1,2,3,4 chỉ vẽ box.
        OCR_INTERVAL = 5 

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if ROTATE_CODE is not None:
                frame = cv2.rotate(frame, ROTATE_CODE)

            # Logic nhảy frame:
            is_run_ocr = (frame_idx % OCR_INTERVAL == 0)

            frame_idx += 1
            print(f"Frame {frame_idx}/{total_frames} (OCR: {is_run_ocr})...", end='\r')
            
            processed_frame = self._process_frame(frame, run_ocr=is_run_ocr)
            
            if writer:
                writer.write(processed_frame)

        cap.release()
        if writer:
            writer.release()
        self.save_final_results()
        print(f"\n Hoàn tất! Video đã lưu tại: {save_path}")

if __name__ == "__main__":
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(CURRENT_DIR, '..', 'models', 'yolo', 'weights', 'best.pt')
    INPUT_FILE = os.path.join(CURRENT_DIR, '..', 'assets', 'test_video.mp4') 

    filename = os.path.basename(INPUT_FILE)
    OUTPUT_FILE = os.path.join(CURRENT_DIR, '..', 'output', 'result_' + filename)

    try:
        # Nhớ chỉnh use_gpu=True nếu máy có card rời
        app = ALPRPipeline(yolo_path=MODEL_PATH, use_gpu=True) 
        app.run(INPUT_FILE, save_path=OUTPUT_FILE, show=False)
        
    except Exception as e:
        print(f"Lỗi: {e}")