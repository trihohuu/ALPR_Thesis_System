import cv2
import os
import time
from core.detector import PlateDetector
from core.ocr_paddle import LicensePlateOCR
# from core.ocr_easy import LicensePlateEasyOCR as LicensePlateOCR
from core.tracker import Tracker
from core.image_utils import preprocess_plate, draw_results

class ALPRPipeline:
    def __init__(self, yolo_path, use_gpu=True):
        if not os.path.exists(yolo_path):
            print(f"DEBUG: Looking for model at: {os.path.abspath(yolo_path)}")
            raise FileNotFoundError(f"YOLO weights not found at: {yolo_path}")
        
        self.detector = PlateDetector(model_path=yolo_path)
        self.ocr = LicensePlateOCR(use_gpu=use_gpu)
        self.tracker = Tracker(iou_threshold=0.5)
        
        current_file_dir = os.path.dirname(os.path.abspath(__file__))

        self.crop_dir = os.path.join(current_file_dir, '..', 'output', 'plates')
        os.makedirs(self.crop_dir, exist_ok=True)
        self.save_ids = set()

    def _process_frame(self, frame, run_ocr=True):
        detections = self.detector.detect(frame)
        if run_ocr:
            for item in detections:
                plate_img = item['plate_img']
                processed_plate = preprocess_plate(plate_img)
                text, conf = self.ocr.predict(processed_plate)
                
                item['text'] = text
                item['ocr_conf'] = conf
                if conf < 0.5: # Ngưỡng lọc text rác
                    item['text'] = "" 
        else:
            for item in detections:
                item['text'] = ""
                item['ocr_conf'] = 0.0

        final_results = self.tracker.update(detections)
        result_frame = draw_results(frame, final_results)
        return result_frame

    def process_single_frame(self, frame, frame_idx):
        run_ocr = (frame_idx % 5 == 0)
        detections = self.detector.detect(frame)
        
        if run_ocr:
            for item in detections:
                plate_img = item['plate_img']
                processed_plate = preprocess_plate(plate_img)
                text, conf = self.ocr.predict(processed_plate)
                
                item['text'] = text
                item['ocr_conf'] = conf
                if conf < 0.5: 
                    item['text'] = "" 
        else:
            for item in detections:
                item['text'] = ""
                item['ocr_conf'] = 0.0

        final_results = self.tracker.update(detections)
        result_frame = draw_results(frame, final_results)
        
        return result_frame, final_results

    def save_final_results(self):
        count = 0
        all_vehicles = list(self.tracker.all_tracks.values())
        
        for track in all_vehicles:
            plate_txt = track.get('best_text', '')
            plate_img = track.get('best_img', None)
            if plate_txt and plate_img is not None and len(plate_txt) >= 6:
                safe_name = "".join([c for c in plate_txt if c.isalnum()])
                filename = f"{safe_name}.jpg"
                save_path = os.path.join(self.crop_dir, filename)
                
                try:
                    cv2.imwrite(save_path, plate_img)
                    count += 1
                except Exception as e:
                    print(f"Lỗi lưu file {filename}: {e}")
                    
        print(f"Tổng cộng đã lưu: {count} biển số.")
    
    def run(self, source_path, show=True, save_path=None):
        if not os.path.exists(source_path):
            print(f"File not founded")
            return

        ext = os.path.splitext(source_path)[1].lower()
        video_exts = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
        if ext in video_exts:
            self._process_video(source_path, show, save_path)
        else:
            self._process_image(source_path, show, save_path)

        print("Done.")

    def _process_image(self, img_path, save_path):
        frame = cv2.imread(img_path)
        if frame is None:
            return

        processed_frame = self._process_frame(frame, run_ocr=True)
        if save_path:
            cv2.imwrite(save_path, processed_frame)

    def _process_video(self, video_path, show, save_path):
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        writer = None
        if save_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

        frame_idx = 0
        OCR_INTERVAL = 5 
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break
            
            # Run OCR every (OCR_INTERVAL) frames
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

if __name__ == "__main__":
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(CURRENT_DIR, '..', 'models', 'yolo', 'weights', 'best.pt')
    INPUT_FILE = os.path.join(CURRENT_DIR, '..', 'assets', 'test_video.mp4') 

    filename = os.path.basename(INPUT_FILE)
    OUTPUT_FILE = os.path.join(CURRENT_DIR, '..', 'output', 'result_' + filename)
    try:
        app = ALPRPipeline(yolo_path=MODEL_PATH, use_gpu=True) 
        app.run(INPUT_FILE, save_path=OUTPUT_FILE, show=False)
    except Exception as e:
        print(f"Error: {e}")