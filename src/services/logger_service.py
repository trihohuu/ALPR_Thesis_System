# src/services/logger_service.py
import os
import cv2
import csv
import uuid
import threading
import queue
import time
from datetime import datetime

class LoggerService:
    def __init__(self, project_root):
        self.data_dir = os.path.join(project_root, 'data')
        self.raw_dir = os.path.join(self.data_dir, 'raw')
        self.hard_dir = os.path.join(self.data_dir, 'hard_samples')
        self.anno_dir = os.path.join(self.data_dir, 'annotations')
        self.log_file = os.path.join(self.data_dir, 'logs.csv')

        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.hard_dir, exist_ok=True)
        os.makedirs(self.anno_dir, exist_ok=True)

        if not os.path.exists(self.log_file):
            with open(self.log_file, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'camera_id', 'track_id', 'text', 'confidence', 'image_path', 'type'])

        self.log_queue = queue.Queue()
        self.stop_event = threading.Event()

        self.worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.worker_thread.start()

    def log_detection(self, track_data, camera_id="CAM_01"):
        plate_img = track_data.get('plate_img')
        if plate_img is None:
            return

        img_copy = plate_img.copy()

        log_packet = {
            "plate_img": img_copy,
            "text": track_data.get('text', ''),
            "ocr_conf": track_data.get('ocr_conf', 0.0),
            "track_id": track_data.get('id', 'unknown'),
            "camera_id": camera_id,
            "timestamp_obj": datetime.now()
        }

        self.log_queue.put(log_packet)

    def _process_queue(self):
        while not self.stop_event.is_set():
            try:
                packet = self.log_queue.get(timeout=1)
            except queue.Empty:
                continue

            try:
                self._save_to_disk(packet)
            except Exception as e:
                print(f"Error saving: {e}")
            finally:
                self.log_queue.task_done()

    def _save_to_disk(self, packet):
        text = packet['text']
        conf = packet['ocr_conf']
        track_id = packet['track_id']
        camera_id = packet['camera_id']
        plate_img = packet['plate_img']
        timestamp_obj = packet['timestamp_obj']

        is_hard_sample = (conf < 0.8) or (len(text) == 0)
        save_folder = self.hard_dir if is_hard_sample else self.raw_dir
        sample_type = 'hard' if is_hard_sample else 'normal'

        timestamp_str = timestamp_obj.strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]

        filename = f"{timestamp_str}_ID{track_id}_{unique_id}.jpg"
        file_path = os.path.join(save_folder, filename)

        cv2.imwrite(file_path, plate_img)

        with open(self.log_file, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp_obj.isoformat(),
                camera_id,
                track_id,
                text,
                f"{conf:.4f}",
                file_path,
                sample_type
            ])

    def stop(self):
        print("Stopping Logger worker thread")
        self.stop_event.set()
        self.worker_thread.join()