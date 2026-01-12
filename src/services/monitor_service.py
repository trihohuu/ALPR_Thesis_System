import time
import logging
import json
import os
import numpy as np
from datetime import datetime

class MonitorService:
    def __init__(self, project_root, log_interval_frames=100, log_interval_seconds=60):
        self.interval_frames = log_interval_frames
        self.interval_seconds = log_interval_seconds

        self.logger = logging.getLogger("health_monitor")
        self.logger.setLevel(logging.INFO)
        
        log_dir = os.path.join(project_root, 'data')
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "health.log")

        if not self.logger.handlers:
            handler = logging.FileHandler(log_path)
            handler.setFormatter(logging.Formatter('%(message)s'))
            self.logger.addHandler(handler)

            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter('[MONITOR] %(message)s'))
            self.logger.addHandler(console_handler)

        self.reset_metrics()
        
    def reset_metrics(self):
        self.window_start_time = time.time()
        self.frame_count = 0
        self.total_process_time = 0.0
        self.conf_scores = []
        self.plate_counts = 0
        
    def update(self, process_time_ms, tracks):
        self.frame_count += 1
        self.total_process_time += process_time_ms

        frame_has_plate = False
        for trk in tracks:
            if trk.get('text'):
                conf = trk.get('ocr_conf', 0.0)
                if conf > 0: 
                    self.conf_scores.append(conf)
                
                frame_has_plate = True
        if frame_has_plate:
            self.plate_counts += 1
        self.check_and_log()

    def check_and_log(self):
        current_time = time.time()
        elapsed_time = current_time - self.window_start_time

        if self.frame_count >= self.interval_frames or elapsed_time >= self.interval_seconds:
            avg_fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
            avg_latency = self.total_process_time / self.frame_count if self.frame_count > 0 else 0
            
            avg_conf = 0.0
            if len(self.conf_scores) > 0:
                avg_conf = float(np.mean(self.conf_scores))

            log_payload = {
                "timestamp": datetime.now().isoformat(),
                "type": "health_metric",
                "metrics": {
                    "fps_avg": round(avg_fps, 2),
                    "latency_avg_ms": round(avg_latency, 2),
                    "ocr_conf_avg": round(avg_conf, 4),
                    "plates_detected_in_window": self.plate_counts,
                    "total_frames_processed": self.frame_count
                },
                "window_duration_sec": round(elapsed_time, 2)
            }

            self.logger.info(json.dumps(log_payload))
            self.reset_metrics()