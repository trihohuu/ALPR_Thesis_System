# Trích dẫn từ file logger_service.py
def _save_to_disk(self, packet):
    # ...
    is_hard_sample = (conf < 0.8) or (len(text) == 0)
    save_folder = self.hard_dir if is_hard_sample else self.raw_dir
    
    # ... Lưu ảnh và ghi log CSV
    cv2.imwrite(file_path, plate_img)