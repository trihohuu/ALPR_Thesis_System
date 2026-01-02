import easyocr
import logging
import numpy as np

class LicensePlateEasyOCR:
    def __init__(self, use_gpu=True):
        # Tắt logging debug của easyocr cho đỡ rối
        logging.getLogger("easyocr").setLevel(logging.WARNING)
        
        print("Đang load EasyOCR model (lần đầu sẽ cần tải model về)...")
        # 'en' là đủ cho biển số. 
        # gpu=True sẽ tận dụng CUDA (bạn đang dùng Laptop RTX 4050 nên sẽ rất nhanh)
        self.reader = easyocr.Reader(['en'], gpu=use_gpu)
        print("Load xong EasyOCR")

    def predict(self, image_array):
            if image_array is None:
                return "", 0.0
                
            try:
                # allowlist giúp lọc nhiễu ngay từ đầu
                allow_chars = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-'
                results = self.reader.readtext(image_array, allowlist=allow_chars)
                
                if not results:
                    return "", 0.0

                # --- BẮT ĐẦU LOGIC SẮP XẾP (Giống Paddle) ---
                # EasyOCR trả về: ([[x1,y1],[x2,y2],[x3,y3],[x4,y4]], text, conf)
                
                # Hàm lấy tọa độ Y trung tâm của box
                def get_center_y(item):
                    box = item[0] 
                    # box là list 4 điểm: tl, tr, br, bl. Lấy trung bình Y.
                    return sum([p[1] for p in box]) / 4

                # 1. Sắp xếp sơ bộ theo chiều dọc (Y)
                results = sorted(results, key=get_center_y)
                
                # 2. Tính chiều cao trung bình để phân dòng
                if len(results) > 0:
                    heights = [abs(item[0][2][1] - item[0][0][1]) for item in results]
                    avg_height = sum(heights) / len(heights)
                else:
                    avg_height = 20
                    
                y_threshold = avg_height * 0.5 

                final_boxes = []
                current_line = [results[0]]
                
                # 3. Gom nhóm các box nằm trên cùng một dòng
                for i in range(1, len(results)):
                    box = results[i]
                    # So sánh Y của box hiện tại với Y của dòng đang xét
                    if abs(get_center_y(box) - get_center_y(current_line[0])) < y_threshold:
                        current_line.append(box)
                    else:
                        # Hết dòng cũ -> sort dòng cũ từ trái qua phải (theo X)
                        current_line = sorted(current_line, key=lambda x: x[0][0][0])
                        final_boxes.extend(current_line)
                        current_line = [box] # Bắt đầu dòng mới

                # Xử lý dòng cuối cùng
                current_line = sorted(current_line, key=lambda x: x[0][0][0])
                final_boxes.extend(current_line)
                # --- KẾT THÚC LOGIC SẮP XẾP ---

                full_text = ""
                total_conf = 0.0
                for item in final_boxes:
                    _, text, conf = item
                    full_text += text + "-" # Thêm dấu gạch để dễ nhìn debugging
                    total_conf += conf
                
                avg_conf = total_conf / len(final_boxes)
                return self._clean_text(full_text), avg_conf

            except Exception as e:
                print(f"EasyOCR Error: {e}")
                return "", 0.0

    def _clean_text(self, text):
        import re
        # Loại bỏ các ký tự không phải chữ số hoặc chữ cái (giữ lại dấu gạch ngang nếu muốn)
        clean = re.sub(r'[^a-zA-Z0-9]', '', text) 
        return clean.upper()