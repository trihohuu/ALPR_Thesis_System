from paddleocr import PaddleOCR
import logging

class LicensePlateOCR:
    def __init__(self, lang='en', use_gpu=True):
        logging.getLogger("ppocr").setLevel(logging.WARNING)
        self.ocr = PaddleOCR(use_angle_cls=True, lang=lang)
        print("Load xong OCR")

    def predict(self, image_array):
        if image_array is None:
            return "", 0.0
            
        try:
            result = self.ocr.ocr(image_array, cls=True)
            if not result or result[0] is None:
                return "", 0.0
            
            boxes = result[0]
            
            print(f"Số lượng chữ: {len(boxes)}")
            # Tính chiều cao trung bình của các box chữ để làm ngưỡng (threshold)
            # Box format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] -> chiều cao = y4 - y1
            heights = [abs(b[0][3][1] - b[0][0][1]) for b in boxes]
            avg_height = sum(heights) / len(heights) if heights else 20
            
            # Ngưỡng dòng = 1/2 chiều cao chữ (linh hoạt theo kích thước ảnh)
            y_threshold = avg_height * 0.5 

            # Sắp xếp sơ bộ theo Y (tâm của box thay vì cạnh trên)
            # b[0] là list 4 điểm, lấy trung bình cộng Y của 4 điểm để ra tâm Y
            def get_center_y(box):
                return sum([p[1] for p in box[0]]) / 4

            boxes = sorted(boxes, key=get_center_y)
            
            final_boxes = []
            current_line = [boxes[0]]
            
            for i in range(1, len(boxes)):
                box = boxes[i]
                prev_box = current_line[-1]
                
                # So sánh tâm Y của box hiện tại với tâm Y của box đầu dòng
                if abs(get_center_y(box) - get_center_y(current_line[0])) < y_threshold:
                    current_line.append(box)
                else:
                    # Hết dòng -> Sort dòng cũ theo X (từ trái qua phải)
                    current_line = sorted(current_line, key=lambda x: x[0][0][0])
                    final_boxes.extend(current_line)
                    current_line = [box]
            
            # Xử lý dòng cuối cùng
            current_line = sorted(current_line, key=lambda x: x[0][0][0])
            final_boxes.extend(current_line)
            
            # Ghép text
            full_text = ""
            total_score = 0
            for line in final_boxes:
                text, score = line[1]
                print(f"DEBUG: {text} | Y-Center: {get_center_y(line):.1f}") 
                full_text += text + " "
                total_score += score
            
            avg_score = total_score / len(final_boxes)
            return self._clean_text(full_text), avg_score

        except Exception as e:
            print(f"Lỗi OCR: {e}")
            return "", 0.0

    def _clean_text(self, text):
        import re
        # Chỉ giữ lại chữ cái và số, xóa dấu cách, dấu chấm, gạch ngang
        # Biển số VN: 59-P1 123.45 -> 59P112345
        clean = re.sub(r'[^a-zA-Z0-9]', '', text) 
        return clean.upper()