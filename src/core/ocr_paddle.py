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
            heights = [abs(b[0][3][1] - b[0][0][1]) for b in boxes]
            avg_height = sum(heights) / len(heights) if heights else 20

            y_threshold = avg_height * 0.5 

            def get_center_y(box):
                return sum([p[1] for p in box[0]]) / 4

            boxes = sorted(boxes, key=get_center_y)
            
            final_boxes = []
            current_line = [boxes[0]]
            
            for i in range(1, len(boxes)):
                box = boxes[i]
                prev_box = current_line[-1]

                if abs(get_center_y(box) - get_center_y(current_line[0])) < y_threshold:
                    current_line.append(box)
                else:
                    current_line = sorted(current_line, key=lambda x: x[0][0][0])
                    final_boxes.extend(current_line)
                    current_line = [box]

            current_line = sorted(current_line, key=lambda x: x[0][0][0])
            final_boxes.extend(current_line)

            full_text = ""
            total_score = 0
            for line in final_boxes:
                text, score = line[1]
                # print(f"DEBUG: {text} | Y-Center: {get_center_y(line):.1f}") 
                full_text += text + "-"
                total_score += score
            
            avg_score = total_score / len(final_boxes)

            cleaned_text = self._clean_text(full_text)    
            final_text = self._heuristic_uk_format(cleaned_text) 
            
            return final_text, avg_score

        except Exception as e:
            print(f"OCR ERROR?: {e}")
            return "", 0.0

    def _clean_text(self, text):
        import re
        clean = re.sub(r'[^a-zA-Z0-9]', '', text) 
        return clean.upper()
    
    def _heuristic_uk_format(self, text):
        if len(text) != 7:
            return text 

        dict_char_to_num = {'O': '0', 'D': '0', 'Q': '0', 'I': '1', 'J': '1', 'L': '1', 'Z': '2', 'B': '8', 'S': '5', 'G': '6'}
        dict_num_to_char = {'0': 'O', '1': 'I', '2': 'Z', '5': 'S', '6': 'G', '8': 'B'}
        
        text_list = list(text)

        for i in [0, 1]:
            if text_list[i] in dict_num_to_char:
                text_list[i] = dict_num_to_char[text_list[i]]

        for i in [2, 3]:
            if text_list[i] in dict_char_to_num:
                text_list[i] = dict_char_to_num[text_list[i]]

        for i in [4, 5, 6]:
            if text_list[i] in dict_num_to_char:
                text_list[i] = dict_num_to_char[text_list[i]]

        return "".join(text_list)