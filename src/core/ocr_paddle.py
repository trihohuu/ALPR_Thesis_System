from paddleocr import PaddleOCR
import logging

class LicensePlateOCR:
    def __init__(self, lang='en', use_gpu=True):
        logging.getLogger("ppocr").setLevel(logging.WARNING)
        self.ocr = PaddleOCR(use_angle_cls=True, lang=lang)
        print("Load xong OCR")

    def predict(self, image_array):
        """
        Input: Ảnh (numpy array) đã được crop (chỉ chứa biển số)
        Output: (text, score) - Text biển số và độ tin cậy
        """
        if image_array is None:
            return "", 0.0
            
        try:
            result = self.ocr.ocr(image_array, cls=True)
            if not result or result[0] is None:
                return "", 0.0
            
            # Ghép các dòng text lại (với biển xe máy 2 dòng)
            full_text = ""
            total_score = 0
            count = 0
            
            for line in result[0]:
                text, score = line[1]
                full_text += text + " "
                total_score += score
                count += 1
            
            # Tính trung bình score
            avg_score = total_score / count if count > 0 else 0
            
            # Làm sạch text (Xóa ký tự lạ, dấu chấm, gạch ngang thừa)
            clean_text = self._clean_text(full_text)
            
            return clean_text, avg_score

        except Exception as e:
            print(f"Lỗi OCR: {e}")
            return "", 0.0

    def _clean_text(self, text):
        import re
        # Chỉ giữ lại chữ cái và số, xóa dấu cách, dấu chấm, gạch ngang
        # Biển số VN: 59-P1 123.45 -> 59P112345
        clean = re.sub(r'[^a-zA-Z0-9]', '', text) 
        return clean.upper()