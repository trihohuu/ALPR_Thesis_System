from ultralytics import YOLO
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR) 
DATA_YAML_PATH = os.path.join(ROOT_DIR, "data", "processed", "data.yaml")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
MODEL_TYPE = "yolov8n.pt" 

def train_model():
    print(f" Bắt đầu train model {MODEL_TYPE} trên GPU...")
    model = YOLO(MODEL_TYPE) 
    results = model.train(
        data = DATA_YAML_PATH,
        epochs = 50,
        imgsz = 640,
        batch = 16,
        device = 0, 
        project = MODELS_DIR, 
        name = "yolo",           
        exist_ok = True, 
        workers = 2, 
        cache = False, 
        amp = True
    )
    
    print("Training hoàn tất!")
    print(f"Kết quả (weights) được lưu tại: ALPR_Thesis_System/models/yolo/weights/best.pt")

if __name__ == '__main__':
    from multiprocessing import freeze_support
    print(ROOT_DIR)
    freeze_support()
    train_model()