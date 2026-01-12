from ultralytics import YOLO
import os
import wandb
from datetime import datetime

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR) 

DATA_AUG_PATH = os.path.join(ROOT_DIR, "data", "processed", "data_aug.yaml")
DATA_ORIG_PATH = os.path.join(ROOT_DIR, "data", "processed", "data.yaml")

if os.path.exists(DATA_AUG_PATH):
    DATA_YAML_PATH = DATA_AUG_PATH
    print(f"--> Đang sử dụng dữ liệu Augmented: {DATA_YAML_PATH}")
else:
    DATA_YAML_PATH = DATA_ORIG_PATH
    print(f"--> Đang sử dụng dữ liệu Gốc (Chưa augment): {DATA_YAML_PATH}")

MODELS_DIR = os.path.join(ROOT_DIR, "models")
MODEL_TYPE = "yolov8n.pt" 

def train_model():
    WANDB_PROJECT = "ALPR_Thesis_System"
    run_name = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    try:
        wandb.init(
            project=WANDB_PROJECT,
            name=run_name,
            job_type="training",
            config={
                "model": MODEL_TYPE,
                "epochs": 50,
                "batch_size": 16,
                "imgsz": 640,
                "data": DATA_YAML_PATH # Log data path
            }
        )
    except Exception as e:
        print(f"Lỗi :V")

    print(f" Bắt đầu train model {MODEL_TYPE} trên GPU")
    model = YOLO(MODEL_TYPE) 

    results = model.train(
        data = DATA_YAML_PATH, # Dùng file yaml đã chọn ở trên
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

    val_results = model.val(data=DATA_YAML_PATH, split='val')
    test_results = model.val(data=DATA_YAML_PATH, split='test')

    val_map = val_results.box.map if hasattr(val_results.box, 'map') else 0
    test_map = test_results.box.map if hasattr(test_results.box, 'map') else 0

    print(f"Val mAP: {val_map}, Test mAP: {test_map}")
    
    if wandb.run:
        wandb.log({
            "final_val_map": val_map,
            "final_test_map": test_map
        })
        wandb.finish()
    
    print(f"Weights lưu tại: {os.path.join(MODELS_DIR, 'yolo', 'weights', 'best.pt')}")

if __name__ == '__main__':
    from multiprocessing import freeze_support
    print(f"Root Dir: {ROOT_DIR}")
    freeze_support()
    train_model()