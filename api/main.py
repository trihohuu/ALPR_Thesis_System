# api/main.py
import sys
import os
import cv2
import numpy as np
import base64
import uvicorn
import time
from fastapi import FastAPI, File, UploadFile, HTTPException

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.pipeline import ALPRPipeline
from src.services.logger_service import LoggerService
from src.services.monitor_service import MonitorService

app = FastAPI()

# Biến toàn cục
pipeline_model = None
logger_service = None
monitor_service = None

@app.on_event("startup")
def startup_event():
    global pipeline_model, logger_service, monitor_service
    
    model_path = os.path.join(project_root, 'models', 'yolo', 'weights', 'best.pt')
    if not os.path.exists(model_path):
        print(f"Không tìm thấy model tại {model_path}")
    
    try:
        pipeline_model = ALPRPipeline(yolo_path=model_path, use_gpu=True)
        print("Pipeline AI: Ready.")
    except Exception as e:
        print(f"Lỗi khởi tạo Model: {e}")

    try:
        logger_service = LoggerService(project_root=project_root)
        monitor_service = MonitorService(log_interval_frames=50, log_interval_seconds=30, project_root=project_root)
        print("Services (Logger/Monitor): Ready.")
    except Exception as e:
        print(f"Lỗi khởi tạo Services: {e}")

@app.on_event("shutdown")
def shutdown_event():
    global logger_service
    if logger_service:
        logger_service.stop()

def numpy_to_base64(img_array):
    if img_array is None: return None
    _, buffer = cv2.imencode('.jpg', img_array)
    return base64.b64encode(buffer).decode('utf-8')

def clean_numpy_data(data):
    if isinstance(data, dict):
        return {k: clean_numpy_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_numpy_data(v) for v in data]
    elif isinstance(data, (np.int64, np.int32, np.int_)): return int(data)
    elif isinstance(data, (np.float32, np.float64, np.float_)): return float(data)
    elif isinstance(data, np.ndarray): return data.tolist()
    else: return data


@app.post("/process_frame")
async def process_frame(
    file: UploadFile = File(...), 
    frame_idx: int = 0
):
    global pipeline_model, logger_service, monitor_service
    
    if pipeline_model is None:
        raise HTTPException(status_code=503, detail="Model chưa sẵn sàng.")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="File ảnh lỗi.")

    start_time = time.time()
    
    try:
        _, tracks = pipeline_model.process_single_frame(frame, frame_idx)

        process_time_ms = (time.time() - start_time) * 1000

        if monitor_service:
            monitor_service.update(process_time_ms, tracks)

        if logger_service:
            for trk in tracks:
                if trk.get('text') or trk.get('ocr_conf', 0) > 0:
                    logger_service.log_detection(trk, camera_id="CAM_API")

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý: {str(e)}")

    serializable_tracks = []
    for trk in tracks:
        trk_data = trk.copy()
        if 'plate_img' in trk_data:
            trk_data['plate_img'] = numpy_to_base64(trk_data['plate_img'])
        if 'best_img' in trk_data:
            trk_data['best_img'] = numpy_to_base64(trk_data['best_img'])
        serializable_tracks.append(trk_data)

    result = {
        "processed_frame": None, 
        "tracks": serializable_tracks,
        "server_time_ms": process_time_ms
    }

    return clean_numpy_data(result)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)