import sys
import os
import cv2
import numpy as np
import base64
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_path = os.path.join(project_root, 'src')

from src.pipeline import ALPRPipeline
app = FastAPI()

pipeline_model = None

@app.on_event("startup")
def load_model():
    global pipeline_model
    model_path = os.path.join(project_root, 'models', 'yolo', 'weights', 'best.pt')
    
    if not os.path.exists(model_path):
        print(f"Không tìm thấy model tại {model_path}")
    try:
        pipeline_model = ALPRPipeline(yolo_path=model_path, use_gpu=True)
        print("Ready.")
    except Exception as e:
        print(f"Lỗi khởi tạo model: {e}")

def numpy_to_base64(img_array):
    if img_array is None:
        return None
    _, buffer = cv2.imencode('.jpg', img_array)
    return base64.b64encode(buffer).decode('utf-8')

def clean_numpy_data(data):
    if isinstance(data, dict):
        return {k: clean_numpy_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_numpy_data(v) for v in data]
    elif isinstance(data, (np.int64, np.int32, np.int_)): 
        return int(data)
    elif isinstance(data, (np.float32, np.float64, np.float_)):
        return float(data)
    elif isinstance(data, np.ndarray): 
        return data.tolist()
    else:
        return data

@app.post("/process_frame")
async def process_frame(
    file: UploadFile = File(...), 
    frame_idx: int = 0
):
    global pipeline_model
    if pipeline_model is None:
        raise HTTPException(status_code=503, detail="Model chưa sẵn sàng.")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="File ảnh không hợp lệ.")

    try:
        _, tracks = pipeline_model.process_single_frame(frame, frame_idx)
    except Exception as e:
        print(f"Lỗi Pipeline: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý AI: {str(e)}")

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
        "tracks": serializable_tracks
    }

    return clean_numpy_data(result)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)