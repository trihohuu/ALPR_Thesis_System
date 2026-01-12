from pydantic import BaseModel
from typing import List, Optional, Any

class OCRResult(BaseModel):
    text: str = ""
    conf: float = 0.0
    
class TrackedObject(BaseModel):
    track_id: int
    box: List[float] # [x1, y1, x2, y2]
    class_id: int
    score: float
    ocr: Optional[OCRResult] = None
    crop_b64: Optional[str] = None # Dùng khi gửi qua API

class FrameResult(BaseModel):
    frame_id: int
    timestamp: float
    objects: List[TrackedObject]
    processing_time_ms: float