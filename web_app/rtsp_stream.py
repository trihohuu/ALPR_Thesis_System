
import cv2
import threading
import time
import os

class RTSPVideoStream:
    def __init__(self, src=0):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        self.stream = cv2.VideoCapture(src)
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False
        self.lock = threading.Lock()

    def start(self):
        t = threading.Thread(target=self.update, args=())
        t.daemon = True 
        t.start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return

            grabbed, frame = self.stream.read()
            
            with self.lock:
                self.grabbed = grabbed
                self.frame = frame
            time.sleep(0.005) 

    def read(self):
        with self.lock:
            return self.frame if self.grabbed else None

    def stop(self):
        self.stopped = True
        self.stream.release()