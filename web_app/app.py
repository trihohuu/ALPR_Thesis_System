import streamlit as st
import cv2
import time
import os
import sys
import tempfile
import numpy as np
import requests
import base64
from difflib import SequenceMatcher

# --- SETUP ĐƯỜNG DẪN ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_path = os.path.join(project_root, 'src')
sys.path.append(src_path)

from rtsp_stream import RTSPVideoStream

# --- CẤU HÌNH API ---
API_URL = os.getenv("API_URL", "http://localhost:8000/process_frame")

st.set_page_config(page_title="Hệ thống nhận diện biển số xe", layout="wide")

# --- CSS STYLES ---
st.markdown("""
<style>
    .reportview-container { margin-top: -2em; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .plate-card {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 10px;
        border: 2px solid #4CAF50;
    }
    .plate-text {
        font-weight: bold;
        color: #d32f2f;
        font-size: 18px;
    }
    .plate-conf {
        font-size: 12px;
        color: #555;
    }
</style>
""", unsafe_allow_html=True)

st.title("Hệ thống nhận diện biển số xe")

if 'detected_plates' not in st.session_state:
    st.session_state['detected_plates'] = {} 

def reset_session():
    st.session_state['detected_plates'] = {}

def base64_to_numpy(base64_string):
    if not base64_string: 
        return None
    img_bytes = base64.b64decode(base64_string)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

def draw_tracks_on_frame(frame, tracks):
    for trk in tracks:
        if 'box' in trk:
            x1, y1, x2, y2 = map(int, trk['box'])

            label = trk.get('text', '')
            conf = trk.get('ocr_conf', 0.0)
            track_id = trk.get('id', '?')

            color = (0, 255, 0) 

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            caption = f"{label} ({conf:.2f})" if label else f"ID: {track_id}"

            (t_w, t_h), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - 25), (x1 + t_w, y1), color, -1)
            cv2.putText(frame, caption, (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
    return frame

# API section
def call_process_api(frame, frame_idx):
    try:
        _, img_encoded = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        
        files = {'file': ('frame.jpg', img_encoded.tobytes(), 'image/jpeg')}
        params = {'frame_idx': frame_idx}
        
        response = requests.post(API_URL, files=files, params=params, timeout=5)
        
        if response.status_code == 200:
            result = response.json()
            tracks = result.get('tracks', [])

            processed_frame = draw_tracks_on_frame(frame, tracks)

            for trk in tracks:
                if trk.get('plate_img'):
                    trk['plate_img'] = base64_to_numpy(trk['plate_img'])
                if trk.get('best_img'):
                    trk['best_img'] = base64_to_numpy(trk['best_img'])
            
            return processed_frame, tracks
        else:
            return frame, []
            
    except Exception as e:
        return frame, []

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def update_gallery(tracks):
    for trk in tracks:
        tid = trk['id']
        txt = trk.get('best_text', trk.get('text', ''))
        conf = trk.get('best_conf', trk.get('ocr_conf', 0.0))
        img = trk.get('best_img') if trk.get('best_img') is not None else trk.get('plate_img')
        
        if txt and img is not None and len(txt) >= 5:
            is_duplicate = False
            duplicate_id = None
            for stored_id, data in st.session_state['detected_plates'].items():
                stored_txt = data['text']
                if stored_id == tid:
                    is_duplicate = True; duplicate_id = stored_id; break
                if similar(txt, stored_txt) > 0.8 and abs(tid - stored_id) < 20:
                    is_duplicate = True; duplicate_id = stored_id; break
            
            if is_duplicate:
                current_data = st.session_state['detected_plates'][duplicate_id]
                if conf > current_data['conf'] or len(txt) > len(current_data['text']):
                    st.session_state['detected_plates'][duplicate_id] = {'img': img, 'text': txt, 'conf': conf}
            else:
                st.session_state['detected_plates'][tid] = {'img': img, 'text': txt, 'conf': conf}

def render_gallery_ui(placeholder):
    if not st.session_state['detected_plates']: return
    sorted_plates = sorted(st.session_state['detected_plates'].items(), key=lambda x: x[0], reverse=True)
    with placeholder.container():
        st.write("### Các biển số đã nhận diện:")
        cols = st.columns(5)
        for idx, (tid, data) in enumerate(sorted_plates):
            col = cols[idx % 5]
            with col:
                img_rgb = cv2.cvtColor(data['img'], cv2.COLOR_BGR2RGB)
                st.image(img_rgb, use_container_width=True)
                st.markdown(f"""
                <div class="plate-card">
                    <div class="plate-text">{data['text']}</div>
                    <div class="plate-conf">Conf: {data['conf']:.2f} | ID: {tid}</div>
                </div>""", unsafe_allow_html=True)

st.sidebar.header("Cấu hình Nguồn vào")
source_option = st.sidebar.radio("Chọn nguồn dữ liệu:", ("Upload Video/Ảnh", "RTSP Camera", "Webcam Laptop"))

run_system = False
uploaded_file = None
rtsp_url = 0

if source_option == "Upload Video/Ảnh":
    uploaded_file = st.sidebar.file_uploader("Chọn file...", type=['mp4', 'avi', 'mov', 'jpg', 'png', 'jpeg'])
    if uploaded_file is not None:
        run_system = st.sidebar.button("Bắt đầu xử lý")
        if st.sidebar.button("Reset kết quả"): reset_session(); st.rerun()

elif source_option == "RTSP Camera":
    rtsp_url = st.sidebar.text_input("RTSP URL:", value="")
    run_system = st.sidebar.checkbox("Bắt đầu chạy Stream")
    if not run_system: reset_session()

elif source_option == "Webcam Laptop":
    rtsp_url = 0
    run_system = st.sidebar.checkbox("Bắt đầu chạy Webcam")
    if not run_system: reset_session()

st_frame = st.empty()    
st_gallery = st.empty() 
st_status = st.empty()

# Kiểm tra API
try:
    requests.get("http://localhost:8000/docs", timeout=1)
except:
    st.warning("Hãy chạy Backend trước")

if run_system:
    # 1. UPLOAD FILE
    if source_option == "Upload Video/Ảnh" and uploaded_file is not None:
        file_type = uploaded_file.name.split('.')[-1].lower()
        if file_type in ['jpg', 'png', 'jpeg']:
            st_status.info("Đang xử lý")
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, 1)
            processed_frame, tracks = call_process_api(frame, 0)
            update_gallery(tracks)
            st_frame.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
            render_gallery_ui(st_gallery)
            st_status.success("Done")
        
        elif file_type in ['mp4', 'avi', 'mov']:
            tfile = tempfile.NamedTemporaryFile(delete=False) 
            tfile.write(uploaded_file.read())
            cap = cv2.VideoCapture(tfile.name)
            st_status.info("Đang xử lý")
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                processed_frame, tracks = call_process_api(frame, frame_idx)
                update_gallery(tracks)
                
                st_frame.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                if frame_idx % 5 == 0: render_gallery_ui(st_gallery)
                frame_idx += 1
            cap.release()
            st_status.success("Hoàn tất")

    # 2. STREAM / WEBCAM
    elif source_option in ["RTSP Camera", "Webcam Laptop"]:
        st_status.info("Đang kết nối")
        streamer = RTSPVideoStream(rtsp_url).start()
        time.sleep(1.0)
        
        if not streamer.grabbed:
            st.error("Không thể kết nối camera.")
            streamer.stop()
        else:
            st_status.success("Đang chạy Stream (Client-Side Rendering)...")
            frame_count = 0
            while run_system:
                frame = streamer.read()
                if frame is None: continue
                
                # Gọi API lấy tọa độ -> Tự vẽ -> Hiển thị
                processed_frame, tracks = call_process_api(frame, frame_count)
                
                update_gallery(tracks)
                frame_count += 1
                
                st_frame.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                if frame_count % 10 == 0:
                    render_gallery_ui(st_gallery)
            
            streamer.stop()