import streamlit as st
import cv2
import time
import os
import torch
import sys
import tempfile
import numpy as np
import Levenshtein
from difflib import SequenceMatcher
from PIL import Image

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_path = os.path.join(project_root, 'src')
sys.path.append(src_path)

from rtsp_stream import RTSPVideoStream
from pipeline import ALPRPipeline

st.set_page_config(page_title="ALPR Thesis System", layout="wide")

st.markdown("""
<style>
    .reportview-container { margin-top: -2em; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    
    /* CSS cho Gallery biển số */
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

st.title("Automatic License Plate Recognition System")

if 'detected_plates' not in st.session_state:
    st.session_state['detected_plates'] = {} 

def reset_session():
    st.session_state['detected_plates'] = {}

@st.cache_resource
def load_pipeline():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, '..', 'models', 'yolo', 'weights', 'best.pt') 
    
    use_gpu = torch.cuda.is_available()
    print(f"Loading pipeline... GPU: {use_gpu}")
    
    try:
        app_pipeline = ALPRPipeline(yolo_path=model_path, use_gpu=use_gpu)
        return app_pipeline
    except Exception as e:
        st.error(f"Không load được Model: {e}")
        return None

pipeline = load_pipeline()

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
                    is_duplicate = True
                    duplicate_id = stored_id
                    break

                if similar(txt, stored_txt) > 0.8 and abs(tid - stored_id) < 20:
                    is_duplicate = True
                    duplicate_id = stored_id
                    break
            
            if is_duplicate:
                current_data = st.session_state['detected_plates'][duplicate_id]
                if conf > current_data['conf'] or len(txt) > len(current_data['text']):
                    st.session_state['detected_plates'][duplicate_id] = {
                        'img': img, 
                        'text': txt, 
                        'conf': conf
                    }
            else:
                st.session_state['detected_plates'][tid] = {
                    'img': img, 
                    'text': txt, 
                    'conf': conf
                }

def render_gallery_ui(placeholder):
    if not st.session_state['detected_plates']:
        return

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
                </div>
                """, unsafe_allow_html=True)

# --- 4. GIAO DIỆN SIDEBAR ---
st.sidebar.header("Cấu hình Nguồn vào")
source_option = st.sidebar.radio(
    "Chọn nguồn dữ liệu:", 
    ("Upload Video/Ảnh", "RTSP Camera", "Webcam Laptop")
)

run_system = False
uploaded_file = None
rtsp_url = 0

if source_option == "Upload Video/Ảnh":
    uploaded_file = st.sidebar.file_uploader("Chọn file...", type=['mp4', 'avi', 'mov', 'jpg', 'png', 'jpeg'])
    if uploaded_file is not None:
        run_system = st.sidebar.button("Bắt đầu xử lý")
        if st.sidebar.button("Reset kết quả"):
            reset_session()
            st.rerun()

elif source_option == "RTSP Camera":
    rtsp_url = st.sidebar.text_input("RTSP URL:", value="")
    run_system = st.sidebar.checkbox("Bắt đầu chạy Stream")
    if not run_system:
        reset_session() 

elif source_option == "Webcam Laptop":
    rtsp_url = 0
    run_system = st.sidebar.checkbox("Bắt đầu chạy Webcam")
    if not run_system:
        reset_session()

# --- 5. MAIN LOGIC ---
st_frame = st.empty()    
st_gallery = st.empty() 
st_status = st.empty()

if run_system and pipeline:
    
    # === TRƯỜNG HỢP 1: UPLOAD FILE ẢNH/VIDEO ===
    if source_option == "Upload Video/Ảnh" and uploaded_file is not None:
        file_type = uploaded_file.name.split('.')[-1].lower()
        
        # -- XỬ LÝ ẢNH TĨNH --
        if file_type in ['jpg', 'png', 'jpeg']:
            st_status.info("Đang xử lý ảnh...")

            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, 1)

            processed_frame, tracks = pipeline.process_single_frame(frame, frame_idx=0)
            update_gallery(tracks)

            frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            st_frame.image(frame_rgb, channels="RGB", use_container_width=True)
            render_gallery_ui(st_gallery)
            st_status.success("Hoàn tất!")

        # -- XỬ LÝ VIDEO --
        elif file_type in ['mp4', 'avi', 'mov']:
            tfile = tempfile.NamedTemporaryFile(delete=False) 
            tfile.write(uploaded_file.read())
            
            cap = cv2.VideoCapture(tfile.name)
            st_status.info("Đang xử lý video...")
            
            frame_idx = 0
            writer = None
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                processed_frame, tracks = pipeline.process_single_frame(frame, frame_idx)
                update_gallery(tracks)
                
                frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                st_frame.image(frame_rgb, channels="RGB", use_container_width=True)
                if frame_idx % 5 == 0:
                    render_gallery_ui(st_gallery)
                
                frame_idx += 1
                if writer:
                    writer.write(processed_frame)

            cap.release()
            if writer:
                writer.release()
            pipeline.save_final_results()
            render_gallery_ui(st_gallery) 
            st_status.success("Đã chạy xong video!")

    # === LIVE STREAM (WEBCAM/RTSP) ===
    elif source_option in ["RTSP Camera", "Webcam Laptop"]:
        st_status.info("Đang kết nối tới Camera...")
        
        try:
            streamer = RTSPVideoStream(rtsp_url).start()
        except Exception as e:
            st.error(f"Lỗi khởi tạo: {e}")
            st.stop()
            
        time.sleep(1.0)
        
        if not streamer.grabbed:
            st.error("Không kết nối được camera.")
            streamer.stop()
        else:
            st_status.success("Đã kết nối! Đang xử lý...")
            frame_count = 0
            fps_start = time.time()
            
            while run_system:
                frame = streamer.read()
                if frame is None:
                    continue
                processed_frame, tracks = pipeline.process_single_frame(frame, frame_count)
                update_gallery(tracks)
                frame_count += 1
                if frame_count % 10 == 0:
                    fps = 10 / (time.time() - fps_start)
                    fps_start = time.time()
                    cv2.putText(processed_frame, f"FPS: {fps:.1f}", (20, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                st_frame.image(frame_rgb, channels="RGB", use_container_width=True)
                if frame_count % 10 == 0:
                    render_gallery_ui(st_gallery)
            
            streamer.stop()
else:
    if not pipeline:
        st.warning("Hệ thống chưa load được model.")
    elif not run_system:
        st.info("Hãy chọn nguồn và nhấn Bắt đầu.")