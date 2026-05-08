import streamlit as st
from ultralytics import YOLO
import os
import cv2
import pandas as pd
import tempfile
import numpy as np
import queue
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import torch

# Fix Streamlit watcher bug dengan PyTorch
try:
    torch.classes.__path__ = []
except Exception:
    pass

@st.cache_resource
def get_webrtc_queue():
    return queue.Queue(maxsize=1)

webrtc_queue = get_webrtc_queue()

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="MBG Food.",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }

    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}

    .stApp {
        background-color: #fafafa;
    }

    .top-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem 0;
        border-bottom: 1px solid #eaeaea;
        margin-bottom: 3rem;
    }
    
    .logo {
        font-size: 1.5rem;
        font-weight: 800;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        color: #111827;
    }
    
    .status-badge {
        background-color: #d1fae5;
        color: #059669;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        border: 1px solid #a7f3d0;
    }

    .main-heading {
        font-size: 3.2rem;
        font-weight: 800;
        line-height: 1.15;
        color: #111827;
        margin-bottom: 1.5rem;
        letter-spacing: -0.02em;
    }

    .main-heading span {
        color: #10B981;
    }

    .sub-heading {
        font-size: 1.1rem;
        color: #6B7280;
        line-height: 1.6;
        margin-bottom: 2.5rem;
        max-width: 90%;
    }

    .feature-list {
        display: flex;
        gap: 1.5rem;
        margin-bottom: 2rem;
        flex-wrap: wrap;
    }

    .feature-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-weight: 600;
        color: #4B5563;
        font-size: 0.95rem;
    }

    /* Metrics styling */
    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #f3f4f6;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
    }
    div[data-testid="stMetric"] label {
        color: #6b7280 !important;
        font-weight: 500;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #111827 !important;
        font-weight: 700;
    }

    .food-tag {
        display: inline-block;
        background: #ecfdf5;
        color: #047857;
        padding: 6px 16px;
        border-radius: 20px;
        margin: 4px;
        font-size: 0.9rem;
        font-weight: 600;
        border: 1px solid #a7f3d0;
    }
    
    .nutrition-header {
        font-size: 1.25rem;
        font-weight: 700;
        color: #111827;
        margin-top: 1rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #10B981;
        display: inline-block;
        padding-bottom: 0.25rem;
    }

    /* Uploader styling override to look more like the box in the image */
    .stFileUploader {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -2px rgba(0, 0, 0, 0.02);
        border: 1px solid #f3f4f6;
    }
    
    .footer-text {
        text-align: center;
        color: #9CA3AF;
        font-size: 0.875rem;
        margin-top: 5rem;
        padding-top: 2rem;
        border-top: 1px solid #E5E7EB;
        padding-bottom: 2rem;
    }

    /* Center Radio Buttons for Top Nav */
    div.stRadio > div[role="radiogroup"] {
        flex-direction: row;
        justify-content: center;
        gap: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Load Model & Data ────────────────────────────────────────
@st.cache_resource
def load_model():
    return YOLO("best.pt")

@st.cache_data
def load_tkpi():
    return pd.read_csv("data_tkpi.csv")

model = load_model()
df_tkpi = load_tkpi()


# ── Helper Functions ─────────────────────────────────────────
def get_detected_classes(results):
    """Extract detected class names from YOLO results."""
    detected = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        cls_name = results[0].names[cls_id]
        conf = float(box.conf[0])
        detected.append({"nama": cls_name, "confidence": conf})
    return detected

def build_nutrition_table(detected_items):
    """Build nutrition DataFrame from detected food items."""
    if not detected_items:
        return None

    class_counts = {}
    for item in detected_items:
        name = item["nama"]
        class_counts[name] = class_counts.get(name, 0) + 1

    rows = []
    for cls_name, count in class_counts.items():
        match = df_tkpi[df_tkpi["label_model"] == cls_name]
        if not match.empty:
            row = match.iloc[0]
            rows.append({
                "Makanan": row["nama_pangan"],
                "Jumlah": count,
                "Energi (kkal)": row["energi_kkal"],
                "Protein (g)": row["protein_g"],
                "Lemak (g)": row["lemak_g"],
                "Karbohidrat (g)": row["karbohidrat_g"],
                "Serat (g)": row["serat_g"],
            })

    return pd.DataFrame(rows) if rows else None

def show_nutrition_summary(df_nutrition):
    """Display nutrition summary metrics."""
    total_energi = (df_nutrition["Energi (kkal)"] * df_nutrition["Jumlah"]).sum()
    total_protein = (df_nutrition["Protein (g)"] * df_nutrition["Jumlah"]).sum()
    total_lemak = (df_nutrition["Lemak (g)"] * df_nutrition["Jumlah"]).sum()
    total_karbo = (df_nutrition["Karbohidrat (g)"] * df_nutrition["Jumlah"]).sum()
    total_serat = (df_nutrition["Serat (g)"] * df_nutrition["Jumlah"]).sum()

    st.markdown('<div class="nutrition-header">📋 Estimasi Total Nutrisi</div>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🔥 Energi", f"{total_energi:.0f} kkal")
    c2.metric("🥩 Protein", f"{total_protein:.1f} g")
    c3.metric("🧈 Lemak", f"{total_lemak:.1f} g")
    c4.metric("🍚 Karbohidrat", f"{total_karbo:.1f} g")
    c5.metric("🥬 Serat", f"{total_serat:.1f} g")

    st.info("⚠️ Catatan: Estimasi nutrisi dihitung berdasarkan data TKPI 2017 dengan asumsi 1 porsi = 100 gram per makanan yang terdeteksi.")


# ── Top Navigation / Header ──────────────────────────────────
col_logo, col_nav, col_status = st.columns([1, 2, 1])

with col_logo:
    st.markdown('<div class="logo">🍽️ MBG Food.</div>', unsafe_allow_html=True)

with col_nav:
    mode = st.radio("Navigasi", ["Upload Gambar", "Streaming Real-Time"], label_visibility="collapsed")

with col_status:
    st.markdown('<div style="display: flex; justify-content: flex-end;"><div class="status-badge">🟢 YOLOv11 Ready</div></div>', unsafe_allow_html=True)


# ── Main Content Area ────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)

col_text, col_action = st.columns([1.2, 1])

with col_text:
    st.markdown("""
        <div class="main-heading">
            Deteksi Makanan Bergizi Gratis(MBG) &<br>
            <span>Estimasi Nutrisi AI</span>
        </div>
        <div class="sub-heading">
            Mendukung Program Makan Bergizi Gratis (MBG) melalui deteksi otomatis jenis makanan dan estimasi kandungan nutrisi berbasis AI.
        </div>
        <div class="feature-list">
            <div class="feature-item">⚡ Deteksi Instan</div>
            <div class="feature-item">🎯 27 Jenis Makanan</div>
            <div class="feature-item">📊 Estimasi Nutrisi</div>
        </div>
    """, unsafe_allow_html=True)

with col_action:
    if mode == "Upload Gambar":
        uploaded_file = st.file_uploader(
            "Drag & drop foto makanan (JPG, PNG, WEBP — Max 10MB)",
            type=["jpg", "jpeg", "png", "webp"]
        )
    else:
        st.markdown("### 📷 Live Stream Camera")
        run_camera = st.checkbox("Mulai Kamera Real-Time", key="run_cam")
        st.caption("Pastikan memberikan izin akses kamera (localhost).")


# ── Logic for Modes ──────────────────────────────────────────
if mode == "Upload Gambar" and 'uploaded_file' in locals() and uploaded_file is not None:
    st.markdown("---")
    
    file_bytes = uploaded_file.read()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    tmp.write(file_bytes)
    tmp.close()
    file_path = tmp.name

    img_check = cv2.imread(file_path)
    if img_check is None:
        st.error("❌ Gambar tidak dapat dibaca. Pastikan file gambar tidak rusak.")
        os.unlink(file_path)
        st.stop()

    try:
        with st.spinner("🔍 Mendeteksi makanan..."):
            results = model.predict(source=file_path, conf=0.25)
    except Exception as e:
        st.error(f"❌ Gagal mendeteksi makanan: {e}")
        os.unlink(file_path)
        st.stop()
    finally:
        if os.path.exists(file_path):
            os.unlink(file_path)

    plotted = results[0].plot()
    plotted = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
    detected_items = get_detected_classes(results)

    col_img, col_info = st.columns([1, 1])

    with col_img:
        st.markdown("#### 📸 Hasil Deteksi")
        st.image(plotted, use_container_width=True)

    with col_info:
        st.markdown("#### 🍴 Makanan Terdeteksi")
        if detected_items:
            tags_html = ""
            for item in detected_items:
                tags_html += f'<span class="food-tag">{item["nama"]} ({item["confidence"]:.0%})</span>'
            st.markdown(tags_html, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="nutrition-header">📊 Informasi Nutrisi per 100g</div>', unsafe_allow_html=True)
            df_nutrition = build_nutrition_table(detected_items)
            if df_nutrition is not None:
                st.dataframe(df_nutrition, use_container_width=True, hide_index=True)
        else:
            st.info("Tidak ada makanan yang terdeteksi pada gambar.")

    if detected_items:
        df_nutrition = build_nutrition_table(detected_items)
        if df_nutrition is not None:
            st.markdown("---")
            show_nutrition_summary(df_nutrition)

elif mode == "Streaming Real-Time" and run_camera:
    st.markdown("---")
    
    col_cam, col_res = st.columns([1, 1])
    
    with col_cam:
        st.markdown("#### 📸 Live View WebRTC")
        
        def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            
            # Prediksi YOLO (imgsz=480 untuk performa lebih stabil di server/real-time)
            results = model.predict(source=img, conf=0.25, verbose=False, imgsz=480)
            plotted = results[0].plot()
            
            # Ekstrak data yang terdeteksi
            detected = []
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                cls_name = results[0].names[cls_id]
                conf = float(box.conf[0])
                detected.append({"nama": cls_name, "confidence": conf})
                
            try:
                # Masukkan ke antrean. Jika penuh, buang yang lama/abaikan agar UI tidak telat (backlog).
                webrtc_queue.put_nowait(detected)
            except queue.Full:
                pass
                
            return av.VideoFrame.from_ndarray(plotted, format="bgr24")

        webrtc_ctx = webrtc_streamer(
            key="yolo-webrtc",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
            video_frame_callback=video_frame_callback,
            media_stream_constraints={"video": {"facingMode": "environment"}, "audio": False},
            async_processing=True,
        )
        
    with col_res:
        st.markdown("#### 🍴 Hasil Deteksi & Nutrisi")
        INFO_WINDOW = st.empty()
        
    if webrtc_ctx.state.playing:
        prev_detected_str = None
        while True:
            try:
                # Ambil data terbaru dari antrean dengan timeout 0.5 detik
                detected_items = webrtc_queue.get(timeout=0.5)
                
                # Buat representasi string untuk deteksi perubahan (agar tabel tidak render tiap frame)
                current_detected_str = ",".join(sorted([item["nama"] for item in detected_items]))
                
                if current_detected_str != prev_detected_str:
                    prev_detected_str = current_detected_str
                    with INFO_WINDOW.container():
                        if detected_items:
                            tags_html = "".join([f'<span class="food-tag">{item["nama"]} ({item["confidence"]:.0%})</span>' for item in detected_items])
                            st.markdown(tags_html, unsafe_allow_html=True)
                            
                            df_nutrition = build_nutrition_table(detected_items)
                            if df_nutrition is not None:
                                st.markdown('<div class="nutrition-header" style="margin-top: 15px;">📊 Nutrisi per 100g</div>', unsafe_allow_html=True)
                                st.dataframe(df_nutrition, use_container_width=True, hide_index=True)
                        else:
                            st.info("Mencari makanan di depan kamera...")
            except queue.Empty:
                pass

# ── Footer ───────────────────────────────────────────────────
st.markdown("""
<div class="footer-text">
    Sistem Deteksi Makanan & Estimasi Nutrisi — Powered by YOLOv11 & ONNX Runtime Web<br>
    Data nutrisi berdasarkan TKPI 2017 (Tabel Komposisi Pangan Indonesia)
</div>
""", unsafe_allow_html=True)