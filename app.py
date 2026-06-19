import os
import tempfile
from pathlib import Path

import cv2

import pandas as pd
import streamlit as st
import torch
from ultralytics import YOLO

# Harus dipanggil sebelum perintah Streamlit lainnya
st.set_page_config(
    page_title="Sistem Deteksi & Estimasi Nutrisi pada Menu Makan Bergizi Gratis (MBG)",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Workaround bug Streamlit watcher dengan PyTorch
try:
    torch.classes.__path__ = []
except Exception:
    pass


# Konstanta 

# Angka Kecukupan Gizi (AKG) harian anak sekolah rata-rata berdasarkan standar Indonesia
DAILY_NUTRIENT_REFERENCE = {
    "kalori":  1800,
    "protein": 45,
    "karbo":   270,
    "lemak":   60,
    "serat":   25,
}

SUPPORTED_IMAGE_FORMATS = ["jpg", "jpeg", "png", "webp"]


# CSS 

CSS_PATH = Path(__file__).parent / "style.css"


@st.cache_data
def load_css() -> str:
    """Memuat CSS dari file eksternal."""
    return f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>"


# Inisialisasi Model & Data ─

import urllib.request

@st.cache_resource
def load_yolo_model() -> YOLO:
    model_path = "best_v6.pt"
    # URL download dari GitHub Release kamu
    model_url = "https://github.com/yogidwiyanto/MBG-Streamlit/releases/download/v1.0/best_v6.pt"
    
    if not os.path.exists(model_path):
        with st.spinner("Mendownload model AI (sekitar 49MB), mohon tunggu..."):
            try:
                urllib.request.urlretrieve(model_url, model_path)
            except Exception as e:
                st.error(f"Gagal mendownload model: {e}")
                st.stop()
                
    return YOLO(model_path)


@st.cache_data
def load_nutrition_database() -> pd.DataFrame:
    return pd.read_csv("nutrisi_tkpi2020_final.csv")


# ── Logika Deteksi ────────────────────────────────────────────────────────────

def extract_detected_classes(results) -> list[dict]:
    """Mengembalikan daftar nama kelas dan confidence dari hasil prediksi YOLO."""
    detected = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        cls_name = results[0].names[cls_id]
        confidence = float(box.conf[0])
        detected.append({"nama": cls_name, "confidence": confidence})
    return detected


def build_nutrition_table(detected_items: list[dict], db: pd.DataFrame) -> pd.DataFrame | None:
    """
    Membuat tabel nutrisi berdasarkan item yang terdeteksi.
    Menghitung jumlah kemunculan setiap kelas, lalu mencocokkan dengan database TKPI.
    """
    if not detected_items:
        return None

    # Hitung frekuensi kemunculan tiap kelas
    class_counts: dict[str, int] = {}
    for item in detected_items:
        name = item["nama"]
        class_counts[name] = class_counts.get(name, 0) + 1

    rows = []
    for cls_name, count in class_counts.items():
        match = db[db["label_model"] == cls_name]
        if match.empty:
            continue
        row = match.iloc[0]
        rows.append({
            "Makanan":         row["nama_pangan"],
            "Jumlah":          count,
            "Energi (kkal)":   row["energi_kkal"],
            "Protein (g)":     row["protein_g"],
            "Lemak (g)":       row["lemak_g"],
            "Karbohidrat (g)": row["karbohidrat_g"],
            "Serat (g)":       row["serat_g"],
        })

    return pd.DataFrame(rows) if rows else None


def calculate_totals(df: pd.DataFrame) -> dict[str, float]:
    """Menghitung total nutrisi dari tabel (mempertimbangkan jumlah porsi)."""
    return {
        "kalori":  (df["Energi (kkal)"]   * df["Jumlah"]).sum(),
        "protein": (df["Protein (g)"]     * df["Jumlah"]).sum(),
        "lemak":   (df["Lemak (g)"]       * df["Jumlah"]).sum(),
        "karbo":   (df["Karbohidrat (g)"] * df["Jumlah"]).sum(),
        "serat":   (df["Serat (g)"]       * df["Jumlah"]).sum(),
    }


# ── Logika Kelayakan Nutrisi ──────────────────────────────────────────────────

def evaluate_nutrient(
    key: str,
    value: float,
    ref: dict[str, float],
) -> dict:
    """
    Mengevaluasi satu nutrisi terhadap AKG dan mengembalikan item kelayakan.
    Thresholds berbeda per nutrisi sesuai konteks satu kali makan.
    """
    pct = (value / ref[key]) * 100

    if key == "kalori":
        if pct < 20:
            return {"level": "warning", "icon": "!", "text": f"<strong>Kalori rendah</strong> — Menyumbang {pct:.0f}% AKG ({value:.0f} kkal) dan berada di bawah rentang referensi yang digunakan (20–35% AKG)."}
        if pct <= 35:
            return {"level": "good",    "icon": "v", "text": f"<strong>Kalori memadai</strong> — Menyumbang {pct:.0f}% AKG ({value:.0f} kkal) berdasarkan nilai referensi yang digunakan."}
        return         {"level": "bad",     "icon": "!", "text": f"<strong>Kalori tinggi</strong> — Menyumbang {pct:.0f}% AKG ({value:.0f} kkal) dan berada di atas rentang referensi yang digunakan."}

    if key == "protein":
        if pct < 20:
            return {"level": "warning", "icon": "!", "text": f"<strong>Protein rendah</strong> ({value:.1f}g, {pct:.0f}% AKG). Kandungan protein masih berada di bawah nilai referensi yang digunakan."}
        if pct <= 40:
            return {"level": "good",    "icon": "v", "text": f"<strong>Protein memadai</strong> ({value:.1f}g, {pct:.0f}% AKG). Kontribusi protein cukup terhadap kebutuhan harian."}
        return         {"level": "good",    "icon": "v", "text": f"<strong>Protein sangat baik</strong> ({value:.1f}g, {pct:.0f}% AKG). Tinggi protein sangat baik untuk tumbuh kembang."}

    if key == "serat":
        if pct < 15:
            return {"level": "warning", "icon": "!", "text": f"<strong>Serat rendah</strong> ({value:.1f}g, {pct:.0f}% AKG). Kandungan serat masih rendah sehingga konsumsi sumber serat tambahan dapat dipertimbangkan."}
        if pct <= 40:
            return {"level": "good",    "icon": "v", "text": f"<strong>Serat memadai</strong> ({value:.1f}g, {pct:.0f}% AKG)."}
        return         {"level": "good",    "icon": "v", "text": f"<strong>Serat tinggi</strong> ({value:.1f}g, {pct:.0f}% AKG). Sangat baik untuk pencernaan."}

    if key == "lemak":
        if pct > 35:
            return {"level": "bad", "icon": "!", "text": f"<strong>Lemak tinggi</strong> ({value:.1f}g, {pct:.0f}% AKG). Kandungan lemak relatif tinggi dibandingkan nilai referensi yang digunakan."}
        return None

    return None


def build_adequacy_items(totals: dict[str, float]) -> list[dict]:
    """Mengembalikan daftar item kelayakan untuk semua nutrisi yang relevan."""
    nutrient_order = ["kalori", "protein", "serat", "lemak"]
    items = []
    for key in nutrient_order:
        item = evaluate_nutrient(key, totals[key], DAILY_NUTRIENT_REFERENCE)
        if item:
            items.append(item)
    return items


# ── Komponen UI ───────────────────────────────────────────────────────────────

def render_sidebar(db: pd.DataFrame) -> str:
    """Render sidebar navigation dan mengembalikan nama halaman yang dipilih."""
    if "active_page" not in st.session_state:
        st.session_state.active_page = "deteksi"

    with st.sidebar:
        st.markdown(
            '<div style="font-size:1.3rem;font-weight:800;color:#111827;'
            'margin-bottom:0.5rem;">📋 Menu Navigasi</div>',
            unsafe_allow_html=True,
        )
        st.markdown("---")

        if st.button("Deteksi Makanan", key="nav_deteksi", use_container_width=True,
                      type="primary" if st.session_state.active_page == "deteksi" else "secondary"):
            st.session_state.active_page = "deteksi"
            st.rerun()

        if st.button("Referensi AKG", key="nav_akg", use_container_width=True,
                      type="primary" if st.session_state.active_page == "akg" else "secondary"):
            st.session_state.active_page = "akg"
            st.rerun()

        if st.button("Data Nutrisi Makanan", key="nav_nutrisi", use_container_width=True,
                      type="primary" if st.session_state.active_page == "nutrisi" else "secondary"):
            st.session_state.active_page = "nutrisi"
            st.rerun()

        st.markdown("---")
        st.caption("Sistem Deteksi & Estimasi Nutrisi")

    return st.session_state.active_page


def render_page_akg():
    """Halaman Tabel Referensi AKG yang digunakan dalam sistem."""
    st.markdown("""
        <div class="main-heading" style="font-size:2.2rem;">
            📊 Tabel Referensi AKG
        </div>
        <div class="sub-heading">
            Angka Kecukupan Gizi (AKG) harian yang digunakan sebagai nilai referensi
            dalam sistem ini. Nilai berikut merupakan kebutuhan harian rata-rata
            anak usia sekolah berdasarkan data yang digunakan dalam penelitian.
        </div>
    """, unsafe_allow_html=True)

    # Tabel AKG Harian
    akg_data = pd.DataFrame([
        {"Zat Gizi": "Energi",      "Nilai AKG Harian": "1.800 kkal", "Satuan": "kkal", "Rentang Referensi per Satu Kali Makan (20–35% AKG)": "360 – 630 kkal"},
        {"Zat Gizi": "Protein",     "Nilai AKG Harian": "45 g",       "Satuan": "gram", "Rentang Referensi per Satu Kali Makan (20–35% AKG)": "9 – 15,75 g"},
        {"Zat Gizi": "Lemak",       "Nilai AKG Harian": "60 g",       "Satuan": "gram", "Rentang Referensi per Satu Kali Makan (20–35% AKG)": "12 – 21 g"},
        {"Zat Gizi": "Karbohidrat", "Nilai AKG Harian": "270 g",      "Satuan": "gram", "Rentang Referensi per Satu Kali Makan (20–35% AKG)": "54 – 94,5 g"},
        {"Zat Gizi": "Serat",       "Nilai AKG Harian": "25 g",       "Satuan": "gram", "Rentang Referensi per Satu Kali Makan (20–35% AKG)": "5 – 8,75 g"},
    ])

    st.markdown('<div class="section-title">Nilai AKG Harian</div>', unsafe_allow_html=True)
    st.dataframe(akg_data, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info(
        "**Catatan:** Nilai AKG yang digunakan merupakan kebutuhan harian rata-rata "
        "anak usia sekolah. Informasi yang ditampilkan bersifat indikatif sebagai "
        "bantuan interpretasi hasil estimasi nutrisi, bukan rekomendasi medis."
    )


def render_page_food_data(db: pd.DataFrame):
    """Halaman Tabel Class Makanan dan Data Nutrisi."""
    st.markdown("""
        <div class="main-heading" style="font-size:2.2rem;">
            🍽️ Data Nutrisi Makanan
        </div>
        <div class="sub-heading">
            Daftar seluruh kelas makanan yang dapat dideteksi oleh sistem beserta
            data kandungan nutrisi per 100 gram berdasarkan sumber referensi.
        </div>
    """, unsafe_allow_html=True)

    # Siapkan tabel tampilan
    display_df = db.copy()
    display_df = display_df.rename(columns={
        "label_model":     "Label Model",
        "nama_pangan":     "Nama Pangan",
        "energi_kkal":     "Energi (kkal)",
        "protein_g":       "Protein (g)",
        "lemak_g":         "Lemak (g)",
        "karbohidrat_g":   "Karbohidrat (g)",
        "serat_g":         "Serat (g)",
        "porsi_acuan":     "Porsi Acuan",
        "sumber":          "Sumber Data",
    })
    display_df.index = range(1, len(display_df) + 1)
    display_df.index.name = "No"

    st.markdown('<div class="section-title">Seluruh Kelas Makanan yang Dapat Dideteksi</div>', unsafe_allow_html=True)

    # Ringkasan statistik
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Kelas Makanan", f"{len(display_df)}")
    col2.metric("Sumber Data Utama", "TKPI 2020")
    col3.metric("Porsi Acuan", "100 gram")

    st.markdown("<br>", unsafe_allow_html=True)

    # Search / filter
    search = st.text_input("🔍 Cari makanan...", placeholder="Ketik nama makanan, misal: tempe, ayam, apel...")
    if search:
        mask = display_df["Nama Pangan"].str.contains(search, case=False, na=False) | \
               display_df["Label Model"].str.contains(search, case=False, na=False)
        filtered = display_df[mask]
    else:
        filtered = display_df

    st.dataframe(filtered, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info(
        f"**Total:** {len(filtered)} dari {len(display_df)} makanan ditampilkan. "
        "Data nutrisi bersumber dari TKPI 2020 (Tabel Komposisi Pangan Indonesia) "
        "dan USDA FoodData Central."
    )


def render_header():
    col_logo, col_status = st.columns([1, 1])
    with col_logo:
        st.markdown('<div class="logo">Sistem Deteksi & Estimasi Nutrisi</div>', unsafe_allow_html=True)
    with col_status:
        st.markdown(
            '<div style="display:flex;justify-content:flex-end;">'
            '<div class="status-badge">YOLOv11 Ready</div></div>',
            unsafe_allow_html=True,
        )


def render_hero():
    st.markdown("""
        <div class="main-heading">
            Deteksi & Estimasi Nutrisi<br>
            <span> Pada Menu Makan Bergizi Gratis (MBG)</span>
        </div>
        <div class="sub-heading">
            Deteksi otomatis jenis makanan dan estimasi kandungan nutrisi
            berbasis AI dengan referensi AKG harian.
        </div>
        <div class="feature-list">
            <div class="feature-item">Deteksi Instan</div>
            <div class="feature-item">30 Jenis Makanan</div>
            <div class="feature-item">Estimasi Nutrisi</div>
        </div>
    """, unsafe_allow_html=True)


def render_food_tags(detected_items: list[dict]):
    tags_html = "".join(
        f'<span class="food-tag">{item["nama"]}</span>'
        for item in detected_items
    )
    st.markdown(tags_html, unsafe_allow_html=True)


def render_nutrition_metrics(totals: dict[str, float]):
    st.markdown('<div class="section-title">Estimasi Total Nutrisi</div>', unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Energi",       f"{totals['kalori']:.0f} kkal")
    col2.metric("Protein",      f"{totals['protein']:.1f} g")
    col3.metric("Lemak",        f"{totals['lemak']:.1f} g")
    col4.metric("Karbohidrat",  f"{totals['karbo']:.1f} g")
    col5.metric("Serat",        f"{totals['serat']:.1f} g")

    st.info(
        "Catatan: Estimasi dihitung berdasarkan data TKPI 2020 "
        "dengan asumsi 1 porsi = 100 gram per makanan yang terdeteksi."
    )


def render_adequacy_card(items: list[dict]):
    items_html = "".join(
        f'<div class="adequacy-item {item["level"]}">'
        f'<div class="adequacy-text">{item["text"]}</div>'
        f'</div>'
        for item in items
    )
    st.markdown(
        f'<div class="adequacy-card">'
        f'<div class="adequacy-title">Kesimpulan Kelayakan Nutrisi</div>'
        f'{items_html}</div>',
        unsafe_allow_html=True,
    )


def render_detection_results(results, detected_items: list[dict], db: pd.DataFrame):
    """Menampilkan gambar hasil deteksi dan panel informasi nutrisi."""
    # Konversi gambar dari BGR ke RGB untuk Streamlit
    annotated_image = cv2.cvtColor(results[0].plot(conf=False), cv2.COLOR_BGR2RGB)

    col_image, col_info = st.columns([0.7, 1.3])

    with col_image:
        st.markdown("#### Hasil Deteksi")
        st.image(annotated_image, use_container_width=True)

    with col_info:
        st.markdown("#### Makanan Terdeteksi")

        if not detected_items:
            st.info("Tidak ada makanan yang terdeteksi pada gambar.")
            return

        render_food_tags(detected_items)
        st.markdown("<br>", unsafe_allow_html=True)

        df_nutrition = build_nutrition_table(detected_items, db)
        if df_nutrition is None:
            st.warning("Data nutrisi tidak ditemukan untuk makanan yang terdeteksi.")
            return

        st.markdown('<div class="section-title">Informasi Nutrisi per 100g</div>', unsafe_allow_html=True)
        st.dataframe(df_nutrition.drop(columns=["Jumlah"]), use_container_width=True, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)

        totals = calculate_totals(df_nutrition)
        render_nutrition_metrics(totals)

        adequacy_items = build_adequacy_items(totals)
        render_adequacy_card(adequacy_items)


def render_footer():
    st.markdown("""
        <div class="footer-text">
            Sistem Deteksi Makanan &amp; Estimasi Nutrisi — Powered by YOLOv11<br>
            Data nutrisi berdasarkan TKPI 2020 (Tabel Komposisi Pangan Indonesia)
        </div>
    """, unsafe_allow_html=True)


# ── Alur Utama Aplikasi ───────────────────────────────────────────────────────

def run_detection(uploaded_file, model: YOLO):
    """
    Menyimpan file sementara, menjalankan YOLO, lalu menghapus file setelah selesai.
    Mengembalikan hasil prediksi atau None jika gagal.
    """
    file_bytes = uploaded_file.read()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        image = cv2.imread(tmp_path)
        if image is None:
            st.error("Gambar tidak dapat dibaca. Pastikan file tidak rusak.")
            return None

        with st.spinner("Mendeteksi makanan..."):
            results = model.predict(source=tmp_path, conf=0.25)

        return results

    except Exception as error:
        st.error(f"Gagal mendeteksi makanan: {error}")
        return None

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def page_detection(model: YOLO, db: pd.DataFrame):
    """Halaman utama: Deteksi Makanan."""
    render_header()
    st.markdown("<br>", unsafe_allow_html=True)
    render_hero()
    st.markdown("<br>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload gambar makanan",
        type=SUPPORTED_IMAGE_FORMATS,
        label_visibility="collapsed",
    )
    st.caption("Format didukung: JPG, PNG, WEBP — Maks 200MB")

    if uploaded_file is None:
        render_footer()
        return

    st.markdown("---")

    results = run_detection(uploaded_file, model)
    if results is None:
        return

    detected_items = extract_detected_classes(results)
    render_detection_results(results, detected_items, db)

    render_footer()


def main():
    st.markdown(load_css(), unsafe_allow_html=True)
    model = load_yolo_model()
    db = load_nutrition_database()

    page = render_sidebar(db)

    if page == "deteksi":
        page_detection(model, db)
    elif page == "akg":
        render_page_akg()
    elif page == "nutrisi":
        render_page_food_data(db)


if __name__ == "__main__":
    main()