"""
The Slices: Apple or Orange? 🍎🍊
Aplikasi Streamlit untuk model klasifikasi Apel vs Jeruk (MobileNetV2, Keras .h5).

Catatan penting tentang preprocessing:
- Model sudah punya layer Rescaling sendiri (nama: "normalisasi", skala 1/127.5, offset -1).
- Karena itu aplikasi ini mengirim piksel MENTAH 0-255 (float32). JANGAN dinormalisasi lagi,
  kalau tidak, hasil akan selalu mendekati 50:50.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

# ---------------------------------------------------------------------------
# Konfigurasi dasar
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model_apel_jeruk.h5"
CONFIG_PATH = BASE_DIR / "kelas_apel_jeruk.json"

MAX_UPLOAD_MB = 10          # batas ukuran file unggahan
MIN_SIDE_PX = 32            # gambar lebih kecil dari ini dianggap tidak layak
AMBIGUOUS_MARGIN = 0.05     # |P(jeruk) - 0.5| < margin  -> dianggap "ragu-ragu" (sekitar 50:50)

st.set_page_config(
    page_title="The Slices: Apel atau Jeruk?",
    page_icon="🍊",
    layout="centered",
)

# ---------------------------------------------------------------------------
# CSS: tema pastel terkunci (tidak mengikuti mode gelap/terang sistem)
# ---------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fredoka:wght@400;500;600;700&display=swap');

:root {
  --orange-1: #FFD1A4;
  --orange-2: #FFAB76;
  --pink-1:   #FFB7B2;
  --pink-2:   #FF9AA2;
  --ink:      #5B3A3A;   /* teks gelap hangat: kontras baik di atas pastel */
  --ink-soft: #7A5555;
  --cream:    #FFF6EC;
  --card:     #FFFFFF;
}

html, body, .stApp {
  font-family: 'Fredoka', 'Segoe UI', system-ui, sans-serif !important;
  color-scheme: light !important;
}
.stApp {
  background: linear-gradient(160deg, #FFF1E2 0%, #FFE4DC 55%, #FFD9D6 100%) !important;
}
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"], #MainMenu, footer { visibility: hidden; }
.block-container { max-width: 760px !important; padding-top: 1.5rem !important; }

/* Teks: paksa warna gelap hangat di semua mode */
.stApp p, .stApp span, .stApp label, .stApp small, .stApp li,
.stApp h1, .stApp h2, .stApp h3, .stApp h4,
[data-testid="stMarkdownContainer"], [data-testid="stCaptionContainer"] {
  color: var(--ink) !important;
  font-family: 'Fredoka', 'Segoe UI', system-ui, sans-serif !important;
}

/* Buah melayang di latar belakang */
.floaty { position: fixed; z-index: 0; pointer-events: none; opacity: .55;
          animation: floaty 7s ease-in-out infinite; }
.floaty.f1 { top: 12%; left: 5%;  font-size: 2.2rem; animation-delay: 0s; }
.floaty.f2 { top: 28%; right: 6%; font-size: 1.8rem; animation-delay: 1.2s; }
.floaty.f3 { top: 62%; left: 4%;  font-size: 1.6rem; animation-delay: 2.4s; }
.floaty.f4 { top: 78%; right: 5%; font-size: 2.4rem; animation-delay: 3.1s; }
.floaty.f5 { top: 45%; left: 92%; font-size: 1.4rem; animation-delay: 4s; }
@keyframes floaty {
  0%, 100% { transform: translateY(0) rotate(-6deg); }
  50%      { transform: translateY(-18px) rotate(8deg); }
}
@keyframes bounce { 0%,100% { transform: translateY(-4%); } 50% { transform: none; } }
@keyframes pop    { 0% { transform: scale(.85); opacity: 0; } 100% { transform: scale(1); opacity: 1; } }
@keyframes shine  { 0% { background-position: 0% 50%; } 100% { background-position: 200% 50%; } }

/* Header */
.hero { text-align: center; margin: .5rem 0 1.2rem 0; position: relative; z-index: 1; }
.hero .badge {
  display: inline-flex; align-items: center; gap: .5rem;
  background: var(--card); border: 2px solid var(--orange-1); border-radius: 999px;
  padding: .45rem 1.1rem; box-shadow: 0 6px 18px rgba(255,171,118,.28);
  animation: bounce 3s ease-in-out infinite;
  font-weight: 700; letter-spacing: .08em; text-transform: uppercase; font-size: .78rem;
  color: #C2603A !important;
}
.hero h1 {
  font-size: 2.5rem; font-weight: 700; margin: .7rem 0 .2rem 0; line-height: 1.15;
  background: linear-gradient(90deg, #FF8E9A, #FFAB76, #FF8E9A);
  background-size: 200% auto; -webkit-background-clip: text; background-clip: text;
  -webkit-text-fill-color: transparent !important; color: transparent !important;
  animation: shine 5s linear infinite;
}
.hero p { color: var(--ink-soft) !important; font-size: 1rem; margin: 0; }

/* Kartu utama (st.container(border=True)) */
[data-testid="stVerticalBlockBorderWrapper"] {
  background: rgba(255,255,255,.88) !important;
  border: 2px solid var(--orange-1) !important;
  border-radius: 28px !important;
  box-shadow: 0 14px 38px rgba(255,154,162,.28) !important;
  position: relative; z-index: 1;
}

/* Maskot */
.mascot {
  display: flex; align-items: center; gap: 1rem;
  background: linear-gradient(135deg, #FFE9D3, #FFE0DD);
  border: 2px solid var(--orange-1); border-radius: 20px; padding: .9rem 1.2rem;
  animation: pop .35s ease-out;
}
.mascot .emo { font-size: 2.8rem; line-height: 1; animation: bounce 2.4s ease-in-out infinite; }
.mascot .ttl { font-weight: 700; font-size: 1.1rem; color: #9A4A2E !important; margin: 0; }
.mascot .sub { font-size: .88rem; color: var(--ink-soft) !important; margin: .15rem 0 0 0; }

/* Uploader: area drop */
[data-testid="stFileUploader"] section,
[data-testid="stFileUploaderDropzone"] {
  background: #FFF6EC !important;
  border: 2.5px dashed var(--orange-2) !important;
  border-radius: 22px !important;
  transition: transform .2s ease, box-shadow .2s ease, border-color .2s ease, background .2s ease;
}
[data-testid="stFileUploader"] section:hover,
[data-testid="stFileUploaderDropzone"]:hover {
  transform: translateY(-3px) scale(1.01);
  border-color: var(--pink-2) !important;
  background: #FFEFE0 !important;
  box-shadow: 0 10px 24px rgba(255,154,162,.35);
}
[data-testid="stFileUploader"] section *, [data-testid="stFileUploaderDropzone"] * {
  color: var(--ink) !important;
}
[data-testid="stFileUploader"] button {
  background: linear-gradient(90deg, var(--orange-2), var(--pink-2)) !important;
  color: #4A2323 !important; border: none !important; border-radius: 999px !important;
  font-weight: 600 !important;
}
[data-testid="stFileUploader"] button * { color: #4A2323 !important; }
[data-testid="stFileUploaderFile"] { background: #FFF6EC !important; border-radius: 14px !important; }
[data-testid="stFileUploaderFile"] * { color: var(--ink) !important; }

/* Gambar pratinjau */
[data-testid="stImage"] img {
  border-radius: 20px; box-shadow: 0 8px 22px rgba(91,58,58,.18); border: 3px solid #fff;
}

/* Tombol */
.stButton > button {
  width: 100%; border: none !important; border-radius: 999px !important;
  padding: .7rem 1rem !important; font-weight: 700 !important; font-size: 1rem !important;
  background: linear-gradient(90deg, var(--orange-2), var(--pink-2)) !important;
  box-shadow: 0 8px 18px rgba(255,154,162,.4) !important;
  transition: transform .15s ease, box-shadow .15s ease;
}
.stButton > button, .stButton > button * { color: #4A2323 !important; }
.stButton > button:hover:not(:disabled) {
  transform: translateY(-2px) scale(1.02); box-shadow: 0 12px 24px rgba(255,154,162,.5) !important;
}
.stButton > button:disabled { opacity: .45 !important; cursor: not-allowed; box-shadow: none !important; }
.stButton > button[kind="secondary"] { background: linear-gradient(90deg, var(--orange-1), var(--pink-1)) !important; }

/* Progress & spinner */
[data-testid="stProgress"] > div > div { background: #FFE0CC !important; border-radius: 999px; }
[data-testid="stProgress"] [role="progressbar"] > div,
[data-testid="stProgress"] div[data-baseweb="progress-bar"] div {
  background: linear-gradient(90deg, var(--orange-2), var(--pink-2)) !important;
}
[data-testid="stSpinner"] * { color: var(--ink) !important; }

/* Alert (error/warning) tetap terbaca */
[data-testid="stAlert"] { border-radius: 16px !important; background: #FFF0EE !important; border: 2px solid var(--pink-2) !important; }
[data-testid="stAlert"] * { color: #7A2E2E !important; }

/* Kartu hasil */
.result {
  border-radius: 24px; padding: 1.3rem 1.4rem; text-align: center; animation: pop .45s ease-out;
  border: 2.5px solid transparent; box-shadow: 0 12px 30px rgba(91,58,58,.14);
}
.result.apel  { background: linear-gradient(145deg, #FFD6D3, #FFB7B2); border-color: #FF9AA2; }
.result.jeruk { background: linear-gradient(145deg, #FFE3C7, #FFD1A4); border-color: #FFAB76; }
.result.ragu  { background: linear-gradient(145deg, #FFF3E0, #FFE9D6); border-color: #E7C9A9; }
.result .tag {
  display: inline-block; background: rgba(255,255,255,.8); border-radius: 999px;
  padding: .2rem .9rem; font-size: .75rem; font-weight: 700; letter-spacing: .08em;
  text-transform: uppercase; color: #8A3F2B !important;
}
.result .big   { font-size: 2.1rem; font-weight: 700; margin: .5rem 0 0 0; color: #4A2323 !important; }
.result .conf  { font-size: 1.05rem; margin: .1rem 0 .9rem 0; color: #5B3A3A !important; font-weight: 500; }
.bars { text-align: left; margin-top: .4rem; }
.bar-row { margin: .55rem 0; }
.bar-label { display: flex; justify-content: space-between; font-weight: 600; font-size: .92rem; color: #4A2323 !important; }
.bar-track { background: rgba(255,255,255,.75); border-radius: 999px; height: 16px; overflow: hidden; margin-top: .2rem; }
.bar-fill  { height: 100%; border-radius: 999px; }
.bar-fill.apel  { background: linear-gradient(90deg, #FF9AA2, #FFB7B2); }
.bar-fill.jeruk { background: linear-gradient(90deg, #FFAB76, #FFD1A4); }
.note { font-size: .8rem; color: var(--ink-soft) !important; text-align: center; margin-top: .6rem; }
.foot { text-align: center; font-size: .8rem; color: var(--ink-soft) !important; margin-top: 1rem; position: relative; z-index: 1; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def html(block: str) -> str:
    """Hilangkan indentasi tiap baris supaya Markdown tidak menganggapnya blok kode."""
    return "\n".join(line.strip() for line in block.strip().splitlines())


st.markdown(
    html(
        """
        <span class="floaty f1">🍎</span><span class="floaty f2">🍊</span>
        <span class="floaty f3">✨</span><span class="floaty f4">🍊</span>
        <span class="floaty f5">🍎</span>
        <div class="hero">
          <span class="badge"><span>🍎🍊</span><span>The Slices Classifier</span></span>
          <h1>Apel atau Jeruk? ✨</h1>
          <p>Unggah foto buah, lalu biarkan maskot AI kecil kami menebak: Apel atau Jeruk?</p>
        </div>
        """
    ),
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Memuat model (di-cache) -- error asli ditampilkan, tidak ditelan
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_assets(model_path: str, config_path: str):
    """Muat label dari JSON dan model dari .h5. Bentuk input dibaca dari model."""
    model_file, config_file = Path(model_path), Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"File konfigurasi tidak ditemukan: {config_file.name}")
    if not model_file.exists():
        raise FileNotFoundError(f"File model tidak ditemukan: {model_file.name}")

    config = json.loads(config_file.read_text(encoding="utf-8"))
    labels = config.get("labels")
    if not (
        isinstance(labels, list)
        and len(labels) == 2
        and all(isinstance(x, str) and x.strip() for x in labels)
    ):
        raise ValueError(
            'model_config.json harus berisi "labels": ["<kelas indeks 0>", "<kelas indeks 1>"], '
            'contoh: ["Apel", "Jeruk"].'
        )

    import keras  # impor di sini agar kegagalan impor ikut tertangkap & ditampilkan

    model = keras.models.load_model(str(model_file), compile=False)

    input_shape = model.input_shape
    output_shape = model.output_shape
    if not (isinstance(input_shape, tuple) and len(input_shape) == 4):
        raise ValueError(f"Bentuk input model tidak sesuai (diharapkan 4 dimensi): {input_shape}")
    _, height, width, channels = input_shape
    if not (isinstance(height, int) and isinstance(width, int) and channels == 3):
        raise ValueError(f"Input model harus berupa gambar RGB berukuran tetap, ditemukan: {input_shape}")
    if not (isinstance(output_shape, tuple) and len(output_shape) == 2 and output_shape[-1] == 1):
        raise ValueError(
            f"Aplikasi ini untuk klasifikasi biner dengan 1 unit sigmoid; output model: {output_shape}"
        )

    return model, labels, (height, width)


def predict_p_class1(model, batch: np.ndarray) -> float:
    """Kembalikan P(kelas indeks 1) dari output sigmoid. Validasi rentang 0..1."""
    raw = np.asarray(model(batch, training=False)).reshape(-1)
    if raw.size != 1 or not np.isfinite(raw[0]):
        raise ValueError(f"Output model tidak valid: {raw}")
    p = float(raw[0])
    if not (0.0 <= p <= 1.0):
        raise ValueError(
            f"Output model {p:.4f} di luar rentang 0-1, padahal diharapkan hasil sigmoid. "
            "Model yang dimuat mungkin bukan model yang sama dengan saat training."
        )
    return p


# ---------------------------------------------------------------------------
# Pemrosesan gambar
# ---------------------------------------------------------------------------
def load_image(data: bytes) -> Image.Image:
    """Buka gambar, perbaiki orientasi EXIF, ratakan transparansi ke latar putih, ubah ke RGB."""
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"File tidak bisa dibaca sebagai gambar ({exc}).") from exc

    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA", "P") and (img.mode != "P" or "transparency" in img.info):
        rgba = img.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, rgba)
    img = img.convert("RGB")

    if min(img.size) < MIN_SIDE_PX:
        raise ValueError(f"Gambar terlalu kecil ({img.size[0]}x{img.size[1]} px). Minimal {MIN_SIDE_PX} px per sisi.")
    return img


def to_model_input(img: Image.Image, size_hw: tuple[int, int]) -> np.ndarray:
    """Ubah ke tensor (1, H, W, 3) float32 berskala 0-255. Normalisasi dilakukan layer di dalam model."""
    height, width = size_hw
    resized = img.resize((width, height), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32)[np.newaxis, ...]


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "result" not in st.session_state:
    st.session_state.result = None

# Muat model
model = labels = input_hw = None
load_error: str | None = None
try:
    model, labels, input_hw = load_assets(str(MODEL_PATH), str(CONFIG_PATH))
except Exception as exc:  # ditampilkan apa adanya ke pengguna
    load_error = f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Antarmuka utama
# ---------------------------------------------------------------------------
with st.container(border=True):
    mascot_slot = st.empty()

    if load_error:
        st.error(
            "Model gagal dimuat, jadi fitur tebak dimatikan.\n\n"
            f"**Error asli:** `{load_error}`\n\n"
            "**Cek:** (1) `model_apel_jeruk.h5` dan `model_config.json` berada di folder yang sama dengan `app.py`; "
            "(2) versi `keras`/`tensorflow` di `requirements.txt` sama atau lebih baru dari versi saat training."
        )

    uploaded = st.file_uploader(
        "Seret & lepas foto buah di sini, atau klik untuk memilih 📸",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
        key=f"uploader_{st.session_state.uploader_key}",
        help=f"Format JPG, JPEG, atau PNG. Maksimal {MAX_UPLOAD_MB} MB.",
    )

    image: Image.Image | None = None
    file_key: str | None = None
    input_error: str | None = None
    data: bytes | None = None

    if uploaded is not None:
        data = uploaded.getvalue()
        if len(data) == 0:
            input_error = "File kosong (0 byte). Silakan pilih file lain."
        elif len(data) > MAX_UPLOAD_MB * 1024 * 1024:
            input_error = f"Ukuran file {len(data) / 1024 / 1024:.1f} MB melebihi batas {MAX_UPLOAD_MB} MB."
        else:
            try:
                image = load_image(data)
                file_key = hashlib.sha256(data).hexdigest()
            except ValueError as exc:
                input_error = str(exc)

    if input_error:
        st.error(input_error)
    if image is not None:
        st.image(image)

    col_predict, col_reset = st.columns([3, 2])
    with col_predict:
        predict_clicked = st.button(
            "🔍 Tebak Buahnya!",
            disabled=(image is None or model is None),
            key="btn_predict",
        )
    with col_reset:
        reset_clicked = st.button(
            "🔄 Ulangi",
            disabled=(uploaded is None and st.session_state.result is None),
            key="btn_reset",
            type="secondary",
        )

    # --- Reset: kosongkan uploader (dengan mengganti key) dan hasil ---
    if reset_clicked:
        st.session_state.uploader_key += 1
        st.session_state.result = None
        st.rerun()

    # --- Prediksi: selesai dulu, baru hasil dirender ---
    if predict_clicked and image is not None and model is not None:
        progress = st.progress(0, text="Menyiapkan gambar... 🍋")
        try:
            with st.spinner("Mengendus dan mengiris buah... 🔪"):
                batch = to_model_input(image, input_hw)
                progress.progress(40, text="Gambar siap, menanyakan ke maskot AI... 🤔")
                p_class1 = predict_p_class1(model, batch)
                progress.progress(100, text="Selesai! 🎉")
            st.session_state.result = {"file_key": file_key, "p_class1": p_class1}
        except Exception as exc:
            st.session_state.result = None
            st.error(f"Prediksi gagal — {type(exc).__name__}: {exc}")
        finally:
            progress.empty()

    # --- Ambil hasil hanya jika milik gambar yang sedang dipakai ---
    result = st.session_state.result
    if result is not None and (file_key is None or result["file_key"] != file_key):
        st.session_state.result = None
        result = None

    # --- Maskot menyesuaikan keadaan ---
    if load_error:
        emo, ttl, sub = "😿", "Oopsie, model belum siap", "Lihat pesan error di bawah ya."
    elif input_error:
        emo, ttl, sub = "😿", "Hmm, filenya bermasalah", "Coba unggah gambar lain."
    elif result is not None:
        p1 = result["p_class1"]
        if abs(p1 - 0.5) < AMBIGUOUS_MARGIN:
            emo, ttl, sub = "🤔", "Aku ragu-ragu nih...", "Hasilnya hampir 50:50."
        elif p1 >= 0.5:
            emo, ttl, sub = "🍊", f"Ini {labels[1]}! Segar!", "Lihat detail tebakannya di bawah."
        else:
            emo, ttl, sub = "🍎", f"Ini {labels[0]}! Yay!", "Lihat detail tebakannya di bawah."
    elif image is not None:
        emo, ttl, sub = "👀", "Wah, buah yang menarik!", "Tekan tombol tebak untuk membuka rahasianya."
    else:
        emo, ttl, sub = "🧐", "Siap mengiris?", "Unggah gambar buah untuk mulai."
    mascot_slot.markdown(
        html(
            f"""
            <div class="mascot">
              <div class="emo">{emo}</div>
              <div><p class="ttl">{ttl}</p><p class="sub">{sub}</p></div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )

    # --- Kartu hasil ---
    if result is not None and labels is not None:
        p1 = result["p_class1"]
        p0 = 1.0 - p1
        names = {0: labels[0], 1: labels[1]}
        css_cls = {0: "apel", 1: "jeruk"}  # urutan sesuai model_config.json (0=Apel, 1=Jeruk)

        if abs(p1 - 0.5) < AMBIGUOUS_MARGIN:
            card_cls, tag, big = "ragu", "🤔 Belum yakin", "Hampir 50:50"
            conf_line = f"{names[0]} {p0 * 100:.1f}% vs {names[1]} {p1 * 100:.1f}%. Coba foto lain yang lebih jelas."
        else:
            win = 1 if p1 >= 0.5 else 0
            win_p = p1 if win == 1 else p0
            emoji = "🍊" if win == 1 else "🍎"
            card_cls, tag, big = css_cls[win], f"{emoji} Tebakan AI", f"{names[win]}"
            conf_line = f"Tingkat keyakinan: {win_p * 100:.1f}%"

        st.markdown(
            html(
                f"""
                <div class="result {card_cls}">
                  <span class="tag">{tag}</span>
                  <p class="big">{big}</p>
                  <p class="conf">{conf_line}</p>
                  <div class="bars">
                    <div class="bar-row">
                      <div class="bar-label"><span>🍎 {names[0]}</span><span>{p0 * 100:.1f}%</span></div>
                      <div class="bar-track"><div class="bar-fill apel" style="width:{p0 * 100:.1f}%"></div></div>
                    </div>
                    <div class="bar-row">
                      <div class="bar-label"><span>🍊 {names[1]}</span><span>{p1 * 100:.1f}%</span></div>
                      <div class="bar-track"><div class="bar-fill jeruk" style="width:{p1 * 100:.1f}%"></div></div>
                    </div>
                  </div>
                  <p class="note">Model hanya mengenal {names[0]} dan {names[1]}. Gambar selain keduanya tetap akan dipaksa masuk ke salah satunya.</p>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

st.markdown('<div class="foot">Dibuat dengan 🍎, 🍊, dan sedikit sihir pastel ✨</div>', unsafe_allow_html=True)
