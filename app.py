from __future__ import annotations

import hashlib
import html
import json
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
from PIL import Image, ImageOps

# -----------------------------------------------------------------------------
# File configuration — place these files in the same folder as app.py.
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model_apel_jeruk.h5"
CONFIG_PATH = BASE_DIR / "kelas_apel_jeruk.json"
BACKSOUND_PATH = BASE_DIR / "backsound.mp3"  # Optional; can also upload audio in sidebar.
IMAGE_SIZE = (224, 224)

st.set_page_config(
    page_title="The Slices: Apple or Orange?",
    page_icon="🍎",
    layout="centered",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Fixed pastel theme. color-scheme: light prevents OS/browser dark mode from
# recoloring native controls. The !important rules deliberately override themes.
# -----------------------------------------------------------------------------
CUSTOM_CSS = r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fredoka:wght@400;500;600;700&family=Inter:wght@400;500;600;700&display=swap');

:root {
  color-scheme: light !important;
  --cream: #FFF8F2;
  --orange-soft: #FFD1A4;
  --orange-strong: #FFAB76;
  --pink-soft: #FFB7B2;
  --red-soft: #FF9AA2;
  --ink: #513A35;
  --muted: #806B65;
  --card: #FFFEFC;
  --line: #F3DCCB;
}
html, body, #root, .stApp,
[data-testid="stAppViewContainer"], [data-testid="stMain"],
[data-testid="stMainBlockContainer"], [data-testid="stHeader"],
[data-testid="stBottom"] {
  background: var(--cream) !important;
  color: var(--ink) !important;
  color-scheme: light !important;
}
html, body, .stApp, [data-testid="stAppViewContainer"] {
  font-family: 'Fredoka', 'Inter', sans-serif !important;
}
[data-testid="stHeader"] { background: rgba(255,248,242,.96) !important; }
[data-testid="stToolbar"] { background: transparent !important; }
[data-testid="stSidebar"] {
  background: #FFF0E5 !important;
  border-right: 1px solid #F2D6C2 !important;
}
[data-testid="stSidebar"] > div:first-child,
[data-testid="stSidebarContent"] {
  background: #FFF0E5 !important;
}
[data-testid="stSidebar"] * { color: var(--ink) !important; }
[data-testid="stMainBlockContainer"] { max-width: 850px !important; padding-top: 1.8rem !important; }
[data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] span, label, p, h1, h2, h3, h4 {
  color: var(--ink) !important;
}
small, .stCaption, [data-testid="stCaptionContainer"] { color: var(--muted) !important; }

/* Hero */
.hero-wrap { text-align:center; margin: .15rem auto 1.5rem; max-width: 680px; }
.hero-badge {
  display:inline-flex; align-items:center; gap:.6rem; padding:.55rem 1.05rem;
  border:1px solid #F2D5BF; border-radius:999px; background:#FFFEFC;
  box-shadow:0 7px 20px rgba(161,105,72,.09); color:#B7663E !important;
  font-size:.76rem; font-weight:700; letter-spacing:.11em; text-transform:uppercase;
  animation: bob 3.2s ease-in-out infinite;
}
.hero-badge .fruit { font-size:1.45rem; letter-spacing:0; }
.hero-title {
  margin:.75rem 0 .35rem; font-size:clamp(2.2rem, 6vw, 3.1rem); line-height:1.12;
  font-weight:700; letter-spacing:-.035em;
  background:linear-gradient(100deg,#DF6B70 5%,#F3A45D 52%,#E98149 95%);
  -webkit-background-clip:text; background-clip:text; color:transparent !important;
}
.hero-subtitle { color:#806B65 !important; font-size:1rem; line-height:1.65; max-width:570px; margin:.4rem auto 0; }
.hero-sparkles { color:#E9A45F !important; letter-spacing:.55rem; font-size:.95rem; margin-top:.65rem; }

/* Main card and mascot */
section.main [data-testid="stVerticalBlockBorderWrapper"] {
  background:rgba(255,254,252,.94) !important; border:1px solid #F3DDCB !important;
  border-radius:30px !important; box-shadow:0 18px 48px rgba(160,102,72,.10) !important;
  padding:1.1rem 1.15rem !important;
}
.mascot-card {
  display:flex; align-items:center; gap:1rem; padding:1rem 1.1rem; width:100%;
  box-sizing:border-box; background:linear-gradient(110deg,rgba(255,209,164,.30),rgba(255,183,178,.20));
  border:1px solid #F3D9C8; border-radius:21px; margin:.2rem 0 1rem;
}
.mascot-emoji { font-size:2.8rem; line-height:1; filter: drop-shadow(0 5px 4px rgba(160,102,72,.12)); }
.mascot-title { margin:0 0 .22rem; font-weight:700; font-size:1.12rem; color:#955632 !important; }
.mascot-subtitle { margin:0; font-size:.88rem; line-height:1.45; color:#806B65 !important; }
.upload-guide {
  padding:.85rem 1rem; margin:.2rem 0 .75rem; border:2px dashed #F0BC96; border-radius:20px;
  background:#FFF7EF; text-align:center;
}
.upload-guide .camera { font-size:2.05rem; margin-bottom:.25rem; }
.upload-guide strong { display:block; font-size:1rem; color:#745044 !important; }
.upload-guide p { margin:.2rem 0 0; font-size:.8rem; color:#92776C !important; }

/* File upload zone */
[data-testid="stFileUploader"] { width:100%; }
[data-testid="stFileUploaderDropzone"] {
  background:#FFF7EF !important; border:2px dashed #EFB98F !important;
  border-radius:20px !important; min-height:100px !important;
}
[data-testid="stFileUploaderDropzone"] * { color:#745044 !important; }
[data-testid="stFileUploaderDropzone"] button {
  background:#FFE1C7 !important; border:1px solid #F2C7A6 !important; color:#744A36 !important;
  border-radius:12px !important;
}
[data-testid="stFileUploaderFile"] { background:#FFFEFC !important; border:1px solid #F1D7C5 !important; border-radius:12px !important; }
[data-testid="stImage"] img { border-radius:18px !important; }

/* Inputs and buttons */
.stButton > button, [data-testid="stDownloadButton"] button {
  border:0 !important; border-radius:16px !important; min-height:3.05rem !important;
  font-family:'Fredoka','Inter',sans-serif !important; font-weight:700 !important;
  transition:transform .18s ease, box-shadow .18s ease, filter .18s ease !important;
}
section.main .stButton > button[kind="primary"], section.main .stButton > button {
  background:linear-gradient(105deg,#FFAB76,#FF966A) !important; color:#FFFFFF !important;
  box-shadow:0 9px 20px rgba(237,137,87,.23) !important;
}
section.main .stButton > button:hover { transform:translateY(-2px); filter:saturate(1.05); box-shadow:0 12px 24px rgba(237,137,87,.27) !important; }
section.main .stButton > button:focus { outline:3px solid #FFB7B2 !important; outline-offset:2px !important; }
section.main .stButton > button:disabled { background:#E9DAD0 !important; color:#9B8880 !important; box-shadow:none !important; }
[data-testid="stSidebar"] .stButton > button {
  background:#FFFEFC !important; color:#8D583F !important; border:1px solid #F0CEB6 !important;
  box-shadow:none !important; border-radius:14px !important;
}
[data-testid="stProgressBar"] > div { background:#F7E8DC !important; border-radius:99px !important; }
[data-testid="stProgressBar"] > div > div { background:linear-gradient(90deg,#FFAB76,#FF9AA2) !important; border-radius:99px !important; }
[data-testid="stAlert"] { background:#FFF4E9 !important; border:1px solid #F2D3B9 !important; border-radius:16px !important; }
[data-testid="stAlert"] * { color:#6F4B3C !important; }
[data-testid="stAudio"] { background:#FFFEFC !important; border-radius:14px !important; }

/* Prediction cards */
.result-card {
  box-sizing:border-box; width:100%; margin:.85rem 0 .2rem; padding:1.35rem 1.2rem;
  border:1px solid #F0D9C8; border-radius:23px; text-align:center;
  background:linear-gradient(145deg,#FFF8F0,#FFFEFC); box-shadow:0 9px 25px rgba(160,102,72,.08);
}
.result-card.apple { background:linear-gradient(145deg,#FFF0EF,#FFFEFC); border-color:#F5C8C5; }
.result-card.orange { background:linear-gradient(145deg,#FFF2E3,#FFFEFC); border-color:#F5D0AE; }
.result-fruit { font-size:3.2rem; line-height:1.15; }
.result-pill {
  display:inline-block; padding:.35rem .75rem; border-radius:999px; margin:.5rem 0 .3rem;
  font-size:.73rem; font-weight:700; text-transform:uppercase; letter-spacing:.07em;
  color:#8C4D3C !important; background:#FFE1D5;
}
.apple .result-pill { color:#AC4B55 !important; background:#FFE0E2; }
.orange .result-pill { color:#A65C2F !important; background:#FFE2C2; }
.result-title { font-size:1.65rem; line-height:1.25; font-weight:700; margin:.2rem 0 .3rem; }
.result-desc { margin:0 auto .8rem; max-width:460px; color:#806B65 !important; font-size:.9rem; line-height:1.55; }
.confidence-line { display:flex; justify-content:space-between; gap:1rem; align-items:center; font-size:.86rem; font-weight:600; color:#76594E !important; margin:.9rem auto .4rem; max-width:470px; }
.confidence-track { max-width:470px; margin:0 auto; height:11px; overflow:hidden; background:#F4E5DA; border-radius:99px; }
.confidence-fill { height:100%; border-radius:99px; background:linear-gradient(90deg,#FFAB76,#FF9AA2); }
.alt-result { margin:.8rem 0 0; color:#92796F !important; font-size:.78rem; }
.sidebar-title { font-size:1.45rem; font-weight:700; margin:.15rem 0 .2rem; color:#9A5A3D !important; }
.sidebar-note { font-size:.84rem; line-height:1.55; color:#806B65 !important; }
.sidebar-chip { display:inline-block; padding:.3rem .65rem; background:#FFFEFC; border:1px solid #F0D5C0; border-radius:999px; color:#9A5A3D !important; font-size:.75rem; font-weight:700; }
.footer-note { text-align:center; color:#9B8175 !important; font-size:.78rem; padding:1rem 0 .35rem; line-height:1.6; }
hr { border-color:#F1DCCB !important; }
@keyframes bob { 0%,100% { transform:translateY(-3px); } 50% { transform:translateY(3px); } }
@media (max-width:600px) {
  [data-testid="stMainBlockContainer"] { padding-left:1rem !important; padding-right:1rem !important; }
  section.main [data-testid="stVerticalBlockBorderWrapper"] { padding:.85rem !important; border-radius:24px !important; }
  .mascot-emoji { font-size:2.25rem; }
  .mascot-card { gap:.75rem; padding:.9rem; }
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Model loading. Supported JSON forms:
# 1) Keras model architecture JSON directly (from model.to_json()) + H5 weights.
# 2) Metadata JSON with `model_json` / `architecture_json` and class labels.
# 3) Metadata JSON with labels while the H5 contains a complete Keras model.
# -----------------------------------------------------------------------------
def _extract_labels(config: dict[str, Any]) -> list[str]:
    labels: Any = None
    for key in ("class_names", "labels", "class_labels", "classes", "class_indices"):
        if key in config:
            labels = config[key]
            break

    if isinstance(labels, dict):
        # Support either {"0": "Apel", "1": "Jeruk"} or {"Apel": 0, "Jeruk": 1}.
        if labels and all(str(k).lstrip("-").isdigit() for k in labels):
            labels = [labels[k] for k in sorted(labels, key=lambda x: int(x))]
        elif labels and all(isinstance(v, (int, float)) for v in labels.values()):
            labels = [k for k, _ in sorted(labels.items(), key=lambda pair: pair[1])]
        else:
            labels = list(labels.values())

    if isinstance(labels, str):
        labels = [item.strip() for item in labels.split(",") if item.strip()]

    if isinstance(labels, (tuple, list)) and labels:
        return [str(label) for label in labels]

    # Default order is only a fallback. It MUST match the model's training order.
    return ["Apel", "Jeruk"]


def _find_architecture_json(config: dict[str, Any], raw_text: str) -> str | None:
    for key in ("model_json", "architecture_json", "model_architecture", "architecture"):
        value = config.get(key)
        if isinstance(value, dict):
            return json.dumps(value)
        if isinstance(value, str) and value.strip():
            candidate = value.strip()
            # A value may contain inline JSON or point to a separate architecture JSON file.
            if candidate.startswith("{") or candidate.startswith("["):
                return candidate
            possible_path = (CONFIG_PATH.parent / candidate).resolve()
            if possible_path.is_file():
                return possible_path.read_text(encoding="utf-8")

    # A raw model.to_json() file has the standard Keras `class_name` + `config` structure.
    if "class_name" in config and "config" in config:
        return raw_text
    return None


@st.cache_resource(show_spinner=False)
def load_model_bundle(
    model_path: str,
    config_path: str,
    model_mtime_ns: int,
    config_mtime_ns: int,
) -> dict[str, Any]:
    """Load/cache model and label metadata; mtimes invalidate cache after file changes."""
    from tensorflow.keras.models import load_model, model_from_json

    model_file = Path(model_path)
    config_file = Path(config_path)
    if not model_file.is_file():
        raise FileNotFoundError(
            f"File model tidak ditemukan: {model_file.name}. Letakkan file tersebut di folder yang sama dengan app.py."
        )
    if not config_file.is_file():
        raise FileNotFoundError(
            f"File konfigurasi tidak ditemukan: {config_file.name}. Letakkan file tersebut di folder yang sama dengan app.py."
        )

    raw_text = config_file.read_text(encoding="utf-8")
    try:
        config = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Isi {config_file.name} bukan JSON yang valid: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError("Root kelas_apel_jeruk.json harus berupa objek JSON.")

    architecture_json = _find_architecture_json(config, raw_text)
    if architecture_json:
        model = model_from_json(architecture_json)
        model.load_weights(str(model_file))
    else:
        # Also supports H5 files that store the full Keras model, not only weights.
        try:
            model = load_model(str(model_file), compile=False)
        except Exception as exc:
            raise ValueError(
                "Model .h5 tidak dapat dibuka sebagai model lengkap dan JSON tidak menyediakan arsitektur. "
                "Untuk file bobot saja, ekspor arsitektur dengan model.to_json() dan simpan ke kelas_apel_jeruk.json."
            ) from exc

    labels = _extract_labels(config)
    preprocessing = str(config.get("preprocessing", config.get("preprocess", "mobilenet_v2"))).lower().strip()
    if isinstance(config.get("preprocessing"), dict):
        preprocessing = str(config["preprocessing"].get("name", "mobilenet_v2")).lower().strip()
    output_type = str(config.get("output_type", "auto")).lower().strip()

    return {
        "model": model,
        "labels": labels,
        "preprocessing": preprocessing,
        "output_type": output_type,
    }


def preprocess_image(image_bytes: bytes, preprocessing: str) -> np.ndarray:
    """Decode, orient, resize to 224×224 RGB, then normalize for model inference."""
    try:
        with Image.open(BytesIO(image_bytes)) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            image = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
            pixels = np.asarray(image, dtype=np.float32)
    except Exception as exc:
        raise ValueError("Gambar tidak dapat dibaca. Unggah file JPG, JPEG, atau PNG yang valid.") from exc

    mode = preprocessing.replace("-", "_").replace(" ", "_")
    if mode in {"mobilenet_v2", "mobilenetv2", "minus1_1", "minus_one_to_one", "_1_1"}:
        # MobileNetV2 preprocess_input maps pixel values from [0, 255] to [-1, 1].
        pixels = (pixels / 127.5) - 1.0
    elif mode in {"rescale_0_1", "zero_to_one", "1_255", "divide_255"}:
        pixels = pixels / 255.0
    elif mode in {"none", "raw", "0_255"}:
        pass
    else:
        raise ValueError(
            f"Mode preprocessing '{preprocessing}' tidak dikenal. Pilih mobilenet_v2, rescale_0_1, atau none di JSON."
        )

    return np.expand_dims(pixels, axis=0).astype(np.float32)


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exps = np.exp(shifted)
    return exps / np.sum(exps)


def prediction_scores(raw_prediction: Any, label_count: int, output_type: str = "auto") -> np.ndarray:
    """Convert common softmax, sigmoid, or logit outputs to comparable scores."""
    if isinstance(raw_prediction, dict):
        raw_prediction = next(iter(raw_prediction.values()))
    if isinstance(raw_prediction, (list, tuple)):
        raw_prediction = raw_prediction[0]

    values = np.asarray(raw_prediction, dtype=np.float64).squeeze()
    if values.ndim == 0:
        values = np.asarray([float(values)])
    values = values.reshape(-1)
    if not np.all(np.isfinite(values)):
        raise ValueError("Model menghasilkan skor non-finite (NaN/Inf). Periksa model dan preprocessing.")

    kind = output_type.lower()
    if values.size == 1:
        if label_count != 2:
            raise ValueError("Output model hanya memiliki 1 nilai, tetapi konfigurasi label bukan dua kelas.")
        value = float(values[0])
        if kind == "logits" or value < 0.0 or value > 1.0:
            value = 1.0 / (1.0 + np.exp(-np.clip(value, -60, 60)))
        else:
            value = float(np.clip(value, 0.0, 1.0))
        return np.asarray([1.0 - value, value], dtype=np.float64)

    if values.size != label_count:
        raise ValueError(
            f"Output model memiliki {values.size} kelas, sedangkan kelas_apel_jeruk.json mendefinisikan {label_count} label."
        )

    if kind == "logits" or np.any(values < 0.0) or np.any(values > 1.0):
        return _softmax(values)

    total = float(values.sum())
    if total <= 0:
        return _softmax(values)
    # Softmax outputs sum to ~1. Independent sigmoid outputs are normalized for display.
    return values / total


def display_class(label: str) -> tuple[str, str, str, str]:
    """Return (localized label, emoji, card class, friendly result text)."""
    normalized = label.casefold().strip().replace("_", " ").replace("-", " ")
    if "apple" in normalized or "apel" in normalized:
        return "Apel", "🍎", "apple", "Manis, renyah, dan siap jadi bintang hari ini!"
    if "orange" in normalized or "jeruk" in normalized:
        return "Jeruk", "🍊", "orange", "Segar, cerah, dan penuh citrus energy!"
    return label, "🍋", "", "Kelas ini mengikuti label yang tersedia pada konfigurasi model."


def get_file_mtime(path: Path) -> int:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return 0


# -----------------------------------------------------------------------------
# Session state / sidebar
# -----------------------------------------------------------------------------
for key, default in {
    "uploader_version": 0,
    "prediction": None,
    "image_digest": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center;padding:.3rem 0 1rem;">
          <div style="font-size:2.6rem;">🍎🍊</div>
          <div class="sidebar-title">The Slices</div>
          <div class="sidebar-note">Tiny fruit lab, big cute energy ✨</div>
          <span class="sidebar-chip">AI FRUIT CLASSIFIER</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown("### 🎵 Backsound corner")
    st.caption("Putar musik lembut saat mencoba klasifikasi.")
    audio_upload = st.file_uploader(
        "Unggah audio (opsional)",
        type=["mp3", "wav", "ogg", "m4a"],
        key="backsound_uploader",
        help="Audio hanya diputar setelah tombol play ditekan; browser tidak selalu mengizinkan autoplay.",
    )
    if audio_upload is not None:
        st.audio(audio_upload.getvalue(), format=audio_upload.type or "audio/mpeg")
    elif BACKSOUND_PATH.is_file():
        st.audio(str(BACKSOUND_PATH))
    else:
        st.info("Belum ada backsound. Unggah file audio di atas atau letakkan `backsound.mp3` di folder aplikasi.")

    st.markdown("---")
    st.markdown("### 🧠 About the tiny brain")
    st.markdown(
        """
        <div class="sidebar-note">
          <b>Architecture:</b> MobileNetV2<br>
          <b>Input:</b> RGB · 224 × 224 px<br>
          <b>Output:</b> class label + confidence<br>
          <b>Runtime:</b> TensorFlow / Keras
        </div>
        """,
        unsafe_allow_html=True,
    )
    if MODEL_PATH.is_file() and CONFIG_PATH.is_file():
        st.success("File model dan konfigurasi ditemukan.")
    else:
        missing = []
        if not MODEL_PATH.is_file():
            missing.append(MODEL_PATH.name)
        if not CONFIG_PATH.is_file():
            missing.append(CONFIG_PATH.name)
        st.warning("Belum lengkap: " + ", ".join(missing))

    st.markdown("---")
    st.markdown("### 🍊 How to use")
    st.markdown(
        """
        <div class="sidebar-note">
          <b>01.</b> Unggah foto apel atau jeruk.<br><br>
          <b>02.</b> Periksa pratinjau gambar.<br><br>
          <b>03.</b> Tekan <i>Detect Fruit!</i>.<br><br>
          <b>04.</b> Lihat kelas dan confidence score.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Tip: gunakan foto yang jelas, terang, dan menampilkan satu buah utama.")


# -----------------------------------------------------------------------------
# Main area — intentionally modeled after the supplied HTML reference.
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-wrap">
      <div class="hero-badge"><span class="fruit">🍎🍊</span><span>The Slices Classifier</span></div>
      <h1 class="hero-title">Apple or Orange? ✨</h1>
      <p class="hero-subtitle">Upload a fruit photo and let our tiny AI mascot do a little vibe check. Cute fruit science starts here!</p>
      <div class="hero-sparkles">✦ · ✧ · ✦</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.container(border=True):
    current_upload_key = f"fruit_uploader_{st.session_state.uploader_version}"
    uploaded_file = st.file_uploader(
        "Choose a fruit image",
        type=["jpg", "jpeg", "png"],
        key=current_upload_key,
        help="Format yang didukung: JPG, JPEG, PNG. Gambar diproses lokal oleh aplikasi ini.",
    )

    image_bytes: bytes | None = None
    digest: str | None = None
    if uploaded_file is not None:
        image_bytes = uploaded_file.getvalue()
        digest = hashlib.sha256(image_bytes).hexdigest()
        if st.session_state.image_digest != digest:
            st.session_state.image_digest = digest
            st.session_state.prediction = None

        st.markdown(
            """
            <div class="mascot-card">
              <div class="mascot-emoji">👀</div>
              <div>
                <p class="mascot-title">Ooh, look at that specimen!</p>
                <p class="mascot-subtitle">Your fruit is in the lab. Tap the button when you're ready for its identity reveal.</p>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        try:
            with Image.open(BytesIO(image_bytes)) as preview_image:
                st.image(
                    ImageOps.exif_transpose(preview_image).convert("RGB"),
                    caption=f"Preview · {html.escape(uploaded_file.name)}",
                    use_container_width=True,
                )
        except Exception:
            st.error("File ini tidak dapat ditampilkan sebagai gambar. Coba unggah JPG, JPEG, atau PNG yang valid.")

        st.markdown(
            '<p style="text-align:center;color:#8A746A!important;font-size:.82rem;">Looks cute already. Let’s see what the model thinks!</p>',
            unsafe_allow_html=True,
        )
        detect_col, reset_col = st.columns([3, 1], gap="small")
        with detect_col:
            detect_clicked = st.button("🔍 Detect Fruit!", type="primary", use_container_width=True)
        with reset_col:
            reset_clicked = st.button("🔄 Reset", use_container_width=True)

        if reset_clicked:
            st.session_state.uploader_version += 1
            st.session_state.prediction = None
            st.session_state.image_digest = None
            st.rerun()

        if detect_clicked:
            progress = st.progress(5, text="Warming up the tiny fruit lab…")
            try:
                progress.progress(15, text="Loading the model and label map…")
                bundle = load_model_bundle(
                    str(MODEL_PATH),
                    str(CONFIG_PATH),
                    get_file_mtime(MODEL_PATH),
                    get_file_mtime(CONFIG_PATH),
                )
                progress.progress(52, text="Resizing to 224 × 224 and normalizing pixels…")
                batch = preprocess_image(image_bytes, bundle["preprocessing"])
                progress.progress(75, text="Asking the model for its best guess…")
                raw = bundle["model"].predict(batch, verbose=0)
                scores = prediction_scores(raw, len(bundle["labels"]), bundle["output_type"])
                winner_index = int(np.argmax(scores))
                winner_label = bundle["labels"][winner_index]
                display_name, fruit_emoji, theme_class, friendly_text = display_class(winner_label)
                confidence = float(scores[winner_index])
                ranked = np.argsort(scores)[::-1]
                second_text = ""
                if len(ranked) > 1:
                    second_name, _, _, _ = display_class(bundle["labels"][int(ranked[1])])
                    second_score = float(scores[int(ranked[1])]) * 100
                    second_text = f"Alternatif berikutnya: {second_name} · {second_score:.1f}%"
                st.session_state.prediction = {
                    "digest": digest,
                    "label": display_name,
                    "emoji": fruit_emoji,
                    "theme": theme_class,
                    "description": friendly_text,
                    "confidence": confidence,
                    "second_text": second_text,
                }
                progress.progress(100, text="Done! Fruit identity unlocked ✨")
            except Exception as exc:
                st.session_state.prediction = None
                st.error(f"Prediksi gagal: {exc}")
            finally:
                progress.empty()

        prediction = st.session_state.prediction
        if prediction and prediction.get("digest") == digest:
            conf_percent = max(0.0, min(100.0, float(prediction["confidence"]) * 100.0))
            theme_class = prediction["theme"]
            display_name = html.escape(str(prediction["label"]))
            fruit_emoji = html.escape(str(prediction["emoji"]))
            title = "It's an Apple! Yay!" if prediction["label"] == "Apel" else (
                "It's an Orange! Zesty!" if prediction["label"] == "Jeruk" else "A mysterious fruit!"
            )
            title = html.escape(title)
            subtitle = html.escape(str(prediction["description"]))
            second_text = html.escape(str(prediction["second_text"]))
            pill = "🍎 Apple confirmed" if prediction["label"] == "Apel" else (
                "🍊 Orange confirmed" if prediction["label"] == "Jeruk" else "✨ Model class"
            )
            st.markdown(
                f"""
                <div class="result-card {theme_class}">
                  <div class="result-fruit">{fruit_emoji}</div>
                  <div class="result-pill">{pill}</div>
                  <div class="result-title">{title}</div>
                  <p class="result-desc">{subtitle}</p>
                  <div class="confidence-line"><span>Model confidence</span><strong>{conf_percent:.1f}%</strong></div>
                  <div class="confidence-track"><div class="confidence-fill" style="width:{conf_percent:.1f}%"></div></div>
                  <p class="alt-result">{second_text}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption("Confidence adalah skor keluaran model, bukan jaminan bahwa prediksi selalu benar.")
    else:
        st.markdown(
            """
            <div class="mascot-card">
              <div class="mascot-emoji">🧐</div>
              <div>
                <p class="mascot-title">Ready for a slice?</p>
                <p class="mascot-subtitle">Drop an image below to start the vibe check! Our tiny mascot is waiting.</p>
              </div>
            </div>
            <div class="upload-guide">
              <div class="camera">📸</div>
              <strong>Click above to upload your fruit photo</strong>
              <p>Supports JPG, JPEG, and PNG · best with one clear fruit in frame</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown(
    """
    <div class="footer-note">
      Made with 🍎, 🍊, and a little bit of machine learning ✨<br>
      The Slices Classifier · TensorFlow + Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
