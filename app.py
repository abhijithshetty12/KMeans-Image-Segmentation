from __future__ import annotations

from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Chroma Cluster", page_icon="🎨", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
  --primary: #2563EB;
  --primary-strong: #1D4ED8;
  --surface: #0B0B0C;
  --surface-elevated: #111214;
  --neutral: #000000;
  --on-surface: #F7F8F8;
  --on-surface-muted: #A1A1AA;
  --border-subtle: #1B1D22;
  --error: #EF4444;
  --r-sm: 4px;
  --r-md: 6px;
  --r-lg: 12px;
  --r-xl: 16px;
}

html, body, .stApp, button, input, textarea, select, label, p, div, li,
h1, h2, h3, h4, h5, h6 {
  font-family: "Söhne", "Soehne", "Inter", -apple-system, BlinkMacSystemFont,
    "Segoe UI", Helvetica, Arial, sans-serif !important;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

.stApp {
  background: var(--neutral) !important;
  color: var(--on-surface) !important;
  min-height: 100vh;
}

#MainMenu, footer { visibility: hidden !important; }
header[data-testid="stHeader"] { background: transparent !important; }

.block-container {
  padding-top: 2.5rem !important;
  padding-bottom: 5rem !important;
  max-width: 1120px !important;
}

h1, h2, h3, h4, h5, h6 {
  color: var(--on-surface) !important;
  letter-spacing: -0.02em !important;
}

p, span, li, div { color: inherit; }

::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: var(--neutral); }
::-webkit-scrollbar-thumb { background: #1B1D22; border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: #26282e; }

/* Sidebar Open/Close Button Override */
[data-testid="collapsedControl"] button,
[data-testid="stSidebarCollapseButton"] button {
  font-size: 0 !important;
  color: transparent !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  width: 36px !important;
  height: 36px !important;
  padding: 0 !important;
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}
[data-testid="collapsedControl"] button *,
[data-testid="stSidebarCollapseButton"] button * {
  display: none !important;
}
[data-testid="collapsedControl"] button::before,
[data-testid="stSidebarCollapseButton"] button::before {
  content: "" !important;
  display: block !important;
  width: 22px !important;
  height: 22px !important;
  background-size: contain !important;
  background-repeat: no-repeat !important;
  background-position: center !important;
}
[data-testid="collapsedControl"] button::before {
  background-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="%23A1A1AA" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><line x1="9" y1="3" x2="9" y2="21"></line></svg>') !important;
}
[data-testid="stSidebarCollapseButton"] button::before {
  background-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="%23A1A1AA" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><line x1="9" y1="3" x2="9" y2="21"></line><polyline points="16 15 13 12 16 9"></polyline></svg>') !important;
}

/* Browse Files Upload Button Override */
div[data-testid="stFileUploaderDropzone"] button {
  font-size: 0 !important;
  color: transparent !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  padding: 0 !important;
  width: 36px !important;
  height: 36px !important;
  border-radius: var(--r-md) !important;
}
div[data-testid="stFileUploaderDropzone"] button * {
  display: none !important;
}
div[data-testid="stFileUploaderDropzone"] button::before {
  content: "" !important;
  display: block !important;
  width: 18px !important;
  height: 18px !important;
  background-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="%23F7F8F8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="17 8 12 3 7 8"></polyline><line x1="12" y1="3" x2="12" y2="15"></line></svg>') !important;
  background-size: contain !important;
  background-repeat: no-repeat !important;
}

/* Delete File (Cross Mark) Override */
div[data-testid="stFileUploaderDeleteBtn"] button,
button[title="Delete file"],
button[aria-label="Delete file"] {
  font-size: 0 !important;
  color: transparent !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  width: 28px !important;
  height: 28px !important;
  padding: 0 !important;
  border-radius: 6px !important;
  background: transparent !important;
  border: none !important;
  transition: background 0.15s ease !important;
}
div[data-testid="stFileUploaderDeleteBtn"] button:hover,
button[title="Delete file"]:hover,
button[aria-label="Delete file"]:hover {
  background: rgba(255, 255, 255, 0.1) !important;
}
div[data-testid="stFileUploaderDeleteBtn"] button *,
button[title="Delete file"] *,
button[aria-label="Delete file"] * {
  display: none !important;
}
div[data-testid="stFileUploaderDeleteBtn"] button::before,
button[title="Delete file"]::before,
button[aria-label="Delete file"]::before {
  content: "" !important;
  display: block !important;
  width: 14px !important;
  height: 14px !important;
  background-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="%23A1A1AA" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>') !important;
  background-size: contain !important;
  background-repeat: no-repeat !important;
}

.qb-hero {
  padding: 8px 0 4px 0;
  margin-bottom: 4px;
  border-bottom: 1px solid var(--border-subtle);
  padding-bottom: 24px;
}
.qb-hero h1 {
  font-size: 30px !important;
  line-height: 36px !important;
  font-weight: 600 !important;
  letter-spacing: -0.75px !important;
  margin: 0 !important;
  color: var(--on-surface) !important;
}
.qb-hero .qb-sub {
  margin: 8px 0 0 0 !important;
  font-size: 14px !important;
  line-height: 22.75px !important;
  color: var(--on-surface-muted) !important;
  font-weight: 400 !important;
  letter-spacing: 0 !important;
  max-width: 720px;
}

.qb-section {
  font-size: 17px;
  line-height: 20px;
  font-weight: 600;
  letter-spacing: 0;
  color: var(--on-surface);
  margin: 40px 0 6px 0;
}
.qb-section + .qb-caption {
  margin-bottom: 18px;
}

.qb-caption {
  font-size: 13px;
  line-height: 20px;
  color: var(--on-surface-muted);
  margin: 0 0 16px 0;
}

.qb-side-title {
  font-size: 14px;
  font-weight: 500;
  line-height: 20px;
  color: var(--on-surface);
  margin: 4px 0 16px 0;
  letter-spacing: 0;
}

section[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border-subtle) !important;
}
section[data-testid="stSidebar"] > div { background: transparent !important; }
section[data-testid="stSidebar"] .block-container {
  padding-top: 1.75rem !important;
  padding-bottom: 2rem !important;
}

div[data-testid="stMetric"] {
  background: var(--surface) !important;
  border: 1px solid var(--border-subtle) !important;
  border-radius: var(--r-lg) !important;
  padding: 16px 18px !important;
  box-shadow: none !important;
}
div[data-testid="stMetric"] label {
  color: var(--on-surface-muted) !important;
  font-size: 12px !important;
  font-weight: 500 !important;
  letter-spacing: 0 !important;
}
div[data-testid="stMetricValue"] {
  color: var(--on-surface) !important;
  font-size: 22px !important;
  font-weight: 600 !important;
  letter-spacing: -0.4px !important;
  line-height: 1.2 !important;
}

div[data-testid="stFileUploaderDropzone"],
div[data-testid="stFileUploader"] section {
  background: var(--surface) !important;
  border: 1px dashed var(--border-subtle) !important;
  border-radius: var(--r-md) !important;
  padding: 1rem !important;
  box-shadow: none !important;
  transition: border-color 0.15s ease !important;
}
div[data-testid="stFileUploaderDropzone"]:hover,
div[data-testid="stFileUploader"] section:hover {
  border-color: #26282e !important;
}

.stButton > button,
.stDownloadButton > button {
  background: var(--surface-elevated) !important;
  border: 1px solid var(--border-subtle) !important;
  border-radius: var(--r-md) !important;
  color: var(--on-surface) !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  letter-spacing: 0 !important;
  padding: 8px 14px !important;
  min-height: 36px !important;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.04) !important;
  transition: background 0.15s ease, border-color 0.15s ease !important;
}
.stButton > button:hover,
.stDownloadButton > button:hover {
  background: #18191c !important;
  border-color: #26282e !important;
}
.stButton > button:focus,
.stDownloadButton > button:focus {
  outline: none !important;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.04), 0 0 0 2px rgba(37,99,235,0.35) !important;
}
.stButton > button[kind="primary"],
.stDownloadButton > button[kind="primary"] {
  background: var(--primary) !important;
  border-color: var(--primary) !important;
  color: #F7F8F8 !important;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.12) !important;
}
.stButton > button[kind="primary"]:hover,
.stDownloadButton > button[kind="primary"]:hover {
  background: var(--primary-strong) !important;
  border-color: var(--primary-strong) !important;
}

div[data-testid="stRadio"] > div[role="radiogroup"] { gap: 6px !important; }
div[data-testid="stRadio"] label {
  background: transparent !important;
  border: 1px solid transparent !important;
  border-radius: var(--r-md) !important;
  padding: 8px 10px !important;
  transition: background 0.15s ease, border-color 0.15s ease !important;
}
div[data-testid="stRadio"] label:hover {
  background: var(--surface-elevated) !important;
  border-color: var(--border-subtle) !important;
}
div[data-testid="stRadio"] label p {
  font-size: 13px !important;
  font-weight: 500 !important;
  color: var(--on-surface) !important;
}

div[data-baseweb="checkbox"] div[role="checkbox"][aria-checked="true"],
div[data-baseweb="toggle"] button[aria-checked="true"] {
  background: var(--primary) !important;
  border-color: var(--primary) !important;
  box-shadow: none !important;
}

div[data-testid="stSlider"] div[role="slider"] {
  background: var(--on-surface) !important;
  border: 1px solid var(--on-surface) !important;
  box-shadow: none !important;
}

div[data-testid="stAlert"] {
  background: var(--surface) !important;
  border: 1px solid var(--border-subtle) !important;
  border-radius: var(--r-lg) !important;
  color: var(--on-surface) !important;
  box-shadow: none !important;
  padding: 14px 16px !important;
}
div[data-testid="stAlert"] p,
div[data-testid="stAlert"] div {
  color: var(--on-surface) !important;
  font-size: 13px !important;
  line-height: 20px !important;
}

[data-testid="stImage"] img,
div[data-testid="stImage"] img {
  border-radius: var(--r-lg) !important;
  box-shadow: none !important;
  border: 1px solid var(--border-subtle) !important;
}
div[data-testid="stImage"] figcaption,
[data-testid="stImageCaption"] {
  color: var(--on-surface-muted) !important;
  font-size: 12px !important;
  text-align: center !important;
  padding-top: 8px !important;
  letter-spacing: 0 !important;
  font-weight: 500 !important;
}

.stCaption,
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p {
  color: var(--on-surface-muted) !important;
  font-size: 13px !important;
  line-height: 20px !important;
  letter-spacing: 0 !important;
}

.qb-welcome {
  margin-top: 32px;
  padding: 32px;
  border-radius: var(--r-lg);
  background: var(--surface);
  border: 1px solid var(--border-subtle);
}
.qb-welcome-title {
  font-size: 20px;
  line-height: 22.5px;
  font-weight: 600;
  letter-spacing: -0.375px;
  color: var(--on-surface);
  margin-bottom: 10px;
}
.qb-welcome-body {
  margin: 0 !important;
  font-size: 14px;
  line-height: 22.75px;
  color: var(--on-surface-muted);
  max-width: 640px;
}

.qb-auto-scores {
  margin-top: 12px;
  font-size: 13px;
  line-height: 20px;
  color: var(--on-surface-muted);
  border: 1px solid var(--border-subtle);
  border-radius: var(--r-lg);
  padding: 12px 14px;
  background: var(--surface);
  letter-spacing: 0;
}
.qb-auto-scores strong {
  color: var(--on-surface);
  font-weight: 600;
}

.qb-bar-chart {
  display: flex;
  flex-direction: column;
  gap: 14px;
  background: var(--surface);
  border: 1px solid var(--border-subtle);
  border-radius: var(--r-lg);
  padding: 22px 22px;
  margin-top: 14px;
}
.qb-bar-row {
  display: grid;
  grid-template-columns: minmax(140px, 1.2fr) 3fr auto;
  gap: 16px;
  align-items: center;
}
.qb-bar-label {
  font-size: 13px;
  font-weight: 500;
  color: var(--on-surface-muted);
  display: flex;
  align-items: center;
  gap: 10px;
  letter-spacing: 0;
  white-space: nowrap;
}
.qb-bar-dot {
  width: 10px;
  height: 10px;
  border-radius: 3px;
  flex: 0 0 auto;
  box-shadow: inset 0 0 0 1px rgba(255,255,255,0.08);
}
.qb-bar-track {
  height: 6px;
  background: var(--surface-elevated);
  border-radius: var(--r-sm);
  overflow: hidden;
}
.qb-bar-fill {
  height: 100%;
  border-radius: var(--r-sm);
}
.qb-bar-pct {
  font-size: 12px;
  font-weight: 500;
  color: var(--on-surface);
  min-width: 48px;
  text-align: right;
  font-variant-numeric: tabular-nums;
  letter-spacing: 0;
}

div[data-testid="stVerticalBlock"] { gap: 0.85rem; }
hr { border-color: var(--border-subtle) !important; }

@media (max-width: 768px) {
  .block-container {
    padding-left: 1rem !important;
    padding-right: 1rem !important;
    padding-top: 1.5rem !important;
    padding-bottom: 3rem !important;
  }
  .qb-hero { padding-bottom: 20px; }
  .qb-hero h1 { font-size: 24px !important; line-height: 30px !important; }
  .qb-hero .qb-sub { font-size: 13px !important; }
  .qb-section { font-size: 16px; margin-top: 32px; }
  div[data-testid="stMetric"] { padding: 14px 14px !important; }
  div[data-testid="stMetricValue"] { font-size: 18px !important; }
  .qb-bar-chart { padding: 16px 14px; }
  .qb-bar-row { grid-template-columns: minmax(96px, 1.1fr) 2fr auto; gap: 10px; }
  .qb-bar-label { font-size: 12px; }
  .qb-bar-pct { font-size: 11px; min-width: 40px; }
  .qb-welcome { padding: 22px 20px; }
  .qb-welcome-title { font-size: 17px; }
  [data-testid="stImage"] img, div[data-testid="stImage"] img { border-radius: 10px !important; }
}
</style>
""", unsafe_allow_html=True)

MAX_SIDE = 900
MAX_FIT_PIXELS = 100_000
RANDOM_STATE = 42


def png_bytes(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def prepare_image(data):
    src = Image.open(BytesIO(data))
    src = ImageOps.exif_transpose(src).convert("RGB")
    src.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
    return src


def suggest_k(pixels, max_k=8):
    rng = np.random.default_rng(RANDOM_STATE)
    n = min(1800, len(pixels))
    sample = pixels[rng.choice(len(pixels), size=n, replace=False)]
    unique = len(np.unique(sample, axis=0))
    if unique < 3:
        return max(1, unique), []
    candidates = range(2, min(max_k, unique - 1) + 1)
    scores = []
    for k in candidates:
        model = KMeans(n_clusters=k, n_init=3, random_state=RANDOM_STATE, max_iter=100)
        labels = model.fit_predict(sample)
        if 1 < len(np.unique(labels)) < n:
            score = silhouette_score(sample, labels, sample_size=min(700, n), random_state=RANDOM_STATE)
            scores.append((k, float(score)))
    return (max(scores, key=lambda item: item[1])[0] if scores else 2), scores


def segment(pixels, k, use_minibatch):
    rng = np.random.default_rng(RANDOM_STATE)
    fit_pixels = pixels
    if len(pixels) > MAX_FIT_PIXELS:
        chosen = rng.choice(len(pixels), size=MAX_FIT_PIXELS, replace=False)
        fit_pixels = pixels[chosen]
    if use_minibatch:
        model = MiniBatchKMeans(n_clusters=k, batch_size=2048, n_init=3, random_state=RANDOM_STATE)
    else:
        model = KMeans(n_clusters=k, n_init=5, random_state=RANDOM_STATE)
    model.fit(fit_pixels)
    labels = model.predict(pixels)
    palette = np.clip(np.rint(model.cluster_centers_), 0, 255).astype(np.uint8)
    return labels, palette


def rgb_hex(c):
    return "#" + "".join(f"{int(v):02x}" for v in c)


def cluster_row_html(cluster_id, pct, hex_color):
    width = max(1.5, pct)
    return (
        f'<div class="qb-bar-row">'
        f'<div class="qb-bar-label"><span class="qb-bar-dot" style="background:{hex_color};"></span>Cluster {int(cluster_id) + 1}</div>'
        f'<div class="qb-bar-track"><div class="qb-bar-fill" style="width:{width:.1f}%; background:{hex_color};"></div></div>'
        f'<div class="qb-bar-pct">{pct:.1f}%</div>'
        f'</div>'
    )


st.markdown("""
<div class="qb-hero">
  <h1>Chroma Cluster</h1>
  <p class="qb-sub">K-Means color segmentation for any photo. Upload an image, cluster pixels by RGB similarity, and inspect each separated color region. Runs locally.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="qb-side-title">Settings</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp"])
    mode = st.radio(
        "Number of clusters",
        ["Manual K", "Auto K (silhouette heuristic)"],
        help="Auto K tests 2–8 groups on sampled pixels. It does not reproduce the paper's PCA-based technique.",
    )
    k = st.slider("K (number of colors)", 2, 12, 5) if mode == "Manual K" else None
    fast = st.toggle("Use Mini-Batch K-Means (faster)", value=True)
    show_transparent = st.toggle("Separate clusters with transparency", value=True)
    st.info("RGB color clustering only — not semantic object detection. Similar-colored objects can share a cluster.")

if uploaded is None:
    st.markdown("""
    <div class="qb-welcome">
      <div class="qb-welcome-title">Upload a photo to begin</div>
      <p class="qb-welcome-body">You'll get a K-color segmented image, each isolated color cluster, pixel percentages, and downloadable PNG / ZIP exports.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

try:
    image = prepare_image(uploaded.getvalue())
except Exception as error:
    st.error(f"Unable to open this image: {error}")
    st.stop()

arr = np.asarray(image, dtype=np.uint8)
h, w = arr.shape[:2]
pixels = arr.reshape(-1, 3).astype(np.float32)
unique_colors = len(np.unique(pixels, axis=0))

if unique_colors < 2:
    st.warning("This image contains just one color, so multiple clusters cannot be formed.")
    st.image(image, caption="Original image")
    st.stop()

with st.spinner("Clustering colors…"):
    auto_scores = []
    if k is None:
        k, auto_scores = suggest_k(pixels)
    k = min(k, unique_colors)
    labels, palette = segment(pixels, k, fast)
    shape_labels = labels.reshape(h, w)
    output = Image.fromarray(palette[labels].reshape(h, w, 3), "RGB")
    counts = np.bincount(labels, minlength=k)
    order = np.argsort(-counts)

m1, m2, m3 = st.columns(3)
m1.metric("Image resolution", f"{w} × {h}")
m2.metric("Colors / clusters", str(k))
m3.metric("Pixels analyzed", f"{h * w:,}")

if auto_scores:
    scores_text = " · ".join(f"K={candidate}: {score:.3f}" for candidate, score in auto_scores)
    st.markdown(
        f'<div class="qb-auto-scores">Auto K selected <strong>K={k}</strong> via highest sampled silhouette · {scores_text}</div>',
        unsafe_allow_html=True,
    )

left, right = st.columns(2)
with left:
    st.image(image, caption="Original input", use_container_width=True)
with right:
    st.image(output, caption=f"K-Means segmented (K={k})", use_container_width=True)
    st.download_button(
        "Download segmented image",
        png_bytes(output),
        file_name=f"segmented_k{k}.png",
        mime="image/png",
    )

st.markdown('<div class="qb-section">Separated color clusters</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="qb-caption">Each image retains only pixels assigned to that cluster. Transparent background means the other clusters are hidden.</div>',
    unsafe_allow_html=True,
)

files = [("original.png", png_bytes(image)), (f"segmented_k{k}.png", png_bytes(output))]

for group_start in range(0, len(order), 3):
    cols = st.columns(3)
    for col, cluster_id in zip(cols, order[group_start:group_start + 3]):
        mask = shape_labels == cluster_id
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[..., :3] = arr if show_transparent else palette[cluster_id]
        rgba[..., 3] = np.where(mask, 255, 0).astype(np.uint8)
        layer = Image.fromarray(rgba, "RGBA")
        name = f"cluster_{int(cluster_id) + 1}.png"
        files.append((name, png_bytes(layer)))
        pct = 100 * counts[cluster_id] / len(labels)
        with col:
            st.image(
                layer,
                caption=f"Cluster {int(cluster_id) + 1} · {rgb_hex(palette[cluster_id])} · {pct:.1f}%",
                use_container_width=True,
            )
            st.download_button(
                f"Download cluster {int(cluster_id) + 1}",
                png_bytes(layer),
                file_name=name,
                mime="image/png",
                key=name,
            )

st.markdown('<div class="qb-section">Cluster distribution</div>', unsafe_allow_html=True)
rows_html = "".join(
    cluster_row_html(cid, 100 * counts[cid] / len(labels), rgb_hex(palette[cid]))
    for cid in order
)
st.markdown(f'<div class="qb-bar-chart">{rows_html}</div>', unsafe_allow_html=True)

zip_buffer = BytesIO()
with ZipFile(zip_buffer, "w", compression=ZIP_DEFLATED) as zf:
    for filename, blob in files:
        zf.writestr(filename, blob)

st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.download_button(
        "Download all outputs (ZIP)",
        zip_buffer.getvalue(),
        file_name="kmeans_segmentation_results.zip",
        mime="application/zip",
        type="primary",
        use_container_width=True,
    )