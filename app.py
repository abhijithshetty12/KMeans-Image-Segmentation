"""Interactive K-Means image color segmentation — Module 4 IE demonstration."""
from __future__ import annotations

from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="ColorLab | K-Means Segmentation", page_icon="🎨", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1250px;}
h1 {letter-spacing: -.045em;}
div[data-testid="stMetric"] {border:1px solid rgba(128,128,128,.2); border-radius:12px; padding:14px;}
</style>
""", unsafe_allow_html=True)

MAX_SIDE = 900
MAX_FIT_PIXELS = 100_000
RANDOM_STATE = 42


def png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def prepare_image(data: bytes) -> Image.Image:
    src = Image.open(BytesIO(data))
    src = ImageOps.exif_transpose(src).convert("RGB")
    src.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
    return src


def suggest_k(pixels: np.ndarray, max_k: int = 8) -> tuple[int, list[tuple[int, float]]]:
    """Heuristic K search via sampled silhouette scores; NOT the PCA method from the cited paper."""
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


def segment(pixels: np.ndarray, k: int, use_minibatch: bool) -> tuple[np.ndarray, np.ndarray]:
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


def rgb_hex(c: np.ndarray) -> str:
    return "#" + "".join(f"{int(v):02x}" for v in c)


st.title("🎨 ColorLab — K-Means Image Segmentation")
st.caption("Upload any photo → cluster pixels by RGB similarity → inspect each separated color region. Runs locally on your computer.")

with st.sidebar:
    st.header("Settings")
    uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp"])
    mode = st.radio("Number of clusters", ["Manual K", "Auto K (silhouette heuristic)"], help="Auto K tests 2–8 groups on sampled pixels. It does not reproduce the paper's PCA-based technique.")
    k = st.slider("K (number of colors)", 2, 12, 5) if mode == "Manual K" else None
    fast = st.toggle("Use Mini-Batch K-Means (faster)", value=True)
    show_transparent = st.toggle("Separate clusters with transparency", value=True)
    st.info("This is RGB color clustering—not semantic object detection. Similar-colored objects can end up in the same cluster.")

if uploaded is None:
    st.info("👈 Upload a photo using the sidebar to begin.")
    st.markdown("**What you'll see:** original photo, K-color segmented photo, each isolated color group, cluster percentages, and downloadable PNG/ZIP files.")
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
m3.metric("Pixels analyzed", f"{h*w:,}")
if auto_scores:
    st.caption("Auto K chose the highest sampled silhouette score: " + ", ".join(f"K={candidate}: {score:.3f}" for candidate, score in auto_scores))

left, right = st.columns(2)
with left:
    st.image(image, caption="Original input", use_container_width=True)
with right:
    st.image(output, caption=f"K-Means segmented image (K={k})", use_container_width=True)
    st.download_button("Download segmented image", png_bytes(output), file_name=f"segmented_k{k}.png", mime="image/png")

st.subheader("Separated color clusters")
st.caption("Each image retains only pixels assigned to that cluster. Transparent background means the other clusters are hidden.")
files = [("original.png", png_bytes(image)), (f"segmented_k{k}.png", png_bytes(output))]
for group_start in range(0, len(order), 3):
    cols = st.columns(3)
    for col, cluster_id in zip(cols, order[group_start:group_start + 3]):
        mask = shape_labels == cluster_id
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[..., :3] = arr if show_transparent else palette[cluster_id]
        rgba[..., 3] = np.where(mask, 255, 0).astype(np.uint8)
        layer = Image.fromarray(rgba, "RGBA")
        name = f"cluster_{int(cluster_id)+1}.png"
        files.append((name, png_bytes(layer)))
        with col:
            st.image(layer, caption=f"Cluster {int(cluster_id)+1} • {rgb_hex(palette[cluster_id])} • {100*counts[cluster_id]/len(labels):.1f}% of pixels", use_container_width=True)
            st.download_button(f"Download cluster {int(cluster_id)+1}", png_bytes(layer), file_name=name, mime="image/png", key=name)

st.subheader("Cluster distribution")
chart_data = {f"Cluster {int(i)+1} ({rgb_hex(palette[i])})": round(float(counts[i]) * 100 / len(labels), 2) for i in order}
st.bar_chart(chart_data, horizontal=True)

zip_buffer = BytesIO()
with ZipFile(zip_buffer, "w", compression=ZIP_DEFLATED) as zf:
    for filename, blob in files:
        zf.writestr(filename, blob)
st.download_button("⬇️ Download all outputs (ZIP)", zip_buffer.getvalue(), file_name="kmeans_segmentation_results.zip", mime="application/zip", type="primary")
st.caption("Research note: Automatic K uses a silhouette-score heuristic, not the PCA/weighted-variance algorithm proposed in the selected research paper.")
