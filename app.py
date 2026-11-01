"""Sentinel AI — simple Streamlit UI for CNN image/video detection."""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image

from detector import predict_image, predict_video

IMAGE_TYPES = {"jpg", "jpeg", "png", "webp", "bmp"}
VIDEO_TYPES = {"mp4", "mov", "avi", "mkv", "webm"}
MAX_UPLOAD_BYTES = 25 * 1024 * 1024

st.set_page_config(page_title="Sentinel AI", page_icon="🛡️", layout="wide")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    :root {
        --ink: #111416;
        --paper: #fff4b8;
        --surface: #ffffff;
        --yellow: #ffd91a;
        --line: 2px solid var(--ink);
        --hard-shadow: 6px 6px 0 var(--ink);
    }

    html, body, [class*="stApp"] {
        color: var(--ink);
        font-family: 'Space Grotesk', 'Trebuchet MS', sans-serif;
    }
    [data-testid="stAppViewContainer"] {
        background-color: var(--paper);
        background-image:
            linear-gradient(rgba(17, 20, 22, .045) 1px, transparent 1px),
            linear-gradient(90deg, rgba(17, 20, 22, .045) 1px, transparent 1px);
        background-size: 28px 28px;
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stDecoration"] { background: var(--yellow); }
    .block-container {
        max-width: 980px;
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }
    .block-container > div { animation: rise-in .5s ease-out both; }
    @keyframes rise-in {
        from { opacity: 0; transform: translateY(12px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .sentinel-mark {
        display: flex;
        align-items: center;
        gap: .55rem;
        margin: 0 0 .8rem;
        color: var(--ink);
        font: 500 12px 'DM Mono', monospace;
        text-transform: uppercase;
    }
    .sentinel-mark::before {
        content: '';
        width: 12px;
        height: 12px;
        background: var(--yellow);
        border: var(--line);
        box-shadow: 2px 2px 0 var(--ink);
    }
    h1 {
        margin: 0 0 .15rem;
        font-size: 52px !important;
        line-height: 1.02 !important;
        font-weight: 700 !important;
        color: var(--ink) !important;
    }
    .block-container [data-testid="stCaptionContainer"] {
        color: var(--ink);
    }
    [data-testid="stFileUploader"] section {
        border: 2px dashed var(--ink);
        border-radius: 2px;
        background: var(--surface);
        box-shadow: var(--hard-shadow);
        transition: transform .2s ease, box-shadow .2s ease, background-color .2s ease;
    }
    [data-testid="stFileUploader"] section * { color: var(--ink) !important; }
    [data-testid="stFileUploader"] label { color: var(--ink) !important; }
    [data-testid="stFileUploader"] section:hover {
        transform: translate(-2px, -2px);
        box-shadow: 8px 8px 0 var(--ink);
        background: #ffec80;
    }
    [data-testid="stFileUploaderFile"] {
        padding: .45rem .65rem;
        border: var(--line) !important;
        border-radius: 2px !important;
        background: var(--yellow) !important;
        box-shadow: 3px 3px 0 var(--ink);
        color: var(--ink) !important;
    }
    [data-testid="stFileUploaderFile"] * {
        color: var(--ink) !important;
    }
    [data-testid="stFileUploaderFile"] [title] {
        font-weight: 700 !important;
    }
    [data-testid="stFileUploader"] button,
    [data-testid="stDataFrame"] button {
        border: var(--line) !important;
        border-radius: 2px !important;
        background: var(--yellow) !important;
        color: var(--ink) !important;
        box-shadow: 3px 3px 0 var(--ink);
        transition: transform .16s ease, box-shadow .16s ease, background-color .16s ease;
    }
    [data-testid="stFileUploader"] button:hover,
    [data-testid="stDataFrame"] button:hover {
        transform: translate(-2px, -2px);
        box-shadow: 5px 5px 0 var(--ink);
    }
    div[data-testid="stMetric"] {
        min-height: 112px;
        padding: 1rem 1.1rem;
        border: var(--line);
        border-radius: 2px;
        background: var(--surface);
        box-shadow: var(--hard-shadow);
        transition: transform .2s ease, box-shadow .2s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        box-shadow: 6px 9px 0 var(--ink);
    }
    div[data-testid="stHorizontalBlock"] > div:nth-child(1) [data-testid="stMetric"] { background: var(--yellow); }
    div[data-testid="stHorizontalBlock"] > div:nth-child(2) [data-testid="stMetric"] { background: var(--surface); }
    div[data-testid="stHorizontalBlock"] > div:nth-child(3) [data-testid="stMetric"] { background: var(--yellow); }
    [data-testid="stMetricLabel"] p {
        color: var(--ink) !important;
        font: 500 12px 'DM Mono', monospace !important;
        text-transform: uppercase;
    }
    [data-testid="stMetricValue"] {
        color: var(--ink) !important;
        font-weight: 700;
    }
    [data-testid="stProgress"] [role="progressbar"] {
        box-sizing: border-box;
        height: 14px;
        border: 2px solid var(--ink);
        border-radius: 0;
        background: var(--surface) !important;
        overflow: hidden;
    }
    [data-testid="stProgress"] [role="progressbar"] > div,
    [data-testid="stProgress"] [role="progressbar"] > div > div {
        background: var(--surface) !important;
        border-radius: 0 !important;
    }
    [data-testid="stProgress"] [role="progressbar"] > div > div > div {
        background: var(--yellow) !important;
        border-radius: 0 !important;
        transition: transform .55s cubic-bezier(.2, .8, .2, 1);
    }
    [data-testid="stProgress"] [data-testid="stMarkdownContainer"] p {
        color: var(--ink) !important;
    }
    [data-testid="stAlert"] {
        border: var(--line);
        border-radius: 2px;
        background: var(--yellow) !important;
        color: var(--ink);
        box-shadow: 4px 4px 0 var(--ink);
        animation: rise-in .35s ease-out both;
    }
    [data-testid="stAlert"] * {
        color: var(--ink) !important;
        background-color: transparent !important;
    }
    [data-testid="stAlert"] p { font-size: 1.05rem !important; font-weight: 600; }
    [data-testid="stAlert"] strong { font-weight: 700 !important; }
    [data-testid="stAlert"] svg { color: var(--ink) !important; fill: var(--ink) !important; }
    [data-testid="stImage"] img {
        border: var(--line);
        box-shadow: var(--hard-shadow);
    }
    [data-testid="stDataFrame"] {
        border: var(--line);
        box-shadow: var(--hard-shadow);
    }
    [data-testid="stExpander"] {
        border: var(--line);
        border-radius: 2px;
        background: var(--surface);
    }
    hr { border-color: var(--ink) !important; }

    @media (max-width: 680px) {
        .block-container { padding: 1.4rem 1rem 2rem; }
        h1 { font-size: 42px !important; }
        div[data-testid="stHorizontalBlock"] { flex-direction: column; gap: 1rem; }
        div[data-testid="stMetric"] { min-height: 92px; }
    }
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            animation-duration: .01ms !important;
            animation-iteration-count: 1 !important;
            scroll-behavior: auto !important;
            transition-duration: .01ms !important;
        }
    }
    </style>
    <div class="sentinel-mark">Sentinel / 01</div>
    """,
    unsafe_allow_html=True,
)
st.title("Sentinel AI")
st.caption("CNN detector for AI-generated images and videos")

st.markdown(
    "Upload a photo or a short video. Sentinel scores it with a pretrained "
    "**DenseNet-121** CNN and shows AI vs real probabilities."
)

uploaded = st.file_uploader(
    "Choose an image or video",
    type=sorted(IMAGE_TYPES | VIDEO_TYPES),
)


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _show_scores(result: dict) -> None:
    label = result["label"]
    if label == "AI-generated":
        st.error(f"Prediction: **{label}**")
    else:
        st.success(f"Prediction: **{label}**")

    col1, col2, col3 = st.columns(3)
    col1.metric("AI probability", _pct(result["ai_probability"]))
    col2.metric("Real probability", _pct(result["real_probability"]))
    col3.metric("Confidence", _pct(result["confidence"]))

    st.progress(min(max(result["ai_probability"], 0.0), 1.0), text="AI-generated likelihood")


if uploaded is not None:
    if uploaded.size > MAX_UPLOAD_BYTES:
        st.error("Please upload a file smaller than 25 MB.")
        st.stop()

    suffix = Path(uploaded.name).suffix.lower().lstrip(".")
    with st.spinner("Analyzing with CNN… videos are checking across sampled frames."):
        try:
            if suffix in IMAGE_TYPES:
                image = Image.open(uploaded)
                st.image(image, caption=uploaded.name, use_container_width=True)
                result = predict_image(image)
                _show_scores(result)
            elif suffix in VIDEO_TYPES:
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
                    tmp.write(uploaded.getbuffer())
                    tmp_path = tmp.name
                try:
                    result = predict_video(tmp_path)
                finally:
                    Path(tmp_path).unlink(missing_ok=True)

                _show_scores(result)
                st.caption(f"Averaged over {result['frames_scored']} sampled frames.")
                rows = [
                    {
                        "Frame": row["frame"],
                        "Prediction": row["label"],
                        "AI %": round(row["ai_probability"] * 100, 1),
                        "Real %": round(row["real_probability"] * 100, 1),
                    }
                    for row in result["per_frame"]
                ]
                st.dataframe(rows, hide_index=True, use_container_width=True)
            else:
                st.warning("Unsupported file type.")
        except Exception as exc:
            st.error("Could not analyze this file. Check that it is valid and try again.")
            st.info(
                "If this is the first run, check your internet connection so the public "
                "CNN weights can download. Otherwise, confirm the file is valid and supported."
            )
            with st.expander("Technical details"):
                st.code(str(exc))

st.divider()
st.caption(
    "This is a probabilistic estimate from a CNN, not forensic proof. "
    "Scores can be wrong on new generators, screenshots, or heavy compression."
)
