import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2


# =========================================================
# Page Configuration
# =========================================================

st.set_page_config(
    page_title="Lumbar MRI Analyzer V3.1",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# Responsive RTL CSS
# =========================================================

st.markdown(
    """
    <style>

    /* ---------- General ---------- */

    html, body, [class*="css"] {
        direction: rtl;
        text-align: right;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    /* ---------- Titles ---------- */

    .main-title {
        font-size: 34px;
        font-weight: 800;
        line-height: 1.4;
        margin-bottom: 6px;
    }

    .subtitle {
        font-size: 16px;
        line-height: 1.6;
        margin-bottom: 25px;
    }

    /* ---------- Boxes ---------- */

    .info-box,
    .warning-box,
    .success-box {
        padding: 15px;
        border-radius: 12px;
        margin-bottom: 15px;
        line-height: 1.8;
    }

    .info-box {
        background-color: #f1f5f9;
        border: 1px solid #dbe3ec;
    }

    .warning-box {
        background-color: #fff7ed;
        border: 1px solid #fed7aa;
        margin-top: 20px;
    }

    .success-box {
        background-color: #f0fdf4;
        border: 1px solid #bbf7d0;
    }

    /* ---------- Buttons ---------- */

    .stButton > button {
        width: 100%;
        min-height: 48px;
        border-radius: 10px;
        font-size: 16px;
        font-weight: 600;
    }

    /* ---------- File uploader ---------- */

    [data-testid="stFileUploader"] {
        width: 100%;
    }

    /* ---------- Images ---------- */

    [data-testid="stImage"] img {
        max-width: 100%;
        height: auto;
        border-radius: 10px;
    }

    /* ---------- Sidebar ---------- */

    [data-testid="stSidebar"] {
        direction: rtl;
    }

    [data-testid="stSidebar"] * {
        text-align: right;
    }

    /* ---------- Mobile ---------- */

    @media (max-width: 768px) {

        .block-container {
            padding-left: 0.8rem;
            padding-right: 0.8rem;
            padding-top: 1rem;
        }

        .main-title {
            font-size: 24px;
            line-height: 1.5;
        }

        .subtitle {
            font-size: 13px;
            line-height: 1.7;
        }

        h1 {
            font-size: 24px !important;
        }

        h2 {
            font-size: 21px !important;
        }

        h3 {
            font-size: 18px !important;
        }

        .info-box,
        .warning-box,
        .success-box {
            padding: 12px;
            font-size: 14px;
        }

        .stButton > button {
            min-height: 52px;
            font-size: 15px;
        }

        [data-testid="stFileUploader"] {
            font-size: 14px;
        }

        [data-testid="stImage"] img {
            width: 100% !important;
            max-width: 100% !important;
        }

        /* Prevent horizontal overflow */

        section.main {
            overflow-x: hidden;
        }

        /* Better mobile columns */

        [data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
        }

    }

    /* ---------- Very small phones ---------- */

    @media (max-width: 480px) {

        .block-container {
            padding-left: 0.55rem;
            padding-right: 0.55rem;
        }

        .main-title {
            font-size: 21px;
        }

        .subtitle {
            font-size: 12px;
        }

        .stButton > button {
            min-height: 50px;
        }

    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# Header
# =========================================================

st.markdown(
    '<div class="main-title">🩻 آنالایزر MRI ستون فقرات کمری — V3.1</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">Lumbar MRI Analyzer | Prototype for Sagittal T2 Analysis</div>',
    unsafe_allow_html=True,
)


# =========================================================
# Sidebar
# =========================================================

with st.sidebar:

    st.header("⚙️ تنظیمات تحلیل")

    confidence_threshold = st.slider(
        "حداقل Confidence",
        min_value=0,
        max_value=100,
        value=60,
        step=5,
    )

    show_grid = st.checkbox(
        "نمایش خطوط راهنما",
        value=True,
    )

    show_labels = st.checkbox(
        "نمایش نام مهره‌ها و دیسک‌ها",
        value=True,
    )

    st.divider()

    st.markdown("### وضعیت سیستم")

    st.success("رابط کاربری: فعال")
    st.success("پردازش تصویر: فعال")
    st.info("مدل AI واقعی: در حال توسعه")


# =========================================================
# Information
# =========================================================

st.markdown(
    """
    <div class="info-box">
    <b>هدف نسخه V3.1:</b><br>
    بارگذاری تصویر Sagittal MRI و انجام Localization اولیه
    برای مهره‌های کمری و فضاهای دیسکی L1 تا S1.
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# Upload
# =========================================================

uploaded_file = st.file_uploader(
    "📤 تصویر Sagittal T2 را بارگذاری کنید",
    type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
)


# =========================================================
# Functions
# =========================================================

def load_image(uploaded):
    image = Image.open(uploaded).convert("RGB")
    return image


def detect_horizontal_lines(image):

    img = np.array(image)

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2GRAY
    )

    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    edges = cv2.Canny(
        gray,
        40,
        120
    )

    height, width = gray.shape

    # Central spinal region

    x1 = int(width * 0.25)
    x2 = int(width * 0.75)

    roi = edges[:, x1:x2]

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(10, width // 15), 1)
    )

    horizontal = cv2.morphologyEx(
        roi,
        cv2.MORPH_OPEN,
        horizontal_kernel
    )

    projection = np.sum(
        horizontal,
        axis=1
    )

    threshold = np.percentile(
        projection,
        90
    )

    candidates = np.where(
        projection > threshold
    )[0]

    groups = []

    if len(candidates) > 0:

        current_group = [
            candidates[0]
        ]

        for y in candidates[1:]:

            if y - current:
