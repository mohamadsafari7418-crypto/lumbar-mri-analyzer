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
# RTL + Responsive CSS
# =========================================================

st.markdown(
    """
    <style>

    html, body, [class*="css"] {
        direction: rtl;
        text-align: right;
    }

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    .main-title {
        font-size: 32px;
        font-weight: 800;
        line-height: 1.5;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 15px;
        line-height: 1.7;
        margin-bottom: 20px;
    }

    .info-box,
    .warning-box,
    .success-box {
        padding: 14px;
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
    }

    .success-box {
        background-color: #f0fdf4;
        border: 1px solid #bbf7d0;
    }

    .stButton > button {
        width: 100%;
        min-height: 48px;
        border-radius: 10px;
        font-size: 16px;
        font-weight: 600;
    }

    [data-testid="stImage"] img {
        max-width: 100%;
        height: auto;
        border-radius: 10px;
    }

    [data-testid="stSidebar"] {
        direction: rtl;
    }

    [data-testid="stSidebar"] * {
        text-align: right;
    }

    @media (max-width: 768px) {

        .block-container {
            padding-left: 0.7rem;
            padding-right: 0.7rem;
            padding-top: 0.8rem;
        }

        .main-title {
            font-size: 23px;
        }

        .subtitle {
            font-size: 13px;
        }

        h1 {
            font-size: 23px !important;
        }

        h2 {
            font-size: 20px !important;
        }

        h3 {
            font-size: 17px !important;
        }

        .info-box,
        .warning-box,
        .success-box {
            padding: 11px;
            font-size: 13px;
        }

        .stButton > button {
            min-height: 50px;
            font-size: 15px;
        }

        section.main {
            overflow-x: hidden;
        }

        [data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
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
# Image Loading
# =========================================================

def load_image(uploaded):
    return Image.open(uploaded).convert("RGB")


# =========================================================
# Image Processing
# =========================================================

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

    kernel_width = max(
        10,
        width // 15
    )

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (kernel_width, 1)
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

    if projection.max() == 0:
        return []

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

            if y - current_group[-1] <= 5:
                current_group.append(y)

            else:
                groups.append(
                    current_group
                )

                current_group = [
                    y
                ]

        groups.append(
            current_group
        )

    line_positions = []

    for group in groups:

        if len(group) >= 2:

            center_y = int(
                np.mean(group)
            )

            line_positions.append(
                center_y
            )

    return line_positions


# =========================================================
# Create MRI Overlay
# =========================================================

def create_overlay(
    image,
    line_positions,
    show_grid=True,
    show_labels=True
):

    result = image.copy()

    draw = ImageDraw.Draw(
        result
    )

    width, height = result.size

    # -----------------------------------------------------
    # Guide lines
    # -----------------------------------------------------

    if show_grid:

        for y in line_positions:

            draw.line(
                [
                    (0, y),
                    (width, y)
                ],
                fill=(255, 180, 0),
                width=2
            )

    # -----------------------------------------------------
    # Estimate five disc levels
    # -----------------------------------------------------

    if len(line_positions) >= 6:

        selected = line_positions[:6]

        disc_positions = []

        for i in range(5):

            y1 = selected[i]
            y2 = selected[i + 1]

            disc_y = int(
                (y1 + y2) / 2
            )

            disc_positions.append(
                disc_y
            )

        labels = [
            "L1-L2",
            "L2-L3",
            "L3-L4",
            "L4-L5",
            "L5-S1",
        ]

        for label, y in zip(
            labels,
            disc_positions
        ):

            draw.line(
                [
                    (0, y),
                    (width, y)
                ],
                fill=(0, 255, 0),
                width=3
            )

            if show_labels:

                draw.text(
                    (10, max(5, y - 25)),
                    label,
                    fill=(0, 255, 0)
                )

    # -----------------------------------------------------
    # Vertebral labels
    # -----------------------------------------------------

    if show_labels and len(line_positions) >= 6:

        vertebrae = [
            "L1",
            "L2",
            "L3",
            "L4",
            "L5",
            "S1"
        ]

        for label, y in zip(
            vertebrae,
            line_positions[:6]
        ):

            draw.text(
                (width - 45, max(5, y - 18)),
                label,
                fill=(255, 255, 0)
            )

    return result


# =========================================================
# Analysis
# =========================================================

def analyze_image(image):

    lines = detect_horizontal_lines(
        image
    )

    confidence = min(
        95,
        40 + len(lines) * 8
    )

    return {
        "lines": lines,
        "confidence": confidence
    }


# =========================================================
# Upload
# =========================================================

uploaded_file = st.file_uploader(
    "📤 تصویر Sagittal T2 را بارگذاری کنید",
    type=[
        "png",
        "jpg",
        "jpeg",
        "bmp",
        "tif",
        "tiff"
    ],
)


# =========================================================
# Main Analysis
# =========================================================

if uploaded_file is not None:

    try:

        image = load_image(
            uploaded_file
        )

        st.success(
            "تصویر با موفقیت بارگذاری شد."
        )

        st.subheader(
            "🖼️ تصویر ورودی"
        )

        st.image(
            image,
            use_container_width=True
        )

        if st.button(
            "🔍 شروع تحلیل MRI"
        ):

            with st.spinner(
                "در حال پردازش تصویر..."
            ):

                result = analyze_image(
                    image
                )

            confidence = result[
                "confidence"
            ]

            lines = result[
                "lines"
            ]

            st.divider()

            st.subheader(
                "📊 نتیجه تحلیل اولیه"
            )

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "تعداد خطوط شناسایی‌شده",
                    len(lines)
                )

            with col2:

                st.metric(
                    "Confidence",
                    f"{confidence}%"
                )

            # -------------------------------------------------
            # Confidence
            # -------------------------------------------------

            if confidence >= confidence_threshold:

                st.success(
                    f"Confidence قابل قبول است: {confidence}%"
                )

            else:

                st.warning(
                    f"Confidence پایین است: {confidence}%"
                )

            # -------------------------------------------------
            # Overlay
            # -------------------------------------------------

            overlay = create_overlay(
                image,
                lines,
                show_grid,
                show_labels
            )

            st.subheader(
                "🩻 Localization اولیه"
            )

            st.image(
                overlay,
                use_container_width=True
            )

            # -------------------------------------------------
            # Disc levels
            # -------------------------------------------------

            st.subheader(
                "📍 فضاهای دیسکی"
            )

            disc_levels = [
                "L1-L2",
                "L2-L3",
                "L3-L4",
                "L4-L5",
                "L5-S1"
            ]

            for level in disc_levels:

                st.write(
                    f"• {level} — بررسی اولیه"
                )

            # -------------------------------------------------
            # Warning
            # -------------------------------------------------

            st.markdown(
                """
                <div class="warning-box">
                ⚠️ <b>هشدار مهم:</b><br>
                این نرم‌افزار یک Prototype پژوهشی است.
                نتایج Localization و Confidence تشخیص پزشکی
                محسوب نمی‌شوند و باید توسط پزشک متخصص رادیولوژی
                بررسی و تأیید شوند.
                </div>
                """,
                unsafe_allow_html=True,
            )

    except Exception as e:

        st.error(
            "خطا در پردازش تصویر"
        )

        st.code(
            str(e)
        )


# =========================================================
# No Image
# =========================================================

else:

    st.info(
        "برای شروع، یک تصویر Sagittal T2 را انتخاب کنید."
    )

    st.markdown(
        """
        <div class="warning-box">
        🧪 <b>Prototype V3.1</b><br>
        این نسخه برای تست رابط کاربری، پردازش اولیه تصویر
        و Localization آزمایشی طراحی شده است.
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# Footer
# =========================================================

st.divider()

st.caption(
    "Lumbar MRI Analyzer V3.1 — Research Prototype | محمد صفری"
)
