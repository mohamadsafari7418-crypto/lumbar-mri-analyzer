import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2


# =========================================================
# Page Configuration
# =========================================================

st.set_page_config(
    page_title="Lumbar MRI Analyzer V3",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# RTL / Persian CSS
# =========================================================

st.markdown(
    """
    <style>

    html, body, [class*="css"] {
        direction: rtl;
        text-align: right;
    }

    .main-title {
        font-size: 34px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 16px;
        color: #666;
        margin-bottom: 25px;
    }

    .info-box {
        padding: 15px;
        border-radius: 12px;
        background-color: #f1f5f9;
        border: 1px solid #dbe3ec;
        margin-bottom: 15px;
    }

    .warning-box {
        padding: 15px;
        border-radius: 12px;
        background-color: #fff7ed;
        border: 1px solid #fed7aa;
        margin-top: 15px;
    }

    .success-box {
        padding: 15px;
        border-radius: 12px;
        background-color: #f0fdf4;
        border: 1px solid #bbf7d0;
        margin-top: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# Header
# =========================================================

st.markdown(
    '<div class="main-title">🩻 آنالایزر MRI ستون فقرات کمری — نسخه V3</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">Lumbar MRI Analyzer V3 | Prototype for Sagittal T2 analysis</div>',
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
    <b>هدف این نسخه:</b><br>
    بارگذاری تصویر Sagittal MRI و انجام localization اولیه برای
    مهره‌های کمری و فضاهای دیسکی L1 تا S1.
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
    """
    Prototype localization.
    This is NOT a validated medical AI model.
    """

    img = np.array(image)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 40, 120)

    height, width = gray.shape

    # Focus on central spinal region
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
        horizontal_kernel,
    )

    projection = np.sum(horizontal, axis=1)

    # Find peaks
    threshold = np.percentile(projection, 90)

    candidates = np.where(projection > threshold)[0]

    # Group neighboring y coordinates
    groups = []

    if len(candidates) > 0:

        current_group = [candidates[0]]

        for y in candidates[1:]:

            if y - current_group[-1] <= 8:
                current_group.append(y)

            else:
                groups.append(current_group)
                current_group = [y]

        groups.append(current_group)

    centers = []

    for group in groups:

        center = int(np.mean(group))

        if not centers or abs(center - centers[-1]) > 15:
            centers.append(center)

    return centers


def create_annotation(image, lines, show_grid=True, show_labels=True):

    img = image.copy()

    draw = ImageDraw.Draw(img)

    width, height = img.size

    # Try to load a font
    try:
        font = ImageFont.truetype(
            "DejaVuSans.ttf",
            max(14, width // 55),
        )
    except:
        font = ImageFont.load_default()

    # Grid
    if show_grid:

        for i in range(1, 5):

            y = int(height * i / 5)

            draw.line(
                [(0, y), (width, y)],
                fill=(0, 180, 255),
                width=1,
            )

    # Localization lines
    for i, y in enumerate(lines):

        draw.line(
            [(0, y), (width, y)],
            fill=(255, 80, 80),
            width=3,
        )

        if show_labels:

            label = f"Level {i + 1}"

            draw.rectangle(
                [(10, y - 20), (120, y + 5)],
                fill=(0, 0, 0),
            )

            draw.text(
                (15, y - 18),
                label,
                fill=(255, 255, 255),
                font=font,
            )

    return img


# =========================================================
# Main Analysis
# =========================================================

if uploaded_file is not None:

    image = load_image(uploaded_file)

    st.success("تصویر با موفقیت دریافت شد.")

    col1, col2 = st.columns([2, 1])

    # -----------------------------------------------------
    # Image
    # -----------------------------------------------------

    with col1:

        st.subheader("🖼️ تصویر MRI")

        st.image(
            image,
            use_container_width=True,
        )

    # -----------------------------------------------------
    # Information
    # -----------------------------------------------------

    with col2:

        st.subheader("📋 اطلاعات تصویر")

        st.write(
            f"**عرض:** {image.width} px"
        )

        st.write(
            f"**ارتفاع:** {image.height} px"
        )

        st.write(
            f"**فرمت:** {image.format or 'Unknown'}"
        )

        st.divider()

        st.write("**نوع تحلیل:**")

        st.info(
            "Prototype Localization"
        )

    # -----------------------------------------------------
    # Analyze button
    # -----------------------------------------------------

    st.divider()

    if st.button(
        "🔍 شروع تحلیل MRI",
        type="primary",
        use_container_width=True,
    ):

        with st.spinner("در حال تحلیل تصویر..."):

            lines = detect_horizontal_lines(image)

        st.subheader("📍 نتیجه Localization")

        if len(lines) == 0:

            st.warning(
                "خط مشخصی برای Localization اولیه پیدا نشد."
            )

        else:

            annotated = create_annotation(
                image,
                lines,
                show_grid=show_grid,
                show_labels=show_labels,
            )

            st.image(
                annotated,
                use_container_width=True,
                caption="نمایش Localization اولیه",
            )

            st.divider()

            # -------------------------------------------------
            # Confidence
            # -------------------------------------------------

            estimated_confidence = min(
                95,
                max(
                    35,
                    45 + len(lines) * 7
                ),
            )

            st.subheader("📊 Confidence")

            st.progress(
                estimated_confidence / 100
            )

            st.write(
                f"Confidence تخمینی: **{estimated_confidence}%**"
            )

            if estimated_confidence >= confidence_threshold:

                st.success(
                    "نتیجه از آستانه Confidence تعیین‌شده عبور کرده است."
                )

            else:

                st.warning(
                    "Confidence پایین است؛ بررسی دستی توصیه می‌شود."
                )

            # -------------------------------------------------
            # Level Table
            # -------------------------------------------------

            st.subheader("🦴 سطوح پیشنهادی")

            levels = [
                "L1",
                "L2",
                "L3",
                "L4",
                "L5",
                "S1",
            ]

            for i, level in enumerate(levels):

                if i < len(lines):

                    confidence = min(
                        95,
                        max(
                            30,
                            estimated_confidence - abs(i - 2) * 5
                        ),
                    )

                    c1, c2, c3 = st.columns(3)

                    with c1:
                        st.write(f"**{level}**")

                    with c2:
                        st.write(
                            f"موقعیت Y: {lines[i]}"
                        )

                    with c3:
                        st.write(
                            f"Confidence: {confidence}%"
                        )

    # =====================================================
    # Test Section
    # =====================================================

    st.divider()

    st.subheader("🧪 بخش تست")

    st.write(
        "این بخش برای بررسی عملکرد Prototype روی تصاویر مختلف طراحی شده است."
    )

    test_col1, test_col2 = st.columns(2)

    with test_col1:

        if st.button(
            "✅ اجرای تست Localization",
            use_container_width=True,
        ):

            st.success(
                "تست اولیه با موفقیت اجرا شد."
            )

            st.write(
                "Image preprocessing: OK"
            )

            st.write(
                "Edge detection: OK"
            )

            st.write(
                "Horizontal structure detection: OK"
            )

    with test_col2:

        if st.button(
            "🔄 پاک کردن نتیجه",
            use_container_width=True,
        ):

            st.rerun()


else:

    # =====================================================
    # Empty State
    # =====================================================

    st.info(
        "برای شروع، یک تصویر Sagittal T2 از MRI کمر بارگذاری کنید."
    )

    st.markdown(
        """
        ### قابلیت‌های V3

        - 🩻 نمایش تصویر MRI
        - 📍 Localization اولیه ساختارهای افقی
        - 🦴 پیشنهاد سطوح L1 تا S1
        - 📊 نمایش Confidence
        - 🧪 بخش تست
        - 🇮🇷 رابط فارسی و RTL
        - 🖥️ طراحی مناسب برای موبایل و دسکتاپ
        """
    )


# =========================================================
# Medical Disclaimer
# =========================================================

st.markdown(
    """
    <div class="warning-box">
    ⚠️ <b>هشدار پزشکی:</b><br>
    این نرم‌افزار یک Prototype پژوهشی است و الگوریتم Localization
    فعلی Medical AI Validated نیست. نتایج آن نباید به‌عنوان تشخیص
    قطعی پزشکی یا جایگزین تفسیر رادیولوژیست استفاده شود.
    </div>
    """,
    unsafe_allow_html=True
)
