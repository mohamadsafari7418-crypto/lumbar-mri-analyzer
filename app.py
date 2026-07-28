import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import cv2


# =========================================================
# Page Configuration
# =========================================================

st.set_page_config(
    page_title="Lumbar MRI Analyzer V3.2",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================================================
# Mobile + RTL CSS
# =========================================================

st.markdown(
    """
    <style>

    /* ---------- Global ---------- */

    html, body, [class*="css"] {
        font-family: Tahoma, Arial, sans-serif !important;
    }

    .stApp {
        direction: rtl;
    }

    .main .block-container {
        max-width: 1200px;
        padding-top: 1rem;
        padding-bottom: 5rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }

    /* ---------- Persian text ---------- */

    h1, h2, h3, h4, p, label, div, span {
        text-align: right;
    }

    h1 {
        font-size: 2rem !important;
        line-height: 1.4 !important;
    }

    h2 {
        font-size: 1.5rem !important;
    }

    h3 {
        font-size: 1.25rem !important;
    }

    /* ---------- Header ---------- */

    .app-header {
        width: 100%;
        box-sizing: border-box;
        padding: 18px;
        margin-bottom: 18px;
        border-radius: 16px;
        background: linear-gradient(
            135deg,
            rgba(35, 40, 55, 0.95),
            rgba(20, 25, 35, 0.95)
        );
        border: 1px solid rgba(255,255,255,0.10);
    }

    .app-title {
        font-size: 25px;
        font-weight: 700;
        text-align: center !important;
        margin: 0;
        line-height: 1.5;
    }

    .app-subtitle {
        font-size: 14px;
        text-align: center !important;
        opacity: 0.85;
        margin-top: 7px;
        line-height: 1.7;
    }

    /* ---------- Cards ---------- */

    .info-card {
        width: 100%;
        box-sizing: border-box;
        padding: 16px;
        margin: 12px 0;
        border-radius: 14px;
        background: rgba(40, 42, 52, 0.75);
        border: 1px solid rgba(255,255,255,0.08);
    }

    .info-title {
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .info-text {
        font-size: 14px;
        line-height: 1.8;
    }

    /* ---------- Upload ---------- */

    [data-testid="stFileUploader"] {
        width: 100%;
        direction: rtl;
    }

    [data-testid="stFileUploaderDropzone"] {
        width: 100% !important;
        box-sizing: border-box !important;
        border-radius: 14px !important;
        padding: 18px !important;
    }

    [data-testid="stFileUploaderDropzoneInstructions"] {
        direction: rtl !important;
        text-align: center !important;
    }

    [data-testid="stFileUploaderDropzoneInstructions"] span {
        text-align: center !important;
    }

    /* ---------- Image container ---------- */

    [data-testid="stImage"] {
        width: 100% !important;
        max-width: 100% !important;
        margin: 0 auto !important;
    }

    [data-testid="stImage"] img {
        display: block !important;
        width: 100% !important;
        max-width: 100% !important;
        height: auto !important;
        object-fit: contain !important;
        border-radius: 12px;
    }

    /* ---------- Buttons ---------- */

    .stButton > button {
        width: 100%;
        min-height: 46px;
        border-radius: 12px;
        font-size: 15px;
        font-weight: 600;
    }

    /* ---------- Metrics ---------- */

    [data-testid="stMetric"] {
        direction: rtl;
        text-align: center;
    }

    [data-testid="stMetricLabel"] {
        text-align: center !important;
    }

    [data-testid="stMetricValue"] {
        text-align: center !important;
    }

    /* ---------- Hide Streamlit decoration ---------- */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    /* ---------- MOBILE ---------- */

    @media only screen and (max-width: 768px) {

        .main .block-container {
            padding-top: 0.5rem !important;
            padding-bottom: 6rem !important;
            padding-left: 10px !important;
            padding-right: 10px !important;
            width: 100% !important;
            max-width: 100% !important;
        }

        .app-header {
            padding: 14px 10px;
            margin-bottom: 12px;
            border-radius: 12px;
        }

        .app-title {
            font-size: 20px !important;
            line-height: 1.5 !important;
        }

        .app-subtitle {
            font-size: 12px !important;
            line-height: 1.8 !important;
        }

        h1 {
            font-size: 20px !important;
        }

        h2 {
            font-size: 18px !important;
        }

        h3 {
            font-size: 16px !important;
        }

        p, label {
            font-size: 14px !important;
            line-height: 1.8 !important;
        }

        .info-card {
            padding: 12px;
            margin: 8px 0;
            border-radius: 12px;
        }

        .info-title {
            font-size: 16px;
        }

        .info-text {
            font-size: 13px;
            line-height: 1.8;
        }

        [data-testid="stFileUploaderDropzone"] {
            padding: 12px !important;
        }

        [data-testid="stImage"] {
            width: 100% !important;
        }

        [data-testid="stImage"] img {
            width: 100% !important;
            height: auto !important;
            max-height: none !important;
        }

        .stButton > button {
            min-height: 48px;
            font-size: 14px;
        }

        /* جلوگیری از باریک شدن ستون‌ها */

        [data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
            min-width: 100% !important;
        }

        /* جلوگیری از اسکرول افقی */

        .stApp,
        section.main,
        .main,
        .block-container {
            overflow-x: hidden !important;
            max-width: 100vw !important;
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
    """
    <div class="app-header">

        <div class="app-title">
            🩻 Lumbar MRI Analyzer V3.2
        </div>

        <div class="app-subtitle">
            سامانه آزمایشی تحلیل و Localization تصاویر MRI ستون فقرات کمری
        </div>

        <div class="app-subtitle">
            طراحی‌شده برای نمایش مناسب در موبایل و دسکتاپ
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# Information
# =========================================================

st.markdown(
    """
    <div class="info-card">

        <div class="info-title">
            🎯 هدف نسخه V3.2
        </div>

        <div class="info-text">
            بارگذاری تصویر Sagittal MRI و انجام Localization اولیه
            فضاهای دیسک‌های کمری از L1-L2 تا L5-S1.
        </div>

        <div class="info-text">
            ⚠️ این نسخه Prototype پژوهشی است و برای تشخیص پزشکی
            یا جایگزینی گزارش رادیولوژیست اعتبارسنجی نشده است.
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# Upload
# =========================================================

st.markdown(
    """
    <div class="info-card">
        <div class="info-title">
            📤 تصویر ورودی
        </div>
        <div class="info-text">
            تصویر Sagittal T2 را بارگذاری کنید.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "تصویر MRI را انتخاب کنید",
    type=["png", "jpg", "jpeg"],
    label_visibility="visible",
)


# =========================================================
# Image Processing
# =========================================================

if uploaded_file is not None:

    try:

        image = Image.open(uploaded_file).convert("RGB")

        st.success("تصویر با موفقیت بارگذاری شد.")

        st.markdown(
            """
            <div class="info-card">
                <div class="info-title">
                    🖼️ تصویر ورودی
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # -------------------------------------------------
        # Responsive Image
        # -------------------------------------------------

        st.image(
            image,
            use_container_width=True,
        )

        # -------------------------------------------------
        # Basic image information
        # -------------------------------------------------

        width, height = image.size

        st.markdown(
            """
            <div class="info-card">
                <div class="info-title">
                    📊 اطلاعات تصویر
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("عرض", f"{width}px")

        with col2:
            st.metric("ارتفاع", f"{height}px")

        with col3:
            st.metric("نسبت تصویر", f"{width / height:.2f}")

        # =================================================
        # Image Analysis
        # =================================================

        st.markdown(
            """
            <div class="info-card">

                <div class="info-title">
                    🔬 تحلیل اولیه
                </div>

                <div class="info-text">
                    در این مرحله تصویر برای پردازش اولیه آماده شده است.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

        analyze = st.button(
            "🔎 شروع تحلیل تصویر",
            use_container_width=True,
        )

        if analyze:

            with st.spinner("در حال پردازش تصویر..."):

                img_array = np.array(image)

                gray = cv2.cvtColor(
                    img_array,
                    cv2.COLOR_RGB2GRAY
                )

                # Normalize
                normalized = cv2.normalize(
                    gray,
                    None,
                    0,
                    255,
                    cv2.NORM_MINMAX
                )

                # Simple edge detection
                edges = cv2.Canny(
                    normalized,
                    50,
                    150
                )

                edge_ratio = np.mean(edges > 0)

            st.success("پردازش اولیه با موفقیت انجام شد.")

            # ---------------------------------------------
            # Confidence
            # ---------------------------------------------

            confidence = min(
                99,
                max(
                    50,
                    int(70 + edge_ratio * 100)
                )
            )

            st.markdown(
                """
                <div class="info-card">

                    <div class="info-title">
                        🎯 Confidence
                    </div>

                    <div class="info-text">
                        میزان Confidence فعلی صرفاً مربوط به پردازش
                        تصویری Prototype است و اعتبار تشخیصی ندارد.
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

            st.metric(
                "Confidence اولیه",
                f"{confidence}%"
            )

            # ---------------------------------------------
            # Result
            # ---------------------------------------------

            st.markdown(
                """
                <div class="info-card">

                    <div class="info-title">
                        📍 Localization
                    </div>

                    <div class="info-text">
                        تشخیص دقیق L1 تا S1 در این نسخه هنوز الگوریتم
                        بالینی/AI معتبر ندارد.
                    </div>

                    <div class="info-text">
                        مرحله بعدی پروژه می‌تواند استفاده از
                        MONAI / PyTorch و مدل Segmentation/Detection
                        برای Localization واقعی مهره‌ها و دیسک‌ها باشد.
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

            # Show processed image
            processed = Image.fromarray(edges)

            st.image(
                processed,
                caption="پردازش اولیه لبه‌ها",
                use_container_width=True,
            )


else:

    st.info(
        "👆 ابتدا یک تصویر Sagittal T2 را انتخاب کنید."
    )


# =========================================================
# Footer
# =========================================================

st.markdown(
    """
    <div style="
        text-align:center;
        margin-top:35px;
        padding:15px;
        opacity:0.65;
        font-size:12px;
        line-height:1.8;
    ">
        Lumbar MRI Analyzer V3.2<br>
        Prototype / Research & Educational Use<br>
        محمد صفری
    </div>
    """,
    unsafe_allow_html=True,
)
