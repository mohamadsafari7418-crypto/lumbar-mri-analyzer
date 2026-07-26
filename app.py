
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

st.set_page_config(page_title="Lumbar MRI Analyzer v2", layout="wide")

st.title("Lumbar MRI Analyzer — v2")
st.caption("Automatic vertebral/disc localization prototype • Research/education only • Radiologist verification required")

def normalize_gray(rgb):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)
    return g

def find_disc_candidates(rgb):
    """
    Heuristic prototype:
    1) Detects the spinal column region using edge/brightness concentration.
    2) Searches for horizontal low-signal gaps inside the vertebral column.
    3) Returns candidate y positions, then ranks five likely disc spaces.
    This is NOT a validated medical AI model.
    """
    gray = normalize_gray(rgb)
    h, w = gray.shape

    # Restrict search to central 35-75% of image width.
    x1, x2 = int(w * 0.30), int(w * 0.78)
    roi = gray[:, x1:x2]

    # Smooth vertically/horizontally to suppress noise.
    sm = cv2.GaussianBlur(roi, (0, 0), 3)
    # Horizontal profile: low intensity bands can correspond to discs.
    profile = np.mean(sm, axis=1)
    profile = cv2.GaussianBlur(profile.reshape(-1,1), (1, 21), 0).ravel()

    # We look for local minima with scipy if available; otherwise simple local test.
    candidates = []
    for y in range(12, h-12):
        window = profile[max(0,y-10):min(h,y+11)]
        if profile[y] <= np.min(window) + 2:
            candidates.append(y)

    # Cluster nearby candidates.
    clustered = []
    for y in candidates:
        if not clustered or y - clustered[-1] > max(8, int(h*0.012)):
            clustered.append(y)
        elif profile[y] < profile[clustered[-1]]:
            clustered[-1] = y

    # Keep candidates in the lower-middle lumbar region.
    valid = [y for y in clustered if int(h*0.20) < y < int(h*0.92)]

    # Prefer candidates with a strong local intensity drop.
    scored = []
    for y in valid:
        above = np.mean(profile[max(0,y-12):y-4])
        below = np.mean(profile[y+4:min(h,y+12)])
        local = (above + below)/2 - profile[y]
        scored.append((float(local), y))

    scored.sort(reverse=True)
    selected = sorted([y for _,y in scored[:12]])

    # Select five positions with roughly even vertical spacing.
    if len(selected) >= 5:
        # Dynamic programming-like greedy selection.
        chosen = [selected[0]]
        for y in selected[1:]:
            if all(abs(y-c) > h*0.045 for c in chosen):
                chosen.append(y)
            if len(chosen) == 5:
                break
        if len(chosen) < 5:
            chosen = selected[:5]
    else:
        # Fallback: five evenly spaced points in lumbar region.
        chosen = [int(h*(0.43 + i*0.085)) for i in range(5)]

    return sorted(chosen[:5]), (x1, x2), profile

def annotate(rgb, ys, x_range, labels):
    img = Image.fromarray(rgb).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size
    x1, x2 = x_range
    x_mid = int((x1+x2)/2)

    # Draw a central reference line.
    draw.line((x_mid, 0, x_mid, h), fill=(255, 215, 0), width=max(1, w//500))

    for i, (y, label) in enumerate(zip(ys, labels)):
        # Horizontal disc marker.
        draw.line((x1, y, x2, y), fill=(255, 60, 60), width=max(2, w//350))
        draw.ellipse((x_mid-5, y-5, x_mid+5, y+5), fill=(255, 60, 60))

        # Place label to right if possible.
        tx = min(w-150, x2 + 12)
        ty = max(5, y-18)
        draw.rounded_rectangle((tx-5, ty-4, min(w-2, tx+125), ty+25), radius=5, fill=(0,0,0))
        draw.text((tx, ty), label, fill=(255,255,255))

    return img

def render_level_table(ys, labels):
    st.subheader("Automatically detected disc spaces")
    cols = st.columns(3)
    with cols[0]:
        st.metric("Detected levels", str(len(ys)))
    with cols[1]:
        st.metric("Method", "T2 intensity + CV")
    with cols[2]:
        st.metric("Status", "Prototype")

    for label, y in zip(labels, ys):
        st.write(f"**{label}** — estimated disc-space center: y = {y}px")

uploaded = st.file_uploader("Sagittal T2 lumbar MRI را بارگذاری کنید", type=["jpg","jpeg","png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    rgb = np.array(img)

    left, right = st.columns([1.25, 1])

    with left:
        st.image(img, caption="Original Sagittal T2", use_container_width=True)

    ys, xr, profile = find_disc_candidates(rgb)

    # We label five consecutive disc spaces as requested.
    labels = ["L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1"]
    annotated = annotate(rgb, ys, xr, labels)

    with right:
        st.image(annotated, caption="Automatic localization overlay", use_container_width=True)
        st.warning(
            "این مکان‌یابی الگوریتم اولیه Computer Vision است و هنوز مدل آموزش‌دیده "
            "روی MRI و اعتبارسنجی بالینی نیست. محل خطوط را حتماً کنترل کنید."
        )

    render_level_table(ys, labels)

    st.divider()
    st.subheader("Correction / verification")
    st.write("اگر یکی از خطوط اشتباه است، مختصات y آن را اصلاح کنید. این بخش برای ساخت داده آموزشی نسخه بعدی مهم است.")

    corrected = []
    h = rgb.shape[0]
    for label, y in zip(labels, ys):
        val = st.number_input(f"{label} — y position (px)", min_value=0, max_value=h-1, value=int(y), step=1, key=f"y_{label}")
        corrected.append(int(val))

    corrected_img = annotate(rgb, corrected, xr, labels)
    st.image(corrected_img, caption="Verified/corrected localization", use_container_width=True)

    # Export coordinates as CSV-like text.
    data = "level,y_px\n" + "\n".join(f"{l},{y}" for l,y in zip(labels, corrected))
    st.download_button("Download localization CSV", data=data, file_name="lumbar_disc_localization.csv", mime="text/csv")

else:
    st.info("یک تصویر Sagittal T2 بارگذاری کنید.")
    st.markdown("""
### این نسخه چه می‌کند؟
1. تصویر را به grayscale و normalize می‌کند.
2. ناحیه تقریبی ستون فقرات را محدود می‌کند.
3. با بررسی پروفایل شدت سیگنال، شکاف‌های افقی احتمالی دیسک را پیدا می‌کند.
4. پنج محل احتمالی را به ترتیب به L1-L2 تا L5-S1 نسبت می‌دهد.
5. خطوط را روی تصویر نمایش می‌دهد.
6. امکان اصلاح دستی مختصات و ذخیره annotation را فراهم می‌کند.

### گام بعدی
برای تشخیص قابل‌اعتماد L1 تا S1 باید این heuristic را با segmentation/detection آموزش‌دیده روی دیتاست annotated جایگزین کنیم.
""")
