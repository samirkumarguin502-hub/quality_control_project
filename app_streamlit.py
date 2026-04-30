import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
from collections import Counter

# ---------------- MODEL ----------------
model = YOLO("runs/detect/train-8/weights/best.pt")

# ---------------- SMART EXPLANATION ----------------
DEFECT_EXPLANATION = {
    "mouse_bite": "PCB edge erosion → manufacturing issue",
    "spur": "Unwanted copper → may cause short circuit",
    "missing_hole": "Component mounting hole missing",
    "short": "Two tracks connected → circuit failure",
    "open_circuit": "Broken track → no current flow",
    "spurious_copper": "Extra copper → unwanted conductivity"
}

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="PCB AI System", layout="wide")

# ---------------- STYLE ----------------
st.markdown("""
<style>
.main-title {font-size:40px; font-weight:bold;}
.card {
    background-color:#1e1e1e;
    padding:20px;
    border-radius:12px;
    margin-bottom:20px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🔍 PCB AI Defect Detection System</p>', unsafe_allow_html=True)

# ---------------- HELPER FUNCTION ----------------
def detect_image(img_cv):
    results = model.predict(source=img_cv, conf=0.4, imgsz=640)
    r = results[0]

    annotated = r.plot()
    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    defects = []
    for box in r.boxes:
        conf = float(box.conf[0])
        if conf < 0.4:
            continue

        cls_id = int(box.cls[0])
        label = model.names[cls_id]

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        reason = DEFECT_EXPLANATION.get(label, "Unknown issue")

        defects.append({
            "label": label,
            "conf": conf,
            "bbox": (x1, y1, x2, y2),
            "reason": reason
        })

    return annotated, defects

# ---------------- TABS ----------------
tab1, tab2, tab3 = st.tabs(["📁 Upload Images", "🎥 Live Camera", "📊 Dashboard"])

# =====================================================
# 📁 MULTIPLE IMAGE UPLOAD
# =====================================================
with tab1:
    files = st.file_uploader("Upload PCB Images", accept_multiple_files=True, type=["jpg","png"])

    if files:
        all_defects = []

        for file in files:
            file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
            img_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

            annotated, defects = detect_image(img_cv)

            st.markdown('<div class="card">', unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                st.image(img_rgb, caption="Original", width=400)

            with col2:
                st.image(annotated, caption="Detected", width=400)

            if len(defects) == 0:
                st.success("✅ PASS")
            else:
                st.error("❌ DEFECT DETECTED")

                for d in defects:
                    st.markdown(f"""
                    🔴 **{d['label']}**  
                    Confidence: {d['conf']:.2f}  
                    Location: {d['bbox']}  
                    Reason: {d['reason']}
                    """)

            st.markdown('</div>', unsafe_allow_html=True)

            all_defects.extend([d["label"] for d in defects])

# =====================================================
# 🎥 LIVE CAMERA
# =====================================================
with tab2:
    st.subheader("Live Webcam Detection")

    run = st.checkbox("Start Camera")

    FRAME_WINDOW = st.image([])

    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            break

        annotated, defects = detect_image(frame)

        FRAME_WINDOW.image(annotated)

    cap.release()

# =====================================================
# 📊 DASHBOARD
# =====================================================
with tab3:
    st.subheader("Defect Analytics")

    st.info("Upload images first to generate stats")

    if 'all_defects' in locals() and len(all_defects) > 0:

        count = Counter(all_defects)

        st.write("### Defect Count")
        st.write(count)

        # simple chart
        st.bar_chart(count)

    else:
        st.warning("No data available yet")