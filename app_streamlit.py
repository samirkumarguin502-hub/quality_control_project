import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# -------- LOAD MODEL --------
model = YOLO("runs/detect/train-8/weights/best.pt")

DEFECT_EXPLANATION = {
    "mouse_bite": "Edges of PCB are damaged or eaten away.",
    "spur": "Unwanted copper projection causing short circuit risk.",
    "missing_hole": "Hole missing where component should be mounted.",
    "short": "Two conductive paths are unintentionally connected.",
    "open_circuit": "Broken connection interrupting current flow.",
    "spurious_copper": "Extra unwanted copper on PCB surface."
}

st.set_page_config(page_title="PCB Detection", layout="wide")

st.title("🔍 PCB Defect Detection System")

uploaded_file = st.file_uploader("Upload PCB Image", type=["jpg", "png", "jpeg"])

if uploaded_file:

    # -------- FIXED IMAGE LOAD --------
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # original for display
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

    st.subheader("📷 Original Image")
    st.image(img_rgb, width=700)

    # -------- PREDICTION --------
    results = model.predict(source=img_cv, conf=0.35, imgsz=640)
    r = results[0]

    annotated = r.plot()
    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    st.subheader("🧠 Detected Output")
    st.image(annotated, width=700)

    # -------- RESULT --------
    if len(r.boxes) == 0:
        st.success("✅ PASS - No defect detected")
    else:
        st.error("❌ DEFECT DETECTED")

        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            reason = DEFECT_EXPLANATION.get(label, "Unknown issue")

            st.markdown(f"""
            ### 🔴 {label}
            - Confidence: {conf:.2f}
            - Location: ({x1},{y1}) → ({x2},{y2})
            - Reason: {reason}
            """)