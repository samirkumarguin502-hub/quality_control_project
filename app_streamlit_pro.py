import streamlit as st
import cv2
import pandas as pd
from PIL import Image
from ultralytics import YOLO
from io import BytesIO
import os

# ================= CONFIG =================
st.set_page_config(page_title="PCB AI System", layout="wide")

# ================= LOGIN =================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login():
    st.title("🔐 PCB AI Login")
    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")

    if st.button("Login"):
        if user == "admin" and pwd == "1234":
            st.session_state.logged_in = True
            st.success("Login successful")
        else:
            st.error("Invalid credentials")

if not st.session_state.logged_in:
    login()
    st.stop()

# ================= MODEL LOAD =================
MODEL_PATH = r"runs/detect/train-8/weights/best.pt"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model not found at: {MODEL_PATH}")
        return None
    return YOLO(MODEL_PATH)

model = load_model()

# ================= HEADER =================
st.title("🧠 PCB AI Defect Detection System")

# ================= STORAGE =================
if "history" not in st.session_state:
    st.session_state.history = []

# ================= PROCESS FUNCTION =================
def process_image(image):
    if model is None:
        return image, []

    results = model.predict(source=image, conf=0.25)
    r = results[0]

    annotated = r.plot()
    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    defects = []

    if r.boxes is not None:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()

            label = model.names[cls]

            defects.append({
                "type": label,
                "confidence": round(conf, 3),
                "x1": int(xyxy[0]),
                "y1": int(xyxy[1]),
                "x2": int(xyxy[2]),
                "y2": int(xyxy[3])
            })

            st.session_state.history.append({
                "type": label,
                "confidence": round(conf, 3)
            })

    return annotated, defects

# ================= TABS =================
tab1, tab2, tab3 = st.tabs(["📂 Upload", "🎥 Live Camera", "📊 Dashboard"])

# ================= TAB 1: UPLOAD =================
with tab1:
    st.subheader("Upload PCB Images")

    uploaded_files = st.file_uploader(
        "Upload images", type=["jpg", "png", "jpeg"], accept_multiple_files=True
    )

    if uploaded_files:
        for i, file in enumerate(uploaded_files):
            image = Image.open(file)

            st.image(image, caption="Original", use_container_width=True)

            annotated, defects = process_image(image)

            st.image(annotated, caption="Detected", use_container_width=True)

            if defects:
                st.error(f"❌ Defects Found: {len(defects)}")

                df = pd.DataFrame(defects)
                st.dataframe(df)

                # ===== Excel Download =====
                excel_buffer = BytesIO()
                df.to_excel(excel_buffer, index=False)

                st.download_button(
                    "📦 Download Excel",
                    excel_buffer.getvalue(),
                    file_name=f"report_{i}.xlsx",
                    key=f"excel_{i}"
                )

                # ===== PDF Download =====
                try:
                    from reportlab.platypus import SimpleDocTemplate, Paragraph
                    from reportlab.lib.pagesizes import letter

                    pdf_buffer = BytesIO()
                    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)

                    content = [Paragraph(str(df.to_string()), None)]
                    doc.build(content)

                    st.download_button(
                        "📄 Download PDF",
                        pdf_buffer.getvalue(),
                        file_name=f"report_{i}.pdf",
                        key=f"pdf_{i}"
                    )
                except:
                    st.warning("Install reportlab for PDF feature")

            else:
                st.success("✅ No defect detected")

# ================= TAB 2: LIVE CAMERA =================
with tab2:
    st.subheader("🎥 Live Camera Detection")

    run = st.checkbox("Start Camera")

    FRAME = st.image([])
    cam = cv2.VideoCapture(0)

    if run:
        for _ in range(500):
            ret, frame = cam.read()
            if not ret:
                st.error("Camera error")
                break

            annotated, _ = process_image(frame)
            FRAME.image(annotated)

    cam.release()

# ================= TAB 3: DASHBOARD =================
with tab3:
    st.subheader("📊 Defect Dashboard")

    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)

        st.dataframe(df)
        st.bar_chart(df["type"].value_counts())

        st.success(f"Total defects: {len(df)}")
    else:
        st.info("No data yet")