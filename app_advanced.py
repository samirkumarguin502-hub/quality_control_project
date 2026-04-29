import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO
from PIL import Image, ImageTk
import cv2

# Load model
model = YOLO("runs/detect/train-8/weights/best.pt")
names = model.names

DEFECT_EXPLANATION = {
    "mouse_bite": "Edges of PCB are damaged or eaten away.",
    "spur": "Unwanted copper projection causing short circuit risk.",
    "missing_hole": "Hole missing where component should be mounted.",
    "short": "Two conductive paths are unintentionally connected.",
    "open_circuit": "Broken connection interrupting current flow.",
    "spurious_copper": "Extra unwanted copper on PCB surface."
}

root = tk.Tk()
root.title("PCB Defect Detection System")
root.geometry("950x700")
root.configure(bg="#121212")

# -------- SCROLLABLE FRAME --------
canvas = tk.Canvas(root, bg="#121212", highlightthickness=0)
scrollbar = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
scroll_frame = tk.Frame(canvas, bg="#121212")

scroll_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)

canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

# -------- TITLE --------
title = tk.Label(scroll_frame,
                 text="PCB Defect Detection System",
                 font=("Segoe UI", 24, "bold"),
                 fg="white", bg="#121212")
title.pack(pady=20)

# -------- IMAGE CARD --------
image_card = tk.Frame(scroll_frame, bg="#1e1e1e", bd=2, relief="ridge")
image_card.pack(pady=10, padx=20, fill="x")

image_label = tk.Label(image_card, bg="#1e1e1e")
image_label.pack(pady=10)

# -------- RESULT CARD --------
result_card = tk.Frame(scroll_frame, bg="#1e1e1e", bd=2, relief="ridge")
result_card.pack(pady=10, padx=20, fill="x")

result_label = tk.Label(result_card,
                        text="Upload image to detect defects",
                        font=("Segoe UI", 12),
                        fg="white", bg="#1e1e1e",
                        justify="left", wraplength=850)
result_label.pack(padx=10, pady=10)

# -------- FUNCTION --------
def select_image():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return

    results = model.predict(source=file_path, conf=0.4)
    r = results[0]

    annotated = r.plot()

    img = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = img.resize((700, 350))

    img_tk = ImageTk.PhotoImage(img)
    image_label.configure(image=img_tk)
    image_label.image = img_tk

    # ----- RESULT -----
    details = ""

    if len(r.boxes) == 0:
        details = "🟢 PASS\nNo defect detected"
    else:
        details = "🔴 DEFECT DETECTED\n\n"

        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = names[cls_id]

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            reason = DEFECT_EXPLANATION.get(label, "Unknown issue")

            details += f"▪ Type: {label}\n"
            details += f"▪ Confidence: {conf:.2f}\n"
            details += f"▪ Location: ({x1},{y1}) → ({x2},{y2})\n"
            details += f"▪ Reason: {reason}\n\n"

    result_label.config(text=details)

# -------- BUTTON --------
btn = tk.Button(scroll_frame,
                text="Select PCB Image",
                command=select_image,
                font=("Segoe UI", 14),
                bg="#007acc", fg="white",
                padx=20, pady=10)
btn.pack(pady=20)

root.mainloop()