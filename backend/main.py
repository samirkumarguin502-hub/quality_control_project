from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2

from model import detect
from auth import login

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/login")
def user_login(data: dict):
    return login(data["username"], data["password"])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)

    result = detect(img)

    return {"detections": result}