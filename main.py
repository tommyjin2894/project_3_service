# main.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from yolo_model import detect_objects
from t5_model import summarize_text
from io import BytesIO
import base64

app = FastAPI()

def image_to_base64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode('utf-8')

@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    image_bytes = await file.read()

    image = Image.open(BytesIO(image_bytes))

    # 객체 탐지 및 감정 탐지
    object_detection_image, emotion_detection_image = detect_objects(image)

    # 바이트 데이터를 Base64로 인코딩
    object_detection_base64 = image_to_base64(object_detection_image)
    emotion_detection_base64 = image_to_base64(emotion_detection_image)

    # 결과를 JSON으로 반환
    return JSONResponse(content={
        "object_detection": object_detection_base64,
        "emotion_detection": emotion_detection_base64
    })

@app.post("/summarize/")
async def summarize(text: str):
    summary = summarize_text(text)
    return {"summary": summary}

# FastAPI 앱 실행 방법
# uvicorn main:app --host 0.0.0.0 --port 8000 --reload
