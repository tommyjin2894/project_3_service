# main.py
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from PIL import Image
from t5_model import summarize_text
from io import BytesIO
import base64
from py_model import yolo10n_face, yolo_oiv, faster_rcnn, gpt2, kogpt2, t5_base, t5_large
import time
from loguru import logger

app = FastAPI()

def image_to_base64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode('utf-8')

@app.post("/yolov10n")
async def yolov10n_endpoint(file: UploadFile = File(...),lm_opt: str = Form(...)):

    time_ = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    # 로거 생성
    logger.add(f"result/{time_}/yolo_v10_face_{lm_opt}.log",format="{message}", level="INFO")
    
    image_bytes = await file.read()
    image = Image.open(BytesIO(image_bytes))
        
    object_detection_image, pred_face = yolo10n_face.yolo_v10_face_out(image)
    emotion_detection_image, pred_obj = yolo_oiv.yolo_oiv_out(image)
    
    object_detection_base64 = image_to_base64(object_detection_image)
    emotion_detection_base64 = image_to_base64(emotion_detection_image)
    
    # 결과를 JSON으로 반환
    return JSONResponse(content={
        "object_detection": object_detection_base64,
        "emotion_detection": emotion_detection_base64,
    })

# yolo_oiv
# @app.post("/faster-rcnn/")
# faster_rcnn
# yolo_oiv

# @app.post("/gpt2/")
# gpt2
# @app.post("/kogpt2/")
# kogpt2
# @app.post("/t5_base/")
# t5_base
# @app.post("/t5_base/")
# t5_large

# FastAPI 앱 실행 방법
# uvicorn main:app --host 0.0.0.0 --port 1234 --reload

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# @app.post("/detect/")
# async def detect(file: UploadFile = File(...)):
#     image_bytes = await file.read()

#     image = Image.open(BytesIO(image_bytes))

#     # 객체 탐지 및 감정 탐지
#     object_detection_image, emotion_detection_image, t5out, gpt2out = detect_objects(image)

#     # 바이트 데이터를 Base64로 인코딩
#     object_detection_base64 = image_to_base64(object_detection_image)
#     emotion_detection_base64 = image_to_base64(emotion_detection_image)

#     # 결과를 JSON으로 반환
#     return JSONResponse(content={
#         "object_detection": object_detection_base64,
#         "emotion_detection": emotion_detection_base64,
#         "gpt2": t5out,
#         "t5": gpt2out
#     })