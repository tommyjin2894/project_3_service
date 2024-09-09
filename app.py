# app.py
import streamlit as st
import requests
from io import BytesIO
from PIL import Image
import base64

st.title("YOLO 및 T5 웹 서비스")

# YOLO 섹션
st.header("YOLO를 사용한 객체 탐지")
uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "jpeg", "png"])

object_detect_option= st.selectbox("얼굴 표정 검출 모델을 선택하세요", ["yolov10n", "faster_rcnn"])
language_model_option = st.selectbox("언어 모델을 선택하세요", ["gpt2","kogpt2","t5_base", "t5_large"])

object_detect_endpoint = f"http://127.0.0.1:1234/{object_detect_option}/"

if uploaded_file is not None:
    # 바이트로 변환
    image_bytes = uploaded_file.read()

    # 파일 객체 요청
    files = {"file": ("image.png", image_bytes, "image/png")}
    data = {"language_model_option": language_model_option}
    
    # fast api 에다가 보내기
    response = requests.post(object_detect_endpoint, files=files, data=data)

    if response.status_code == 200:
        result = response.json()
        
        # Base64 문자열을 디코딩 후, BytesIO를 사용하여 PIL 이미지로 변환
        object_detection_base64 = result['object_detection']
        emotion_detection_base64 = result['emotion_detection']
        
        # 디코딩 및 이미지 변환
        object_detection_image = Image.open(BytesIO(base64.b64decode(object_detection_base64)))
        emotion_detection_image = Image.open(BytesIO(base64.b64decode(emotion_detection_base64)))
        
        # Streamlit에서 이미지 표시
        st.image(object_detection_image, caption="Object Detection")
        st.image(emotion_detection_image, caption="Emotion Detection")
        # st.write(result['gpt2'])
        # st.write(result['t5'])

    else:
        st.error("이미지 처리 중 오류가 발생했습니다.")
