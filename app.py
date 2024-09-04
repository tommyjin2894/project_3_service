# app.py
import streamlit as st
import requests
from io import BytesIO
from PIL import Image
import base64

# FastAPI 엔드포인트 정의
YOLO_ENDPOINT = "http://127.0.0.1:8000/detect/"
T5_ENDPOINT = "http://127.0.0.1:8000/summarize/"

st.title("YOLO 및 T5 웹 서비스")

# YOLO 섹션
st.header("YOLO를 사용한 객체 탐지")
uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # 바이트로 변환
    image_bytes = uploaded_file.read()

    # 파일 객체 요청
    files = {"file": ("image.png", image_bytes, "image/png")}

    # fast api 에다가 보내기
    response = requests.post(YOLO_ENDPOINT, files=files)

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
        st.write(result['gpt2'])
        st.write(result['t5'])

    else:
        st.error("이미지 처리 중 오류가 발생했습니다.")


# T5 섹션
st.header("T5를 사용한 텍스트 요약")
text_input = st.text_area("요약할 텍스트를 입력하세요")
if st.button("요약하기"):
    response = requests.post(T5_ENDPOINT, json={"text": text_input})
    # st.write(response.json())
    st.write('hello')
