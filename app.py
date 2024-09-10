# app.py
import streamlit as st
import requests
from io import BytesIO
from PIL import Image
import base64
from loguru import logger
import time

st.sidebar.title("AI 모델을 활용한 객체 탐지 및 감정 분석")

# YOLO 섹션
st.sidebar.header("무플 방지 위원회")

st.sidebar.header("옵션 설정")
uploaded_file = st.sidebar.file_uploader("이미지를 업로드하세요", type=["jpg", "jpeg", "png"])

object_detect_option= st.sidebar.selectbox("얼굴 표정 검출 모델을 선택하세요", ["yolov10n", "faster_rcnn"])
language_model_option = st.sidebar.selectbox("언어 모델을 선택하세요", ["t5_base", "t5_large","gpt2","kogpt2"])

object_detect_endpoint = f"http://127.0.0.1:1234/{object_detect_option}"

if uploaded_file is not None:
    
    time_ = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    logger.add(f"result_front/{time_}/{object_detect_option}_{language_model_option}.log", format="{message}", level="INFO")
    # -------------------------------------------------------------------------------------------- #
    # 바이트로 변환
    image_bytes = uploaded_file.read()
    
    # 입력 이미지 출력
    st.subheader("입력 이미지")
    st.image(
        uploaded_file,
        caption="입력 이미지",
        use_column_width=True,
        channels="RGB"  # 이미지가 RGB 채널을 사용하는 경우
    )
    
    # 파일 객체 요청
    files = {"file": ("image.png", image_bytes, "image/png")}
    data = {"lm_opt": language_model_option}
    
    # fast api 에다가 보내기
    response = requests.post(object_detect_endpoint, files=files, data=data)
    response_data = response.json()
    
    object_detection_base64 = response_data["object_detection_image"]
    emotion_detection_base64 = response_data["emotion_detection_image"]
    pred_label = response_data["pred_label"]
    
    # Base64를 이미지로 디코딩
    object_detection_image = Image.open(BytesIO(base64.b64decode(object_detection_base64)))
    emotion_detection_image = Image.open(BytesIO(base64.b64decode(emotion_detection_base64)))
    
    # 라벨 출력
    st.sidebar.write(f"감정 및 객체 감지: {pred_label}")

    st.sidebar.subheader("Emotion Detection Results")
    st.sidebar.image(
        object_detection_image,
        caption="Object Detection Result",
        use_column_width=True,
        channels="RGB"  # 이미지가 RGB 채널을 사용하는 경우
    )

    # Emotion Detection 결과 출력
    st.sidebar.subheader("Object Detection Results")
    st.sidebar.image(
        emotion_detection_image,
        caption="Emotion Detection Result",
        use_column_width=True,
        channels="RGB"  # 이미지가 RGB 채널을 사용하는 경우
    )
    
    # lm_out 출력
    st.subheader(f"{language_model_option} 의 댓글")
    if "출력값" in response_data["lm_out"]:
        out_text = response_data["lm_out"].split("출력값: ")[-1]
    else:
        out_text = response_data["lm_out"]
        
    st.markdown(f"<h2 style='font-size:32px;'>{out_text}</h2>", unsafe_allow_html=True)
    
    logger.info(f"model:{object_detect_option}, lm:{language_model_option}, result:{pred_label,out_text}")
    logger.remove()
    # -------------------------------------------------------------------------------------------- #
