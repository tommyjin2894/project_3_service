import cv2
from ultralytics import YOLO
from ultralytics.models import YOLOv10

from loguru import logger
import numpy as np
import json
import cv2
import sys
from PIL import Image

import time

time_ = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
# 로거 생성
logger.add(f"result/yolo_v10_face/{time_}.log",format="{message}", level="INFO")
# face 모델 라벨
emotion_mapping = {0 : '분노', 1 : '슬픔', 2 : '공포', 3 : '기쁨'}

def yolo_v10_face_out(image: Image.Image):
    # 설정을 로드

    model_face_emmotion = YOLOv10('models/yolov10n-face.pt')
    results = model_face_emmotion(image)
    detections = results[0]

    detected_indices = detections.boxes.cls
    annotated_image = results[0].plot()
    # 이미지 저장
    cv2.imwrite(f"result/yolo_v10_face/{time_}.jpg", annotated_image)
    
    
    # 결과를 로그로 출력
    logger.info(f"Detected Labels: {int(detected_indices)}")
    logger.info(f"Detected Labels_str: {emotion_mapping[int(detected_indices)]}")
    logger.info(detections)
    return annotated_image, detected_indices

# 테스트
if __name__ == "__main__":
    test_image = cv2.imread("test.png")
    yolo_v10_face_out(test_image)