import cv2
from ultralytics.models import YOLOv10

from PIL import Image
from loguru import logger
import time

import json

jabels = json.load(open('models/oiv7_jabels.json', 'r'))
model_face_emmotion = YOLOv10('models/yolov8x-oiv7.pt')


def yolo_oiv_out(image: Image.Image):
    # 설정을 로드
    results = model_face_emmotion(image)
    detections = results[0]

    detected_indices = detections.boxes.cls
    annotated_image = results[0].plot()
    
    #---------------------------------------------------------------
    # 텍스트 입력 처리
    result_label = set([int(i) for i in detected_indices])
    result_label_str = [jabels[str(int(i))] for i in result_label]
    
    exception_lst = ['인간의 얼굴','의류','남자','여자','소년','소녀'] # 텍스트 입력 제외 목록

    text_intput_text = ''
    
    for i in result_label_str:
        if i not in exception_lst:
            text_intput_text +=i + ','
            
    text_intput_text = text_intput_text[:-1]
    #---------------------------------------------------------------
    
    # 결과를 로그로 출력
    logger.info(f"Detected Labels: {result_label}")
    logger.info(f"Detected Labels_str: {text_intput_text}")
    logger.info(detections)
    return annotated_image, text_intput_text

# 테스트
if __name__ == "__main__":
    time_ = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    
    # 로거 생성
    logger.add(f"result/yolo_v8_oiv/{time_}.log",format="{message}", level="INFO")
    
    test_image = cv2.imread("test.png")
    # 이미지 저장
    cv2.imwrite(f"result/yolo_v8_oiv/{time_}.jpg", yolo_oiv_out(test_image))