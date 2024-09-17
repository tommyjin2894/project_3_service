import cv2
from ultralytics.models import YOLOv10

from PIL import Image
from loguru import logger
import time

# face 모델 라벨
emotion_mapping = {0 : '분노', 1 : '슬픔', 2 : '공포', 3 : '기쁨'}
model_face_emmotion = YOLOv10('models/yolov10n-face.pt')

def yolo_v10_face_out(image: Image.Image):
    # 설정을 로드
    results = model_face_emmotion(image)
    detections = results[0]

    detected_indices = detections.boxes.cls
    annotated_image = results[0].plot()
    
    label_cls = [emotion_mapping[int(i)] for i in detected_indices]
    
    # 결과를 로그로 출력
    logger.info(f"Detected Labels: {label_cls}")
    logger.info(f"Detected Labels_str: {label_cls}")
    logger.info(detections)
    return annotated_image, label_cls

# 테스트
if __name__ == "__main__":
    
    time_ = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        # 이미지 저장
    # 로거 생성
    logger.add(f"result/yolo_v10_face/{time_}.log",format="{message}", level="INFO")
    test_image = cv2.imread("test.png")
    cv2.imwrite(f"result/yolo_v10_face/{time_}.jpg", yolo_v10_face_out(test_image)[0])