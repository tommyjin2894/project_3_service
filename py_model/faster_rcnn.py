from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog

from loguru import logger
import numpy as np
import json
import cv2
import sys

import time

time_ = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
# 로거 생성
logger.add(f"result/rcnn/logs/{time_}.log")

def rcnn_out(image):
    # 설정을 로드
    cfg = get_cfg()
    cfg.merge_from_file("detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.WEIGHTS = "models/faster_rcnn_r_50_final_model.pth"  # 훈련된 모델의 가중치 파일 경로
    cfg.DATASETS.TEST = ("face_data_set_valid", )  # 테스트 데이터셋 이름
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4

    # 예측기를 설정
    predictor = DefaultPredictor(cfg)

    # 예측을 수행
    outputs = predictor(image)

    # 예측 결과를 로그로 출력
    logger.info(outputs)

    # 결과 시각화
    v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TEST[0]), scale=1.2)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    result_image = v.get_image()[:, :, ::-1]

    # 결과 이미지를 저장
    cv2.imwrite(f"result/rcnn/{time_}.jpg", result_image)

    # 결과 이미지 저장 시간 기다리기
    time.sleep(1)
    return result_image, outputs

# 테스트
if __name__ == "__main__":
    test_image = cv2.imread("test.png")