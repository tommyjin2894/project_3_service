import torch
import numpy as np
import cv2
import io
import json
from PIL import Image

from ultralytics import YOLO
from ultralytics.models import YOLOv10
from transformers import T5TokenizerFast, T5ForConditionalGeneration, GPT2Tokenizer, GPT2LMHeadModel

# 테스트
if __name__ == "__main__":
    pass