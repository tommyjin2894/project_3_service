# SNS 사진 분석 댓글 및 피드백 프로젝트

### 🍿[메인 페이지](https://github.com/crazy2894/project_3_git)🍿
### 개요 :
이 프로젝트는 SNS에 게시된 사진을 분석하고, 자동으로 적절한 댓글과 피드백을 생성하는 시스템을 개발하는 것을 목표로 합니다. 이를 위해 다양한 딥러닝 모델을 사용하여 이미지 내 객체를 탐지하고, 감정을 분류한 후, 텍스트 생성 모델을 활용해 자연스러운 댓글을 생성합니다.

### 프로젝트 설명
#### 모델 선택 :

- 객체 탐지: YOLOv10n과 Faster R-CNN을 사용하여 얼굴 감정 분류를 수행합니다. YOLO는 빠른 추론 시간과 높은 정확도로 실시간 감정 분류가 가능하며, RCNN은 전이 학습을 통해 감정 분류에 높은 성능을 보여줍니다.

- 언어 모델: T5와 GPT-2 기반의 모델을 활용하여 SNS 사진에 대한 적절한 댓글을 생성합니다. T5 모델은 비교적 가벼우면서도 높은 성능을 자랑하고, GPT-2 모델은 자연스러운 텍스트 생성 능력이 뛰어납니다.

#### 데이터 및 학습 평가 기준:

- 이미지 데이터: wassup 안면 데이터 셋을 사용하여 기본 json 형식을 YOLO 형식으로 변환 하여 yolov10n 훈련에 이용하였고 yolo 의 어노테이션을 COCO 형식으로 변환해 faster rcnn 모델 학습에 활용 하였습니다.

- 텍스트 데이터: 객체 탐지 결과를 바탕으로 생성된 라벨을 사용하여 텍스트 데이터를 생성하고, 이를 기반으로 학습합니다.

#### 학습 평가 기준:

- 모델 학습 및 평가: 객체 탐지 모델은 다양한 성능 지표(AP50, AP50-95)를 사용하여 평가되며, YOLOv10n이 RCNN보다 우수한 성능을 보였습니다. 텍스트 생성 모델은 BLEU, METEOR, ROUGE와 같은 언어 모델 성능 지표를 사용해 평가합니다.

#### 프로젝트의 가치
- 효율성 향상: SNS에서 사용자의 반응을 자동으로 생성함으로써 시간과 비용을 절감할 수 있습니다.
- 정확한 감정 분석: 고도화된 감정 분류 모델을 통해 사용자의 감정 상태를 정확히 파악하고, 이를 바탕으로 맞춤형 피드백을 제공합니다.
- 확장 가능성: 다양한 도메인에 적용 가능하며, 새로운 데이터셋과 요구사항에 맞게 쉽게 조정할 수 있습니다.
- 기존의 멀티 모달(llava 등) 보다 훨씬 적은 memory가 요구되어 효율성에 초점을 맞췄습니다<br>
  ```
  모델 크기 : gpt2- 500mb 미만
  t5 - 1gb 수준
  yolov5n -5mb 수준
  r-cnn - 400mb 수준
  ```
  ([llava 모델 크기](gbhttps://huggingface.co/llava-hf/llava-1.5-7b-hf/tree/main) : 약 15 gb , 훈련 시 약 30gb 이상의 memory를 요구한다.)

#### 주요 성과
**YOLOv10n은 mAP50 89.81%**, **RCNN은 87.23%**, **YOLOv10n**이 더 높은 성능을 보였습니다.
언어 모델 평가에서 **BLEU-N, METEOR, ROUGE-N 등의 지표를 사용하여 성능을 검증**하였으며,
사람이 판단 하기에도 특히 t5 에서 텍스트 생성 결과가 상당히 만족스러운 수준임을 확인했습니다.

### 결론
이 프로젝트는 **SNS 사진에 대한 분석 및 피드백 생성의 자동화를 목표**로 하며, 높은 정확도와 빠른 처리 속도로 실용적이고 효과적인 결과를 목표로 하였습니다. YOLOv10n과 같은 최신 객체 탐지 모델의 활용으로 빠른 응답성을 보장하고, 트랜스 포머 기반 언어 모델을 통한 자연스러운 댓글 생성으로 사용자 경험을 향상시킬 수 있습니다.


## 프론트 백엔드 파일 구조
```
├── detectron2
├── models
├── py_model
│   ├── __init__.py
│   ├── faster_rcnn.py              # faster r-cnn 얼굴 검출
│   ├── gpt2.py                     # gpt2 언어모델
│   ├── kogpt2.py                   # kogpt2 언어모델
│   ├── t5_base.py                  # t5 언어 모델 base
│   ├── t5_large.py                 # t5 언어 모델 large
│   ├── yolo_oiv.py                 # Yolo 객체 검출
│   ├── yolo10n_face.py             # Yolo 얼굴 검출
├── test_pics
│   ├── .gitignore
│   ├── app.py                      # streamlit
│   ├── main.py                     # fast api
│   ├── readme.md
│   ├── requirements.txt
│   ├── test.png

```

## 환경 설정

```bash
conda create -n project3_front python=3.11
```

```bash
conda activate project3_front
```

## 라이브러리 설치
- window 환경
    ```
    install -r requirements.txt

    # 1. 방법
    git clone https://github.com/facebookresearch/detectron2.git
    python -m pip install -e detectron2 

    # 또는
    # 2. 에러 발생 시
    pip install -r requirements.txt
    git clone https://github.com/facebookresearch/detectron2.git
    cd detectron 
    python -m pip install -e . --use-pep517

    # 3. numpy error 발생시
    conda install numpy=1.24.3
    ```

    ```bash
    # detectron 폴더 안
    pyproject.toml 파일을 setup.py가 있는 경로에 만들고 다음 내용 추가
    [build-system]
    requires = ["setuptools>=64", "wheel", "torch", "torchvision"]
    build-backend = "setuptools.build_meta"
    ```

- linux 환경
    ```bash
    pip install -r requirements.txt

    git clone https://github.com/facebookresearch/detectron2.git
    python -m pip install -e detectron2
    ```

## 실행 하기
- fast api 백엔드
```bash
uvicorn main:app --host 0.0.0.0 --port 1234 --reload
```

- streamlit 프론트 엔드
```bash
streamlit run app.py
```
# 결과 이미지
![front_image](front_image.png)