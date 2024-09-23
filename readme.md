# SNS 사진 분석 댓글 및 피드백 프로젝트

### 🍿[메인 페이지 링크](https://github.com/crazy2894/project_3_git)🍿

### 도커 빌드
```
docker compose up --build
```

## 프론트 백엔드 파일 구조
```
project_3_service
│
├── dockerback
│      ├─ py_models
│      │   ├── __init__.py
│      │   ├── faster_rcnn.py
│      │   ├── gpt2.py
│      │   ├── kogpt2.py
│      │   ├── t5_base.py
│      │   ├── t5_large.py
│      │   ├── yolo_oiv.py
│      │   ├── yolo10n_face.py
│      ├── Dockerfile
│      ├── main.py
│      ├── requirements.txt
│
├── dockerfront
│      ├── app.py
│      ├── Dockerfile
│      ├── requirements.txt
│
├── .gitignore
├── docker-compose.yml
├── front_image.png
├── readme.md
├── requirements.txt
├── test.png

```



<details>
  <summary> 환경 설정 (도커 이용하지 않을시) </summary>

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
    conda install -r requirements.txt

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

</details>

# 결과 이미지
![front_image](front_image.png)