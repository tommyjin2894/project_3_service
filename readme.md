fastapi 백엔드

```bash
conda create -n project3_front python=3.11
```

```bash
conda activate project3_front
```

# faster rcnn을 위한 detectron2 설치
- window 환경
    ```bash
    pyproject.toml 파일을 setup.py가 있는 경로에 만들고 다음 내용 추가
    [build-system]
    requires = ["setuptools>=64", "wheel", "torch", "torchvision"]
    build-backend = "setuptools.build_meta"
    ```

    ```
    install -r requirements.txt

    # 1.
    git clone https://github.com/facebookresearch/detectron2.git
    python -m pip install -e detectron2 

    # 또는
    # 2. 에러 발생 시
    pip install -r requirements.txt
    git clone https://github.com/facebookresearch/detectron2.git
    cd detectron 
    python -m pip install -e . --use-pep517

    ```

- linux 환경
    ```bash
    pip install -r requirements.txt

    git clone https://github.com/facebookresearch/detectron2.git
    python -m pip install -e detectron2
    ```


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