fastapi 백엔드

```bash
conda create -n project3_front python=3.11
```

```bash
conda activate project3_front
```

# faster rcnn을 위한 detectron2 설치
```bash
pip install -r requirements.txt

git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2
```

```bash
uvicorn main:app --host 0.0.0.0 --port 1234 --reload
```

streamlit 프론트 엔드
```bash
streamlit run app.py
```
# 결과 이미지
![front_image](front_image.png)