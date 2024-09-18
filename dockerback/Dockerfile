# Python 3.11 이미지를 기반으로 설정
FROM python:3.11

# 작업 디렉토리 설정
WORKDIR /app

# 필수 종속성 설치
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    gcc \
    g++ \
    cmake \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Detectron2 설치
RUN pip install -U torch torchvision torchaudio
RUN pip install 'git+https://github.com/facebookresearch/detectron2.git'

# 필요한 패키지 설치
COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# 애플리케이션 파일 복사
COPY . .

# Streamlit 애플리케이션 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "1234", "--reload"]