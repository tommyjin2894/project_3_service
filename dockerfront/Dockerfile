# Python 3.11 이미지를 기반으로 설정
FROM python:3.11

# 작업 디렉토리 설정
WORKDIR /app

# 깃 설치
RUN apt-get update && apt-get install -y git


# 필요한 패키지 설치
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# 애플리케이션 파일 복사
COPY . .

# Streamlit 애플리케이션 실행
CMD ["streamlit", "run", "app.py", "--server.port=8561", "--server.address=0.0.0.0"]
