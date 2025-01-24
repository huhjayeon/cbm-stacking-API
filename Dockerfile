# 공식 Python 이미지 사용
FROM python:3.10

# 시스템 패키지 업데이트 및 필수 라이브러리 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    gcc \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# 프로젝트 파일 복사
COPY . /app

# 가상 환경 생성 및 패키지 설치
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 최신 pip, numpy, scipy 설치 (캐시 비활성화)
RUN pip install --no-cache-dir -U pip setuptools wheel
RUN pip install --no-cache-dir numpy==1.23.5 scipy==1.10.1 pandas==1.5.3

# 패키지 설치
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 실행
CMD ["python", "app.py"]
