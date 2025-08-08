FROM nvidia/cuda:12.9.1-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    ffmpeg \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121

RUN sed '/torch==/d' requirements.txt > requirements_filtered.txt && \
    pip3 install --no-cache-dir -r requirements_filtered.txt

COPY src/ ./src/
COPY configs/ ./configs/

RUN mkdir -p /app/logs /app/outputs

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]