FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive

# 1. 安裝系統工具
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3.8-dev \
    python3-pip \
    python3.8-distutils \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.8 /usr/bin/python

# 2. 升級 pip
RUN python -m pip install --upgrade pip

# 3. 安裝 PyTorch (修正點：增加了 --default-timeout=1000)
# 這會讓 pip 等待 1000 秒才判定超時，足夠下載大檔案
RUN pip install --default-timeout=1000 --no-cache-dir torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116

# 4. 安裝一般依賴
COPY requirements.txt .
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

# 5. 安裝 Flash Attention
RUN pip install packaging
RUN pip install flash-attn==1.0.2 --no-build-isolation

# ▼ 只改這裡！把新東西加在最後面 ▼
RUN python -m pip install --upgrade pip
RUN pip install ema-pytorch==0.2.1
RUN pip install Augmentor==0.2.12
RUN pip install lmdb==1.7.5

CMD ["/bin/bash"]