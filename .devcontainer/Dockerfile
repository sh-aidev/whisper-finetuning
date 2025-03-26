FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04 AS build

# setting non-interactive mode
ENV DEBIAN_FRONTEND=noninteractive

# setting timezone
RUN ln -fs /usr/share/zoneinfo/Asia/Kolkata /etc/localtime

# installing basic dependencies
RUN apt update && apt install --no-install-recommends -y \
    git \
    curl \
    build-essential \
    sudo \
    ssh \
    python3.12-dev \
    python3.12-venv \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# installing python3.10
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt update && \
    apt remove -y python3-blinker && \
    apt install -y python3-pip

# setting working directory to /workspace
WORKDIR /workspace

# copy requirements.txt
COPY requirements.txt .

# installing python packages
RUN python3 -m pip install --break-system-packages setuptools wheel packaging && \
    python3 -m pip install --break-system-packages -r requirements.txt && \
    python3 -m pip install --break-system-packages flash-attn