ARG CUDA_VERSION=12.8.0
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu24.04 AS base

ARG TARGETARCH
ARG PYTHON_VERSION=3.12
ARG UBUNTU_MIRROR=tsinghua

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    TZ=Asia/Shanghai \
    CUDA_HOME=/usr/local/cuda \
    # ✅ 方案1：自动接受 Conda TOS（官方推荐）
    CONDA_PLUGINS_AUTO_ACCEPT_TOS=yes

# 清华 Ubuntu 源
RUN if [ "$UBUNTU_MIRROR" = "tsinghua" ]; then \
    sed -i 's@//.*archive.ubuntu.com@//mirrors.tuna.tsinghua.edu.cn@g' /etc/apt/sources.list \
    && sed -i 's@//.*security.ubuntu.com@//mirrors.tuna.tsinghua.edu.cn@g' /etc/apt/sources.list; \
fi

# 基础依赖
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
    tzdata ca-certificates curl wget git build-essential \
    && rm -rf /var/lib/apt/lists/*

# 安装 Miniconda3
RUN if [ "$TARGETARCH" = "arm64" ]; then \
        MINICONDA_ARCH="aarch64"; \
    else \
        MINICONDA_ARCH="x86_64"; \
    fi \
    && wget -q https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-${MINICONDA_ARCH}.sh -O /miniconda.sh \
    && bash /miniconda.sh -b -p /opt/conda \
    && rm /miniconda.sh

ENV PATH="/opt/conda/bin:$PATH"

# ==============================================================================
FROM base AS framework
WORKDIR /workspace

ARG CUDA_VERSION
ARG PYTHON_VERSION

# 自动计算 CUDA 索引
RUN case "$CUDA_VERSION" in \
    12.8.*) CUINDEX=128 ;; \
    12.7.*) CUINDEX=127 ;; \
    12.6.*) CUINDEX=126 ;; \
    12.5.*) CUINDEX=125 ;; \
    *) CUINDEX=128 ;; \
    esac && echo "CUINDEX=$CUINDEX" >> /etc/environment
ENV CUINDEX=$CUINDEX

# ✅ 方案2（可选叠加）：显式接受 TOS（确保万无一失）
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
 && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# ✅ Conda 必需：创建环境（现在100%成功）
RUN conda create -n serve python=${PYTHON_VERSION} -y
ENV PATH="/opt/conda/envs/serve/bin:$PATH"

# Conda 清华源（可选但推荐）
RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main \
 && conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r \
 && conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge \
 && conda config --set show_channel_urls yes

# Pip 清华源
RUN pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# 安装 PyTorch
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu${CUINDEX}

# 复制项目
COPY . /workspace/ktransformers
WORKDIR /workspace/ktransformers

# 安装依赖
RUN pip install -e .
