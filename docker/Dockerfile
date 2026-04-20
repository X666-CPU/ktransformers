ARG CUDA_VERSION=12.8.0
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu24.04 AS base

ARG TARGETARCH
ARG GRACE_BLACKWELL=0
ARG HOPPER_SBO=0
ARG CPU_VARIANT=x86-intel-multi
ARG BUILD_ALL_CPU_VARIANTS=1

ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG http_proxy
ARG https_proxy
ENV HTTP_PROXY=${HTTP_PROXY} \
    HTTPS_PROXY=${HTTPS_PROXY} \
    http_proxy=${http_proxy} \
    https_proxy=${https_proxy}

ARG DEEPEP_COMMIT=9af0e0d0e74f3577af1979c9b9e1ac2cad0104ee
ARG BUILD_AND_DOWNLOAD_PARALLEL=8
ARG SGL_KERNEL_VERSION=0.3.19
ARG GDRCOPY_VERSION=2.5.1
ARG UBUNTU_MIRROR=tsinghua
ARG GITHUB_ARTIFACTORY=github.com
ARG FLASHINFER_VERSION=0.5.3

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    CUDA_VERSION=${CUDA_VERSION}

ENV PATH="${PATH}:/usr/local/nvidia/bin" \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"

# 清华源
RUN if [ -n "$UBUNTU_MIRROR" ]; then \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ noble main restricted universe multiverse" > /etc/apt/sources.list && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ noble-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ noble-backports main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb http://security.ubuntu.com/ubuntu/ noble-security main restricted universe multiverse" >> /etc/apt/sources.list && \
    rm -f /etc/apt/sources.list.d/ubuntu.sources 2>/dev/null || true; \
fi

# 依赖
RUN --mount=type=cache,target=/var/cache/apt,id=base-apt \
    apt-get update && apt-get install -y --no-install-recommends \
    tzdata ca-certificates curl wget git git-lfs \
    build-essential cmake patchelf vim htop zsh \
    libopenmpi-dev libnuma-dev numactl ffmpeg \
    libibverbs-dev librdmacm1 libssl-dev libprotobuf-dev unzip \
    && rm -rf /var/lib/apt/lists/*

# locale
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV TZ=Asia/Shanghai

# Miniconda 安装
RUN if [ "$TARGETARCH" = "arm64" ]; then \
        MINICONDA_ARCH="aarch64"; \
    else \
        MINICONDA_ARCH="x86_64"; \
    fi \
    && mkdir -p /opt/miniconda3 \
    && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-${MINICONDA_ARCH}.sh -O /opt/miniconda3/miniconda.sh \
    && bash /opt/miniconda3/miniconda.sh -b -u -p /opt/miniconda3 \
    && rm /opt/miniconda3/miniconda.sh

ENV PATH="/opt/miniconda3/bin:${PATH}"

# ==============================================
# ✅ 关键：直接卸载 TOS 插件，永久关闭！
# ==============================================
RUN conda remove -n base conda-anaconda-tos -y || true

# ==============================================================================
FROM base AS framework
WORKDIR /workspace

ARG CUDA_VERSION

# 自动计算 CUDA 版本
RUN case "$CUDA_VERSION" in \
    12.8.*) CUINDEX=128 ;; \
    12.7.*) CUINDEX=127 ;; \
    12.6.*) CUINDEX=126 ;; \
    12.5.*) CUINDEX=125 ;; \
    *) CUINDEX=128 ;; \
    esac && echo "CUINDEX=$CUINDEX" >> /etc/environment

ENV CUINDEX=${CUINDEX}

# Conda 创建环境（无任何报错）
RUN conda create -n serve python=3.12 -y
ENV PATH="/opt/miniconda3/envs/serve/bin:$PATH"

# pip 清华源
RUN pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# 安装 torch
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu${CUINDEX}

# 复制代码
COPY . /workspace/ktransformers
RUN cd /workspace/ktransformers && git submodule update --init --recursive || true
RUN ln -s /workspace/ktransformers/third_party/sglang /workspace/sglang

# 安装全部依赖
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel \
    torchviz fire typer uvicorn fastapi pydantic

# sgl-kernel
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install sgl-kernel==${SGL_KERNEL_VERSION}

# sglang
RUN --mount=type=cache,target=/root/.cache/pip \
    cd /workspace/sglang \
    && pip install -e "python[all]" --extra-index-url https://download.pytorch.org/whl/cu${CUINDEX}

# FlashInfer
RUN --mount=type=cache,target=/root/.cache/pip \
    FLASHINFER_CUBIN_DOWNLOAD_THREADS=${BUILD_AND_DOWNLOAD_PARALLEL} \
    python -m flashinfer --download-cubin

# DeepEP
RUN curl -L https://${GITHUB_ARTIFACTORY}/deepseek-ai/DeepEP/archive/${DEEPEP_COMMIT}.zip -o /tmp/deepep.zip \
    && unzip -q /tmp/deepep.zip -d /tmp \
    && mv /tmp/DeepEP-${DEEPEP_COMMIT} /workspace/DeepEP \
    && rm /tmp/deepep.zip

RUN --mount=type=cache,target=/root/.cache/pip \
    cd /workspace/DeepEP \
    && TORCH_CUDA_ARCH_LIST="9.0;10.0" MAX_JOBS=${BUILD_AND_DOWNLOAD_PARALLEL} \
    && pip install --no-build-isolation .

# NCCL
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install nvidia-nccl-cu12==2.28.3 --force-reinstall --no-deps

# kt-kernel
RUN cd /workspace/ktransformers/kt-kernel \
    && pip install -e .

# 自动激活环境
RUN conda init bash
RUN echo "conda activate serve" >> /root/.bashrc
RUN echo "cd /workspace/ktransformers" >> /root/.bashrc

WORKDIR /workspace/ktransformers
CMD ["/bin/bash"]
