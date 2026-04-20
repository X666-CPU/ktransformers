ARG CUDA_VERSION=12.8.0
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu24.04 AS base

ARG TARGETARCH
ARG DEEPEP_COMMIT=9af0e0d0e74f3577af1979c9b9e1ac2cad0104ee
ARG BUILD_AND_DOWNLOAD_PARALLEL=8
ARG SGL_KERNEL_VERSION=0.3.19
ARG FLASHINFER_VERSION=0.5.3

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    PATH="${PATH}:/usr/local/nvidia/bin" \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata ca-certificates curl wget git git-lfs \
    build-essential cmake patchelf libopenmpi-dev libnuma-dev \
    libibverbs-dev librdmacm1 libssl-dev libprotobuf-dev unzip \
    && rm -rf /var/lib/apt/lists/*

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 TZ=Asia/Shanghai

# 安装 Miniconda
RUN if [ "$TARGETARCH" = "arm64" ]; then \
    MINICONDA_ARCH="aarch64"; else MINICONDA_ARCH="x86_64"; fi \
    && mkdir -p /opt/miniconda3 \
    && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-${MINICONDA_ARCH}.sh -O /opt/miniconda3/miniconda.sh \
    && bash /opt/miniconda3/miniconda.sh -b -u -p /opt/miniconda3 \
    && rm /opt/miniconda3/miniconda.sh

ENV PATH="/opt/miniconda3/bin:${PATH}"

# 卸载 TOS 插件，永不报错
RUN conda remove -n base conda-anaconda-tos -y || true

# ============================================================================
FROM base AS framework
WORKDIR /workspace

ARG CUDA_VERSION
RUN case "$CUDA_VERSION" in \
    12.8.*) CUINDEX=128 ;; \
    12.7.*) CUINDEX=127 ;; \
    12.6.*) CUINDEX=126 ;; \
    12.5.*) CUINDEX=125 ;; \
    *) CUINDEX=128 ;; esac
ENV CUINDEX=$CUINDEX

# 创建环境
RUN conda create -n serve python=3.12 -y
ENV PATH="/opt/miniconda3/envs/serve/bin:$PATH"

# PIP 清华源
RUN pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# 安装基础包
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu${CUINDEX}

# 复制项目
COPY . /workspace/ktransformers
RUN cd /workspace/ktransformers && git submodule update --init --recursive || true
RUN ln -s /workspace/ktransformers/third_party/sglang /workspace/sglang

# 你要的依赖
RUN pip install torchviz fire typer uvicorn fastapi pydantic

# 项目依赖
RUN pip install sgl-kernel==${SGL_KERNEL_VERSION}
RUN cd /workspace/sglang && pip install -e "python[all]"
RUN python -m flashinfer --download-cubin

# DeepEP
RUN curl -L https://github.com/deepseek-ai/DeepEP/archive/${DEEPEP_COMMIT}.zip -o /tmp/deepep.zip \
    && unzip -q /tmp/deepep.zip -d /tmp \
    && mv /tmp/DeepEP-${DEEPEP_COMMIT} /workspace/DeepEP
RUN cd /workspace/DeepEP && pip install --no-build-isolation .

# NCCL & kt-kernel
RUN pip install nvidia-nccl-cu12==2.28.3 --force-reinstall --no-deps
RUN cd /workspace/ktransformers/kt-kernel && pip install -e .

# 启动
WORKDIR /workspace/ktransformers
CMD ["/bin/bash"]
