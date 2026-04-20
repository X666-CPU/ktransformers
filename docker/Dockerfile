ARG CUDA_VERSION=12.8.1
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu24.04 AS base

ARG TARGETARCH
ARG GRACE_BLACKWELL=0
ARG HOPPER_SBO=0
ARG CPU_VARIANT=x86-intel-multi
ARG BUILD_ALL_CPU_VARIANTS=1

# Proxy settings
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
ARG UBUNTU_MIRROR
ARG GITHUB_ARTIFACTORY=github.com
ARG FLASHINFER_VERSION=0.5.3

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda

ENV PATH="${PATH}:/usr/local/nvidia/bin" \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"

# 使用清华源
RUN if [ -n "$UBUNTU_MIRROR" ]; then \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ noble main restricted universe multiverse" > /etc/apt/sources.list && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ noble-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ noble-backports main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb http://security.ubuntu.com/ubuntu/ noble-security main restricted universe multiverse" >> /etc/apt/sources.list && \
    rm -f /etc/apt/sources.list.d/ubuntu.sources; \
fi

# 安装必需依赖
RUN --mount=type=cache,target=/var/cache/apt,id=base-apt \
    apt-get update && apt-get install -y --no-install-recommends \
    tzdata ca-certificates curl wget git git-lfs \
    build-essential cmake patchelf vim htop zsh \
    libopenmpi-dev libnuma-dev numactl ffmpeg \
    libibverbs-dev librdmacm1 libssl-dev libprotobuf-dev \
    && rm -rf /var/lib/apt/lists/*

#  locale
RUN locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8

# Miniconda
RUN mkdir -p /opt/miniconda3 \
    && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /opt/miniconda3/miniconda.sh \
    && bash /opt/miniconda3/miniconda.sh -b -u -p /opt/miniconda3 \
    && rm /opt/miniconda3/miniconda.sh
ENV PATH="/opt/miniconda3/bin:${PATH}"

# Conda 源
RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main \
    && conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free \
    && conda config --set show_channel_urls yes

# ==============================================================================
# 🔥 核心：彻底关闭 SFT，只保留推理 infer，彻底避开所有 404
# ==============================================================================
FROM base AS framework
ARG CUDA_VERSION
ARG BUILD_AND_DOWNLOAD_PARALLEL
ARG SGL_KERNEL_VERSION
ARG FLASHINFER_VERSION
ARG DEEPEP_COMMIT
ARG GITHUB_ARTIFACTORY
ARG FUNCTIONALITY=infer

WORKDIR /workspace

# 只创建 serve 环境，不创建任何微调环境
RUN conda create -n serve python=3.12 -y
RUN /opt/miniconda3/envs/serve/bin/pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# 克隆代码
RUN git clone --depth 1 https://${GITHUB_ARTIFACTORY}/kvcache-ai/ktransformers.git \
    && cd ktransformers \
    && git submodule update --init --recursive
RUN ln -s /workspace/ktransformers/third_party/sglang /workspace/sglang

# ==============================================================================
# 🔥 安装你需要的全部依赖
# ==============================================================================
RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/miniconda3/envs/serve/bin/pip install --upgrade pip setuptools wheel \
        torchviz fire typer uvicorn fastapi pydantic

# 安装 sgl-kernel
RUN --mount=type=cache,target=/root/.cache/pip \
    case "$CUDA_VERSION" in \
        12.8.1) CUINDEX=128 ;; \
        *) CUINDEX=128 ;; \
    esac \
    && /opt/miniconda3/envs/serve/bin/pip install sgl-kernel==${SGL_KERNEL_VERSION}

# 安装 sglang
RUN --mount=type=cache,target=/root/.cache/pip \
    cd /workspace/sglang \
    && /opt/miniconda3/envs/serve/bin/pip install -e "python[all]" --extra-index-url https://download.pytorch.org/whl/cu${CUINDEX}

# FlashInfer
RUN --mount=type=cache,target=/root/.cache/pip \
    FLASHINFER_CUBIN_DOWNLOAD_THREADS=${BUILD_AND_DOWNLOAD_PARALLEL} \
    /opt/miniconda3/envs/serve/bin/python -m flashinfer --download-cubin

# DeepEP
RUN curl -L https://${GITHUB_ARTIFACTORY}/deepseek-ai/DeepEP/archive/${DEEPEP_COMMIT}.zip -o /tmp/deepep.zip \
    && unzip -q /tmp/deepep.zip -d /tmp \
    && mv /tmp/DeepEP-${DEEPEP_COMMIT} /workspace/DeepEP \
    && rm /tmp/deepep.zip

RUN --mount=type=cache,target=/root/.cache/pip \
    cd /workspace/DeepEP \
    && TORCH_CUDA_ARCH_LIST="9.0;10.0" MAX_JOBS=${BUILD_AND_DOWNLOAD_PARALLEL} \
    /opt/miniconda3/envs/serve/bin/pip install --no-build-isolation .

# NCCL
RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/miniconda3/envs/serve/bin/pip install nvidia-nccl-cu12==2.28.3 --force-reinstall --no-deps

# kt-kernel
RUN cd /workspace/ktransformers/kt-kernel \
    && /opt/miniconda3/envs/serve/bin/python -m pip install -e .

# 收尾
RUN conda init bash
RUN echo 'alias serve="conda activate serve"' >> /root/.bashrc

WORKDIR /workspace
CMD ["/bin/bash"]
