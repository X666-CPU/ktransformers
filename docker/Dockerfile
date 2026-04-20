ARG CUDA_VERSION=12.8.1
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu24.04 AS base
ARG TARGETARCH

ARG HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
ENV HTTP_PROXY=${HTTP_PROXY} HTTPS_PROXY=${HTTPS_PROXY} http_proxy=${http_proxy} https_proxy=${https_proxy}

ARG DEEPEP_COMMIT=9af0e0d0e74f3577af1979c9b9e1ac2cad0104ee
ARG BUILD_AND_DOWNLOAD_PARALLEL=4
ARG SGL_KERNEL_VERSION=0.3.19
ARG KTRANSFORMERS_VERSION=0.5.3
ARG UBUNTU_MIRROR
ARG GITHUB_ARTIFACTORY=github.com

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda

ENV PATH="${PATH}:/usr/local/nvidia/bin" \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"

# ==============================
# 【极致精简】只装必须依赖
# ==============================
RUN --mount=type=cache,target=/var/cache/apt,id=base-apt \
    apt-get update && apt-get install -y --no-install-recommends \
    tzdata ca-certificates \
    git git-lfs \
    build-essential cmake patchelf \
    libnuma1 libnuma-dev numactl \
    libssl-dev libprotobuf-dev pybind11-dev \
    ffmpeg libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# 最小化 locale
RUN locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8

# 安装 miniconda
RUN mkdir -p /opt/miniconda3 \
    && wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /opt/miniconda3/miniconda.sh \
    && bash /opt/miniconda3/miniconda.sh -b -u -p /opt/miniconda3 \
    && rm /opt/miniconda3/miniconda.sh

ENV PATH="/opt/miniconda3/bin:$PATH"

# ==============================
# 只创建推理环境，不装微调
# ==============================
FROM base AS framework
ARG FUNCTIONALITY=infer
WORKDIR /workspace

RUN conda create -n serve python=3.12 -y
RUN /opt/miniconda3/envs/serve/bin/pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# 克隆代码
RUN git clone --depth 1 --branch v${KTRANSFORMERS_VERSION} https://${GITHUB_ARTIFACTORY}/kvcache-ai/ktransformers.git /workspace/ktransformers \
    && cd /workspace/ktransformers && git submodule update --init --recursive

# 安装基础依赖
RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/miniconda3/envs/serve/bin/pip install --upgrade pip setuptools wheel

# 安装 sgl-kernel
RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/miniconda3/envs/serve/bin/pip install sgl-kernel==${SGL_KERNEL_VERSION}

# 安装 sglang
RUN --mount=type=cache,target=/root/.cache/pip \
    cd /workspace/ktransformers/third_party/sglang \
    && /opt/miniconda3/envs/serve/bin/pip install -e "python[all]" --extra-index-url https://download.pytorch.org/whl/cu128

# FlashInfer
RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/miniconda3/envs/serve/bin/python -m flashinfer --download-cubin

# DeepEP
RUN curl -fsSL -o /tmp/deepep.zip https://${GITHUB_ARTIFACTORY}/deepseek-ai/DeepEP/archive/${DEEPEP_COMMIT}.zip \
    && unzip -q /tmp/deepep.zip -d /tmp && mv /tmp/DeepEP-${DEEPEP_COMMIT} /workspace/DeepEP
RUN --mount=type=cache,target=/root/.cache/pip \
    cd /workspace/DeepEP && \
    . /opt/miniconda3/etc/profile.d/conda.sh && conda activate serve && \
    TORCH_CUDA_ARCH_LIST="8.6" MAX_JOBS=4 pip install --no-build-isolation .

# NCCL（PCIe 多卡必须）
RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/miniconda3/envs/serve/bin/pip install nvidia-nccl-cu12==2.28.3 --force-reinstall --no-deps

# 编译 kt-kernel（仅30系）
RUN . /opt/miniconda3/etc/profile.d/conda.sh && conda activate serve \
    && cd /workspace/ktransformers/kt-kernel \
    && TORCH_CUDA_ARCH_LIST="8.6" CPUINFER_BUILD_ALL_VARIANTS=0 ./install.sh build

# 安装 ktransformers
RUN --mount=type=cache,target=/root/.cache/pip \
    cd /workspace/ktransformers && pip install .

# 清理缓存（大幅减小体积）
RUN /opt/miniconda3/envs/serve/bin/pip cache purge
RUN conda clean -a -y

# ==============================
# 无NVLink + PCIe 3.0 优化
# ==============================
ENV NCCL_IB_DISABLE=1 \
    NCCL_P2P_DISABLE=0 \
    NCCL_ALGO=RING \
    NCCL_BUFFSIZE=4194304

WORKDIR /workspace
CMD ["/bin/bash"]
