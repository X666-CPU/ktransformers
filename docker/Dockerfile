ARG CUDA_VERSION=12.8.1
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu24.04 AS base
ARG TARGETARCH
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

# 固定适配30系显卡的版本（移除Hopper/Blackwell无用参数）
ARG DEEPEP_COMMIT=9af0e0d0e74f3577af1979c9b9e1ac2cad0104ee
ARG BUILD_AND_DOWNLOAD_PARALLEL=8
ARG SGL_KERNEL_VERSION=0.3.19
ARG SGL_VERSION=0.5.6.post1
ARG USE_LATEST_SGLANG=0
ARG FLASHINFER_VERSION=0.5.3

# 固定wheel版本
ARG KTRANSFORMERS_VERSION=0.5.3
ARG KTRANSFORMERS_WHEEL=ktransformers-0.5.3+cu128torch28fancy-cp312-cp312-linux_x86_64.whl
ARG FLASH_ATTN_WHEEL=flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda

# 核心优化：RTX3060/3090 专属算力架构 8.6
ENV TORCH_CUDA_ARCH_LIST="8.6"

# Add GKE default lib path
ENV PATH="${PATH}:/usr/local/nvidia/bin" \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"

# Tsinghua Ubuntu Mirror
RUN if [ -n "$UBUNTU_MIRROR" ]; then \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ noble main restricted universe multiverse" > /etc/apt/sources.list && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ noble-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ noble-backports main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb http://security.ubuntu.com/ubuntu/ noble-security main restricted universe multiverse" >> /etc/apt/sources.list && \
    rm -f /etc/apt/sources.list.d/ubuntu.sources; \
fi

# 优化：仅安装必要依赖，删除IB/RDMA/NVLINK/GDRCopy无用服务器组件
RUN --mount=type=cache,target=/var/cache/apt,id=base-apt \
    echo 'tzdata tzdata/Areas select Asia' | debconf-set-selections \
    && echo 'tzdata tzdata/Zones/Asia select Shanghai' | debconf-set-selections \
    && apt-get update && apt-get install -y --no-install-recommends --allow-change-held-packages \
    tzdata ca-certificates software-properties-common netcat-openbsd kmod unzip \
    curl wget git git-lfs build-essential cmake perl patchelf ccache \
    libopenmpi-dev libnuma1 libnuma-dev numactl ffmpeg \
    libgoogle-glog-dev libgtest-dev libjsoncpp-dev libunwind-dev libboost-all-dev \
    libssl-dev libprotobuf-dev protobuf-compiler pybind11-dev libcurl4-openssl-dev \
    devscripts debhelper fakeroot dkms gdb ninja-build vim tmux htop tree less \
    libnccl2 libnccl-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Locale
RUN locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8 \
    LANGUAGE=en_US:en \
    LC_ALL=en_US.UTF-8

########################################################
########## Miniconda ###########################
########################################################
RUN mkdir -p /opt/miniconda3 \
    && wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /opt/miniconda3/miniconda.sh \
    && bash /opt/miniconda3/miniconda.sh -b -u -p /opt/miniconda3 \
    && rm /opt/miniconda3/miniconda.sh

ENV PATH="/opt/miniconda3/bin:${PATH}"

RUN conda tos accept --override-channels \
    && conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main \
    && conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free \
    && conda config --set show_channel_urls yes

########################################################
########## Dual Conda Environment ################
########################################################
FROM base AS framework
ARG CUDA_VERSION
ARG BUILD_AND_DOWNLOAD_PARALLEL
ARG SGL_KERNEL_VERSION
ARG FLASHINFER_VERSION
ARG DEEPEP_COMMIT
ARG KTRANSFORMERS_WHEEL
ARG FLASH_ATTN_WHEEL
ARG FUNCTIONALITY=sft

WORKDIR /workspace

# Create envs
RUN conda create -n serve python=3.12 -y \
    && if [ "$FUNCTIONALITY" = "sft" ]; then conda create -n fine-tune python=3.12 -y; fi

# Pip mirror
RUN /opt/miniconda3/envs/serve/bin/pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple \
    && if [ "$FUNCTIONALITY" = "sft" ]; then \
        /opt/miniconda3/envs/fine-tune/bin/pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple; \
    fi

# Clone code
RUN git clone --depth 1 https://github.com/kvcache-ai/ktransformers.git /workspace/ktransformers \
    && cd /workspace/ktransformers && git submodule update --init --recursive \
    && ln -s /workspace/ktransformers/third_party/sglang /workspace/sglang \
    && if [ "$FUNCTIONALITY" = "sft" ]; then \
        git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git /workspace/LLaMA-Factory; \
    fi

# Download wheels
RUN if [ "$FUNCTIONALITY" = "sft" ]; then \
        curl -fsSL -o /workspace/${KTRANSFORMERS_WHEEL} \
            https://github.com/kvcache-ai/ktransformers/releases/download/v0.5.3/${KTRANSFORMERS_WHEEL} \
        && curl -fsSL -o /workspace/${FLASH_ATTN_WHEEL} \
            https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/${FLASH_ATTN_WHEEL}; \
    fi

########################################################
# Serve Environment (sglang + kt-kernel)
########################################################
RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/miniconda3/envs/serve/bin/pip install --upgrade pip setuptools wheel

# Install sgl-kernel (CUDA128)
RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/miniconda3/envs/serve/bin/pip install sgl-kernel==${SGL_KERNEL_VERSION}

# Install SGLang
RUN --mount=type=cache,target=/root/.cache/pip \
    cd /workspace/sglang \
    && /opt/miniconda3/envs/serve/bin/pip install -e "python[all]" --extra-index-url https://download.pytorch.org/whl/cu128

# FlashInfer
RUN --mount=type=cache,target=/root/.cache/pip \
    FLASHINFER_CUBIN_DOWNLOAD_THREADS=${BUILD_AND_DOWNLOAD_PARALLEL} \
    /opt/miniconda3/envs/serve/bin/python -m flashinfer --download-cubin

# Install DeepEP (仅编译8.6架构)
RUN curl -fsSL -o /tmp/${DEEPEP_COMMIT}.zip \
    https://github.com/deepseek-ai/DeepEP/archive/${DEEPEP_COMMIT}.zip \
    && unzip -q /tmp/${DEEPEP_COMMIT}.zip -d /tmp && rm /tmp/${DEEPEP_COMMIT}.zip \
    && mv /tmp/DeepEP-${DEEPEP_COMMIT} /workspace/DeepEP \
    && cd /workspace/DeepEP \
    && sed -i 's/#define NUM_CPU_TIMEOUT_SECS 100/#define NUM_CPU_TIMEOUT_SECS 1000/' csrc/kernels/configs.cuh

RUN --mount=type=cache,target=/root/.cache/pip \
    cd /workspace/DeepEP \
    && . /opt/miniconda3/etc/profile.d/conda.sh && conda activate serve \
    && TORCH_CUDA_ARCH_LIST="8.6" MAX_JOBS=${BUILD_AND_DOWNLOAD_PARALLEL} \
    pip install --no-build-isolation .

# NCCL
RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/miniconda3/envs/serve/bin/pip install nvidia-nccl-cu12==2.28.3 --force-reinstall --no-deps

# kt-kernel
RUN . /opt/miniconda3/etc/profile.d/conda.sh && conda activate serve \
    && cd /workspace/ktransformers/kt-kernel \
    && CPUINFER_BUILD_ALL_VARIANTS=1 ./install.sh build

########################################################
# Fine-tune Environment
########################################################
RUN if [ "$FUNCTIONALITY" = "sft" ]; then \
        conda install -n fine-tune -y -c conda-forge libstdcxx-ng; \
    fi

# PyTorch 2.8
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "$FUNCTIONALITY" = "sft" ]; then \
        /opt/miniconda3/envs/fine-tune/bin/pip install --upgrade pip setuptools wheel hatchling \
        && /opt/miniconda3/envs/fine-tune/bin/pip install \
            torch==2.8.0 torchvision torchaudio \
            --extra-index-url https://download.pytorch.org/whl/cu128; \
    fi

# LLaMA-Factory
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "$FUNCTIONALITY" = "sft" ]; then \
        cd /workspace/LLaMA-Factory \
        && /opt/miniconda3/envs/fine-tune/bin/pip install -e ".[torch,metrics]" --no-build-isolation; \
    fi

# Wheels
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "$FUNCTIONALITY" = "sft" ]; then \
        /opt/miniconda3/envs/fine-tune/bin/pip install /workspace/${KTRANSFORMERS_WHEEL} ${FLASH_ATTN_WHEEL}; \
    fi

# NCCL
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "$FUNCTIONALITY" = "sft" ]; then \
        /opt/miniconda3/envs/fine-tune/bin/pip install nvidia-nccl-cu12==2.28.3 --force-reinstall --no-deps; \
    fi

########################################################
# 关键优化：NCCL 适配 PCIe3.0 + 无NVLINK 双显卡
########################################################
ENV NCCL_DEBUG=INFO \
    NCCL_IB_DISABLE=1 \
    NCCL_P2P_DISABLE=0 \
    NCCL_SHM_DISABLE=0 \
    NCCL_SOCKET_IFNAME=eth0 \
    CUDA_VISIBLE_DEVICES=0,1 \
    CUDA_DEVICE_ORDER=PCI_BUS_ID

# Cleanup
RUN if [ "$FUNCTIONALITY" = "sft" ]; then \
        rm -f /workspace/${KTRANSFORMERS_WHEEL} /workspace/${FLASH_ATTN_WHEEL}; \
    fi

RUN /opt/miniconda3/bin/conda init bash
RUN echo '\nalias serve="conda activate serve"' >> /root/.bashrc \
    && if [ "$FUNCTIONALITY" = "sft" ]; then \
        echo 'alias finetune="conda activate fine-tune"' >> /root/.bashrc; \
    fi

# Version
RUN cd /workspace/ktransformers && \
    KTRANSFORMERS_VERSION=$(python3 -c "exec(open('version.py').read()); print(__version__)" || echo "unknown") && \
    echo "KTRANSFORMERS_VERSION=$KTRANSFORMERS_VERSION" > /workspace/versions.env && \
    echo "SGLANG_KT_VERSION=$KTRANSFORMERS_VERSION" >> /workspace/versions.env && \
    if [ "$FUNCTIONALITY" = "sft" ]; then \
        . /opt/miniconda3/etc/profile.d/conda.sh && conda activate fine-tune && \
        cd /workspace/LLaMA-Factory && \
        LLAMAFACTORY_VERSION=$(python -c "import sys; sys.path.insert(0, 'src'); from llamafactory import __version__; print(__version__)" || echo "unknown") && \
        echo "LLAMAFACTORY_VERSION=$LLAMAFACTORY_VERSION" >> /workspace/versions.env; \
    else \
        echo "LLAMAFACTORY_VERSION=none" >> /workspace/versions.env; \
    fi

WORKDIR /workspace
CMD ["/bin/bash"]
