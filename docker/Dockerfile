ARG CUDA_VERSION=12.8.1
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu24.04 AS base
ARG TARGETARCH
ARG GRACE_BLACKWELL=0
ARG HOPPER_SBO=0
ARG CPU_VARIANT=x86-intel-multi
ARG BUILD_ALL_CPU_VARIANTS=1

ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG http_proxy
https_proxy
ENV HTTP_PROXY=${HTTP_PROXY} \
    HTTPS_PROXY=${HTTPS_PROXY} \
    http_proxy=${http_proxy} \
    https_proxy=${https_proxy}

ARG GRACE_BLACKWELL_DEEPEP_BRANCH=gb200_blog_part_2
ARG HOPPER_SBO_DEEPEP_COMMIT=9f2fc4b3182a51044ae7ecb6610f7c9c3258c4d6
ARG DEEPEP_COMMIT=9af0e0d0e74f3577af1979c9b9e1ac2cad0104ee
ARG BUILD_AND_DOWNLOAD_PARALLEL=8
ARG SGL_KERNEL_VERSION=0.3.19
ARG SGL_VERSION=0.5.6.post1
ARG USE_LATEST_SGLANG=0
ARG GDRCOPY_VERSION=2.5.1
ARG UBUNTU_MIRROR
ARG GITHUB_ARTIFACTORY=github.com
ARG FLASHINFER_VERSION=0.5.3

ARG KTRANSFORMERS_VERSION=0.5.3
ARG KTRANSFORMERS_WHEEL=ktransformers-0.5.3+cu128torch28fancy-cp312-cp312-linux_x86_64.whl
ARG FLASH_ATTN_WHEEL=flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    FLASHINFER_VERSION=${FLASHINFER_VERSION}

ENV PATH="${PATH}:/usr/local/nvidia/bin" \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"

RUN if [ -n "$UBUNTU_MIRROR" ]; then \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ noble main restricted universe multiverse" > /etc/apt/sources.list && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ noble-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ noble-backports main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb http://security.ubuntu.com/ubuntu/ noble-security main restricted universe multiverse" >> /etc/apt/sources.list && \
    rm -f /etc/apt/sources.list.d/ubuntu.sources; \
fi

# ==============================
# 【安全精简】保留原版所有必需依赖，只删垃圾
# ==============================
RUN --mount=type=cache,target=/var/cache/apt,id=base-apt \
    apt-get update && apt-get install -y --no-install-recommends \
    tzdata ca-certificates software-properties-common \
    curl wget git git-lfs locales \
    build-essential cmake patchelf ninja-build \
    libnuma1 libnuma-dev numactl \
    libssl-dev libprotobuf-dev pybind11-dev \
    ffmpeg libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8 \
    LANGUAGE=en_US:en \
    LC_ALL=en_US.UTF-8

# 【原版安装 Conda，完全不动！】
RUN mkdir -p /opt/miniconda3 \
    && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /opt/miniconda3/miniconda.sh \
    && bash /opt/miniconda3/miniconda.sh -b -u -p /opt/miniconda3 \
    && rm /opt/miniconda3/miniconda.sh

ENV PATH="/opt/miniconda3/bin:${PATH}"

# 【原版 conda 初始化，必须加！】
RUN conda init bash && echo ". /root/.bashrc" >> /etc/profile
SHELL ["/bin/bash", "-c"]

FROM base AS framework
ARG FUNCTIONALITY=infer
ARG CUDA_VERSION
ARG BUILD_AND_DOWNLOAD_PARALLEL
ARG SGL_KERNEL_VERSION
ARG DEEPEP_COMMIT
ARG GITHUB_ARTIFACTORY

WORKDIR /workspace

# ==============================
# 【原版命令】创建环境
# ==============================
RUN conda create -n serve python=3.12 -y
RUN /opt/miniconda3/envs/serve/bin/pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

RUN git clone --depth 1 --branch v${KTRANSFORMERS_VERSION} https://${GITHUB_ARTIFACTORY}/kvcache-ai/ktransformers.git /workspace/ktransformers \
    && cd /workspace/ktransformers && git submodule update --init --recursive \
    && ln -s /workspace/ktransformers/third_party/sglang /workspace/sglang

RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/miniconda3/envs/serve/bin/pip install --upgrade pip setuptools wheel html5lib six

RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/miniconda3/envs/serve/bin/pip install sgl-kernel==${SGL_KERNEL_VERSION}

RUN --mount=type=cache,target=/root/.cache/pip \
    cd /workspace/sglang \
    && /opt/miniconda3/envs/serve/bin/pip install -e "python[all]" --extra-index-url https://download.pytorch.org/whl/cu128

RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/miniconda3/envs/serve/bin/python -m flashinfer --download-cubin

RUN curl -fsSL -o /tmp/deepep.zip https://${GITHUB_ARTIFACTORY}/deepseek-ai/DeepEP/archive/${DEEPEP_COMMIT}.zip \
    && unzip -q /tmp/deepep.zip -d /tmp && mv /tmp/DeepEP-${DEEPEP_COMMIT} /workspace/DeepEP

# 【30系专用】只编译算力8.6
RUN --mount=type=cache,target=/root/.cache/pip \
    cd /workspace/DeepEP && \
    conda activate serve && \
    TORCH_CUDA_ARCH_LIST="8.6" MAX_JOBS=${BUILD_AND_DOWNLOAD_PARALLEL} \
    pip install --no-build-isolation .

RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/miniconda3/envs/serve/bin/pip install nvidia-nccl-cu12==2.28.3 --force-reinstall --no-deps

# 【30系专用】
RUN conda activate serve \
    && cd /workspace/ktransformers/kt-kernel \
    && TORCH_CUDA_ARCH_LIST="8.6" CPUINFER_BUILD_ALL_VARIANTS=0 ./install.sh build

# 清理
RUN /opt/miniconda3/envs/serve/bin/pip cache purge
RUN conda clean -a -y

# 【PCIe 3.0 无NVLink 优化】
ENV NCCL_IB_DISABLE=1 \
    NCCL_P2P_DISABLE=0 \
    NCCL_ALGO=RING

RUN conda init bash
RUN echo 'alias serve="conda activate serve"' >> /root/.bashrc

WORKDIR /workspace
CMD ["/bin/bash"]
