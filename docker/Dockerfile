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
ARG https_proxy
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

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    GDRCOPY_HOME=/usr/src/gdrdrv-${GDRCOPY_VERSION}/ \
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

RUN --mount=type=cache,target=/var/cache/apt,id=base-apt \
    apt-get update && apt-get install -y --no-install-recommends \
    tzdata ca-certificates software-properties-common curl wget git git-lfs \
    build-essential cmake patchelf vim htop zsh \
    libopenmpi-dev libnuma-dev numactl ffmpeg \
    libibverbs-dev librdmacm1 libssl-dev libprotobuf-dev \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /tmp/gdrcopy && cd /tmp \
    && curl -L https://${GITHUB_ARTIFACTORY}/NVIDIA/gdrcopy/archive/refs/tags/v${GDRCOPY_VERSION}.tar.gz -o gdrcopy.tar.gz \
    && tar -zxf gdrcopy.tar.gz && cd gdrcopy-${GDRCOPY_VERSION}/packages \
    && CUDA=/usr/local/cuda ./build-deb-packages.sh \
    && dpkg -i gdrdrv-dkms_*.deb libgdrapi_*.deb \
    && rm -rf /tmp/gdrcopy

RUN ln -sf /usr/lib/$(uname -m)-linux-gnu/libmlx5.so.1 /usr/lib/$(uname -m)-linux-gnu/libmlx5.so

RUN locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8

# Install miniconda
RUN mkdir -p /opt/miniconda3 \
    && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh \
    && bash miniconda.sh -b -u -p /opt/miniconda3 \
    && rm miniconda.sh
ENV PATH="/opt/miniconda3/bin:$PATH"

RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main \
    && conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free

FROM base AS framework

ARG CUDA_VERSION
ARG BUILD_AND_DOWNLOAD_PARALLEL
ARG SGL_KERNEL_VERSION
ARG SGL_VERSION
ARG USE_LATEST_SGLANG
ARG FLASHINFER_VERSION
ARG GRACE_BLACKWELL
ARG HOPPER_SBO
ARG DEEPEP_COMMIT
ARG GITHUB_ARTIFACTORY

# 🔥 直接关闭微调功能，彻底避开 404
ARG FUNCTIONALITY=infer

WORKDIR /workspace

# 只创建 serve 环境，不创建 fine-tune
RUN conda create -n serve python=3.12 -y

RUN /opt/miniconda3/envs/serve/bin/pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# 拉代码
RUN git clone --depth 1 https://${GITHUB_ARTIFACTORY}/kvcache-ai/ktransformers.git \
    && cd ktransformers \
    && git submodule update --init --recursive
RUN ln -s /workspace/ktransformers/third_party/sglang /workspace/sglang

# 安装依赖 + 你要的 web 服务依赖
RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/miniconda3/envs/serve/bin/pip install --upgrade pip setuptools wheel \
        torchviz fire typer uvicorn fastapi pydantic

# 安装 sgl-kernel
RUN --mount=type=cache,target=/root/.cache/pip \
    case "$CUDA_VERSION" in \
        12.8.1) CUINDEX=128 ;; \
        12.9.1) CUINDEX=129 ;; \
        13.0.1) CUINDEX=130 ;; \
        *) CUINDEX=128 ;; \
    esac \
    && /opt/miniconda3/envs/serve/bin/pip install sgl-kernel==${SGL_KERNEL_VERSION}

# 安装 sglang
RUN --mount=type=cache,target=/root/.cache/pip \
    cd /workspace/sglang \
    && /opt/miniconda3/envs/serve/bin/pip install -e "python[all]" --extra-index-url https://download.pytorch.org/whl/cu${CUINDEX}

# flashinfer
RUN --mount=type=cache,target=/root/.cache/pip \
    FLASHINFER_CUBIN_DOWNLOAD_THREADS=${BUILD_AND_DOWNLOAD_PARALLEL} \
    /opt/miniconda3/envs/serve/bin/python -m flashinfer --download-cubin

# DeepEP
RUN curl -L https://${GITHUB_ARTIFACTORY}/deepseek-ai/DeepEP/archive/${DEEPEP_COMMIT}.zip -o deepep.zip \
    && unzip -q deepep.zip && mv DeepEP-${DEEPEP_COMMIT} DeepEP \
    && rm deepep.zip

RUN --mount=type=cache,target=/root/.cache/pip \
    cd DeepEP \
    && TORCH_CUDA_ARCH_LIST="9.0;10.0" MAX_JOBS=${BUILD_AND_DOWNLOAD_PARALLEL} \
    /opt/miniconda3/envs/serve/bin/pip install --no-build-isolation .

# NCCL
RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/miniconda3/envs/serve/bin/pip install nvidia-nccl-cu12==2.28.3 --force-reinstall --no-deps

# 编译 kt-kernel
RUN cd /workspace/ktransformers/kt-kernel \
    && /opt/miniconda3/envs/serve/bin/python -m pip install -e .

RUN conda init bash
RUN echo 'alias serve="conda activate serve"' >> /root/.bashrc

WORKDIR /workspace
CMD ["/bin/bash"]
