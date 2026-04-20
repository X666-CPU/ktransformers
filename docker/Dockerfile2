ARG CUDA_VERSION=12.8.1
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu24.04 AS base
ARG TARGETARCH

# 代理
ARG HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
ENV HTTP_PROXY=${HTTP_PROXY} HTTPS_PROXY=${HTTPS_PROXY} \
    http_proxy=${http_proxy} https_proxy=${https_proxy}

# 版本锁定（支持Qwen3.6）
ARG DEEPEP_COMMIT=9af0e0d0e74f3577af1979c9b9e1ac2cad0104ee
ARG BUILD_AND_DOWNLOAD_PARALLEL=4
ARG SGL_KERNEL_VERSION=0.3.19
ARG KTRANSFORMERS_VERSION=0.5.3
ARG FLASHINFER_VERSION=0.5.3
ARG UBUNTU_MIRROR
ARG GITHUB_ARTIFACTORY=github.com

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    LD_LIBRARY_PATH=/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

# 清华源
RUN if [ -n "$UBUNTU_MIRROR" ]; then \
    sed -i 's/archive.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list \
    && sed -i 's/security.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list; \
fi

# 【最小依赖】只装运行时
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
    tzdata ca-certificates git git-lfs \
    libnuma1 libgomp1 python3.12 python3-pip \
    && rm -rf /var/lib/apt/lists/* && apt-get clean

# 多阶段 builder（编译用）
FROM base AS builder
RUN apt-get update && apt-get install -y build-essential cmake patchelf

# 克隆ktransformers
RUN git clone --depth 1 --branch v${KTRANSFORMERS_VERSION} \
    https://${GITHUB_ARTIFACTORY}/kvcache-ai/ktransformers.git /workspace/ktransformers \
    && cd /workspace/ktransformers && git submodule update --init --recursive

# 编译kt-kernel（仅30系算力8.6）
RUN cd /workspace/ktransformers/kt-kernel \
    && TORCH_CUDA_ARCH_LIST="8.6" CPUINFER_BUILD_ALL_VARIANTS=0 ./install.sh build

# 最终 runtime 镜像
FROM base AS framework
WORKDIR /workspace

# 复制编译产物
COPY --from=builder /workspace/ktransformers /workspace/ktransformers

# 安装miniconda（仅serve环境）
RUN wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh \
    && bash miniconda.sh -b -p /opt/miniconda3 && rm miniconda.sh
ENV PATH=/opt/miniconda3/bin:$PATH

# 创建推理环境
RUN conda create -n serve python=3.12 -y \
    && /opt/miniconda3/envs/serve/bin/pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# 安装依赖（PCIe优化）
RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/miniconda3/envs/serve/bin/pip install --upgrade pip setuptools wheel \
    && /opt/miniconda3/envs/serve/bin/pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128 \
    && /opt/miniconda3/envs/serve/bin/pip install sgl-kernel==${SGL_KERNEL_VERSION} \
    && cd /workspace/ktransformers/third_party/sglang \
    && /opt/miniconda3/envs/serve/bin/pip install -e "python[all]" --extra-index-url https://download.pytorch.org/whl/cu128

# FlashInfer（仅30系）
RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/miniconda3/envs/serve/bin/python -m flashinfer --download-cubin

# DeepEP（仅8.6）
RUN curl -fsSL -o /tmp/deepep.zip https://${GITHUB_ARTIFACTORY}/deepseek-ai/DeepEP/archive/${DEEPEP_COMMIT}.zip \
    && unzip -q /tmp/deepep.zip -d /tmp && mv /tmp/DeepEP-${DEEPEP_COMMIT} /workspace/DeepEP
RUN --mount=type=cache,target=/root/.cache/pip \
    cd /workspace/DeepEP \
    && . /opt/miniconda3/etc/profile.d/conda.sh && conda activate serve \
    && TORCH_CUDA_ARCH_LIST="8.6" MAX_JOBS=${BUILD_AND_DOWNLOAD_PARALLEL} \
    pip install --no-build-isolation .

# NCCL（无NVLink专用）
RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/miniconda3/envs/serve/bin/pip install nvidia-nccl-cu12==2.28.3 --force-reinstall --no-deps

# 安装ktransformers
RUN --mount=type=cache,target=/root/.cache/pip \
    cd /workspace/ktransformers \
    && . /opt/miniconda3/etc/profile.d/conda.sh && conda activate serve \
    && pip install .

# 清理
RUN /opt/miniconda3/envs/serve/bin/pip cache purge \
    && conda clean -a -y

# 环境别名
RUN /opt/miniconda3/bin/conda init bash \
    && echo 'alias serve="conda activate serve"' >> /root/.bashrc

# 无NVLink PCIe优化环境变量（关键）
ENV NCCL_P2P_DISABLE=0 \
    NCCL_IB_DISABLE=1 \
    NCCL_SHM_DISABLE=0 \
    NCCL_ALGO=RING \
    NCCL_BUFFSIZE=4194304 \
    NCCL_MIN_NCHANNELS=2 \
    CUDA_VISIBLE_DEVICES=0,1

WORKDIR /workspace
CMD ["/bin/bash"]
