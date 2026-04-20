ARG CUDA_VERSION=12.8.1
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu24.04 AS base

ENV DEBIAN_FRONTEND=noninteractive

# 安装依赖（修复Python环境）
RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata ca-certificates git git-lfs \
    python3 python3-pip python3-dev python3-venv \
    build-essential cmake patchelf libnuma1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 【修复】PEP 668 错误，允许安装包
RUN python3 -m pip install -U pip --break-system-packages

# 安装编译用 PyTorch
RUN python3 -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128 --break-system-packages

# 克隆代码
RUN git clone --depth 1 --branch v0.5.3 https://github.com/kvcache-ai/ktransformers /workspace/ktransformers
RUN cd /workspace/ktransformers && git submodule update --init --recursive

# 编译内核（仅30系 8.6）
RUN cd /workspace/ktransformers/kt-kernel \
    && TORCH_CUDA_ARCH_LIST="8.6" CPUINFER_BUILD_ALL_VARIANTS=0 python3 install.py build

# ------------------------------
# 最终极小运行镜像
# ------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu24.04 AS runtime

COPY --from=base /workspace/ktransformers /workspace/ktransformers

# 最小运行依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip libnuma1 libgomp1 git \
    && rm -rf /var/lib/apt/lists/*

# 安装依赖（加 --break-system-packages）
RUN python3 -m pip install -U pip wheel --break-system-packages
RUN python3 -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128 --break-system-packages
RUN python3 -m pip install sgl-kernel==0.3.19 --break-system-packages
RUN python3 -m pip install flash-attn==2.8.3 --no-build-isolation --break-system-packages

# 安装 sglang
RUN cd /workspace/ktransformers/third_party/sglang \
    && python3 -m pip install -e "python[all]" --break-system-packages

# 安装 ktransformers
RUN cd /workspace/ktransformers \
    && python3 -m pip install . --break-system-packages

# 清理
RUN python3 -m pip cache purge

# 无NVLink PCIe 3.0 优化
ENV NCCL_IB_DISABLE=1 \
    NCCL_P2P_DISABLE=0 \
    NCCL_ALGO=RING \
    NCCL_BUFFSIZE=4194304

WORKDIR /workspace
CMD ["/bin/bash"]
