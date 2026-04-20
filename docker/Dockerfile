ARG CUDA_VERSION=12.8.0
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu24.04

# 基础环境
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV TZ=Asia/Shanghai

# 安装系统依赖 + 原生 Python3.12（不用 conda！）
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl build-essential cmake patchelf \
    libopenmpi-dev libnuma-dev libibverbs-dev librdmacm1 \
    libssl-dev unzip python3.12 python3.12-dev python3-pip \
    && rm -rf /var/lib/apt/lists/*

# 设置 python3 指向 3.12
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Pip 清华源
RUN pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# 安装 PyTorch
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 安装你要的依赖
RUN pip install torchviz fire typer uvicorn fastapi pydantic

# 克隆你的代码
RUN git clone https://github.com/X666-CPU/ktransformers.git /workspace/ktransformers

# 进入项目
WORKDIR /workspace/ktransformers

# 安装项目依赖
RUN git submodule update --init --recursive
RUN pip install sgl-kernel==0.3.19
RUN cd third_party/sglang && pip install -e "python[all]"

CMD ["/bin/bash"]
