FROM docker.io/nvidia/cuda:11.7.1-devel-ubuntu22.04
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ENV UV_LINK_MODE=copy

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    ca-certificates \
    wget \
    && rm -rf /var/lib/apt/lists/* \
    && update-ca-certificates

RUN mkdir /opt/venv

ENV VIRTUAL_ENV=/opt/venv
RUN uv venv --python=3.8 /opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN uv pip install \
    torch==1.13.1+cu117 \
    torchvision==0.14.1+cu117 \
    torchaudio==0.13.1 \
    --extra-index-url https://download.pytorch.org/whl/cu117

RUN uv pip install plyfile tqdm setuptools ninja

COPY submodules /tmp/submodules

ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"
RUN uv pip install --no-build-isolation /tmp/submodules/diff-gaussian-rasterization \
    && uv pip install --no-build-isolation /tmp/submodules/simple-knn \
    && rm -rf /tmp/submodules

WORKDIR /workspace
CMD ["/bin/bash"]