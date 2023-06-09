# Credit to : https://github.com/oobabooga/text-generation-webui/blob/main/docker/Dockerfile

FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# yhyu13 add host local proxy server
ENV http_proxy http://127.0.0.1:8889
ENV https_proxy=http://127.0.0.1:8889

RUN apt-get update && \
    apt-get install --no-install-recommends -y git vim build-essential python3-dev python3-venv && \
    rm -rf /var/lib/apt/lists/*

RUN git clone --branch cuda_dev https://github.com/yhyu13/AutoGPTQ.git /build

WORKDIR /build

RUN python3 -m venv /build/venv

# yhyu13 : add mirror, pytorch cuda index should match 
RUN . /build/venv/bin/activate && \ 
    pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip3 config set global.extra-index-url https://download.pytorch.org/whl/cu118

RUN . /build/venv/bin/activate && \
    pip3 install --upgrade pip setuptools && \
    pip3 install torch torchvision torchaudio

# https://developer.nvidia.com/cuda-gpus
# for a rtx 2060: ARG TORCH_CUDA_ARCH_LIST="7.5"
# yhyu13 : edit for your rig, mine is 3090, so 8.6
ARG TORCH_CUDA_ARCH_LIST="8.6"
ENV BUILD_CUDA_EXT=1 
RUN . /build/venv/bin/activate && \
    pip3 install wheel && \
    python3 setup.py bdist_wheel -d .

RUN mkdir /result && cp /build/*whl /result

CMD echo "we are done!"