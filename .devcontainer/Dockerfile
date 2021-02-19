FROM nvidia/cuda:11.0-devel-ubuntu20.04

# User
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Apt
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-utils \
    dialog 2>&1 \
    python3-pip \
    python3-dev \
    python3-tk \
    python3-setuptools \
    git \
    build-essential \
    curl \
    ca-certificates \
    procps \
    lsb-release \
    graphviz \
    # PIL Image show
    imagemagick \
    wget \
    # OpenCV
    libopencv-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff5-dev \
    libv4l-dev \
    # OpenGL
    # freeglut3-dev \
    # cuda-drivers \
    # OpenGL check app
    # mesa-utils \
    #x11-apps \
    && \
    rm -rf /var/lib/apt/lists/*

# non-root user create
RUN apt-get update && apt-get install -y sudo
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

RUN pip3 install --upgrade pip setuptools

RUN pip3 install --disable-pip-version-check --no-cache-dir \
    autopep8 \
    flake8 \
    wheel \
    ipykernel \
    numpy \
    requests \
    graphviz \
    h5py \
    # jupter notebook
    tqdm \
    ipywidgets \
    # OpenCV
    opencv-contrib-python

# RUN pip install -r requirements.txt

# [Optional] Set the default user. Omit if you want to keep the default as root.
USER $USERNAME

# embree
# RUN sudo wget https://raw.githubusercontent.com/mikedh/trimesh/master/docker/builds/embree.bash
# RUN sudo bash embree.bash
# RUN sudo rm embree.bash
