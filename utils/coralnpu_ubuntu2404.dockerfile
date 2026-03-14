# syntax=docker/dockerfile:1
# Dockerfile to create a stable CoralNPU build environment
#
# Build command:
# docker build --build-arg _UID=$(id -u) --build-arg _GID=$(id -g) --build-arg _USERNAME=$(whoami) -t coralnpu_ubuntu24 -f utils/coralnpu_ubuntu24.dockerfile .

#
# Run command:
# docker run -it -v $HOME:/home/$USER -w /home/$USER/mycode/coralnpu coralnpu_ubuntu24 /bin/bash

FROM ubuntu:24.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
ARG _UID=1000
ARG _GID=1000
ARG _USERNAME=builder
ENV HOME=/home/${_USERNAME}

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked --mount=type=cache,target=/var/lib/apt,sharing=locked <<EOF
    set -e
    ln -snf "/usr/share/zoneinfo/${TZ}" /etc/localtime
    echo "${TZ}" > /etc/timezone
    echo "APT::Get::Assume-Yes \"true\";" > /etc/apt/apt.conf.d/90assumeyes
    apt-get update
    apt-get install -y -qq \
        apt-transport-https \
        autoconf \
        build-essential \
        ca-certificates \
        ccache \
        curl \
        fuse3 \
        gawk \
        git \
        gnupg \
        libelf-dev \
        libmpfr-dev \
        lsb-release \
        openjdk-17-jdk-headless \
        openjdk-17-jre-headless \
        python-is-python3 \
        python3 \
        python3-pip \
        python3-setuptools \
        python3-venv \
        srecord \
        sudo \
        tzdata \
        unzip \
        xxd \
        zip
    update-ca-certificates
    # Install clang-19 and configure
    apt-get install -y -qq \
        clang-19 \
        clang-tools-19 \
        lld-19 \
        clang-format-19 \
        clang-tidy-19
    ln -s /usr/bin/clang-19 /usr/bin/clang
    ln -s /usr/bin/clang++-19 /usr/bin/clang++
    ln -s /usr/bin/clang-cpp-19 /usr/bin/clang-cpp
    ln -s /usr/bin/clang-format-19 /usr/bin/clang-format
    ln -s /usr/bin/clang-tidy-19 /usr/bin/clang-tidy
    ln -s /usr/bin/ld.lld-19 /usr/bin/ld.lld
    ln -s /usr/bin/llvm-symbolizer-19 /usr/bin/llvm-symbolizer
    # end clang-19
    curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor > /tmp/bazel-archive-keyring.gpg
    mv /tmp/bazel-archive-keyring.gpg /usr/share/keyrings/
    echo "deb [arch=$(dpkg-architecture -q DEB_HOST_ARCH) signed-by=/usr/share/keyrings/bazel-archive-keyring.gpg] https://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
    apt update
    apt install bazel bazel-7.4.1
    echo "${_USERNAME} ALL=(ALL) NOPASSWD:/usr/bin/apt" >> /etc/sudoers.d/${_USERNAME}
    echo "${_USERNAME} ALL=(ALL) NOPASSWD:/bin/mkdir" >> /etc/sudoers.d/${_USERNAME}
    echo "${_USERNAME} ALL=(ALL) NOPASSWD:/bin/chown" >> /etc/sudoers.d/${_USERNAME}
    echo "${_USERNAME} ALL=(ALL) NOPASSWD:/bin/ln" >> /etc/sudoers.d/${_USERNAME}
    echo "${_USERNAME} ALL=(ALL) NOPASSWD:/usr/bin/xargs" >> /etc/sudoers.d/${_USERNAME}
    
    # Ubuntu 24.04 base image comes with a default 'ubuntu' user at UID/GID 1000.
    # We remove it beforehand to avoid conflicts when creating our user.
    if id -u ubuntu >/dev/null 2>&1; then userdel -r ubuntu || true; fi
    
    addgroup --gid ${_GID} ${_USERNAME}
    adduser \
        --home ${HOME} \
        --disabled-password \
        --gecos "" \
        --uid ${_UID} \
        --gid ${_GID} \
        ${_USERNAME}
    chown ${_USERNAME}:${_USERNAME} ${HOME}
    # Work around differeing libmpfr versions between distros
    ln -sf /lib/x86_64-linux-gnu/libmpfr.so.6.2.0 /lib/x86_64-linux-gnu/libmpfr.so.4
EOF
USER ${_USERNAME}
WORKDIR /home/${_USERNAME}/
