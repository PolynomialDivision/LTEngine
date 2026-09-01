# syntax=docker/dockerfile:1

ARG CUDA_VERSION=13.3.1
ARG UBUNTU_VERSION=26.04

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS builder

ARG DEBIAN_FRONTEND=noninteractive
ARG RUST_VERSION=1.98.0

ENV RUSTUP_HOME=/root/.rustup \
    CARGO_HOME=/root/.cargo \
    PATH=/root/.cargo/bin:${PATH}

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        build-essential \
        clang \
        cmake \
        pkg-config \
        libssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf \
        https://sh.rustup.rs \
        -o /tmp/rustup-init.sh \
    && sh /tmp/rustup-init.sh \
        -y \
        --profile minimal \
        --default-toolchain "${RUST_VERSION}" \
    && rm /tmp/rustup-init.sh

WORKDIR /build

COPY . .

RUN --mount=type=cache,id=ltengine-cargo-registry,target=/root/.cargo/registry,sharing=locked \
    --mount=type=cache,id=ltengine-cargo-git,target=/root/.cargo/git,sharing=locked \
    --mount=type=cache,id=ltengine-target-cuda1331-ubuntu2604-nonccl,target=/build/target \
    cargo build \
        --locked \
        --features cuda \
        --release \
        -p ltengine \
    && install -Dm755 -s target/release/ltengine /out/ltengine


FROM nvidia/cuda:${CUDA_VERSION}-base-ubuntu${UBUNTU_VERSION}

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        libssl3t64 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd \
        --system \
        --no-create-home \
        --shell /usr/sbin/nologin \
        ltengine \
    && mkdir -p /models \
    && chown ltengine:ltengine /models

COPY --from=builder /out/ltengine /usr/local/bin/ltengine

ENV HF_HOME=/models \
    LTENGINE_MODEL=gemma3-4b

VOLUME ["/models"]

EXPOSE 5050

USER ltengine

CMD ["sh", "-c", "set -eu; set -- ltengine --host 0.0.0.0 -m \"$LTENGINE_MODEL\"; if [ -n \"${LTENGINE_MODEL_FILE:-}\" ]; then set -- \"$@\" --model-file \"$LTENGINE_MODEL_FILE\"; fi; exec \"$@\""]
