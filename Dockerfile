# syntax=docker/dockerfile:1

# CUDA 12.x runs on any NVIDIA driver >= 525 (minor-version compatibility);
# CUDA 13 would require driver >= 580. The CUDA runtime and cuBLAS are linked
# statically, so the runtime image only needs the driver's libcuda, which the
# NVIDIA Container Toolkit injects.
ARG CUDA_VERSION=12.9.1
ARG UBUNTU_VERSION=24.04

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS builder

ARG DEBIAN_FRONTEND=noninteractive
ARG RUST_VERSION=1.98.1
# GPU architectures to compile kernels for. 75 = Turing (GTX 16xx, RTX 20xx).
# Examples: "75;86;89" for Turing + Ampere + Ada. Fewer architectures build
# much faster and give a smaller binary.
#
# GTX 16xx cards (e.g. GTX 1650) are Turing without tensor cores. llama.cpp
# recommends its Pascal code path for them:
#   --build-arg CUDA_ARCHITECTURES=61-virtual --build-arg GGML_CUDA_FORCE_MMQ=ON
# The driver JIT-compiles that PTX on first use and caches it in
# CUDA_CACHE_PATH (on the /models volume, so only the first start is slow).
ARG CUDA_ARCHITECTURES=75
ARG GGML_CUDA_FORCE_MMQ=OFF
# CPU instruction set for llama.cpp's CPU code (used for layers that don't fit
# on the GPU). x86-64-v3 = AVX2/FMA/F16C/BMI2: every x86 CPU since ~2015
# (Intel Haswell, AMD Zen 1). Without it llama.cpp is built for plain SSE2.
# Don't use "native": it would target the build machine's CPU.
ARG RUST_TARGET_CPU=x86-64-v3

ENV RUSTUP_HOME=/root/.rustup \
    CARGO_HOME=/root/.cargo \
    PATH=/root/.cargo/bin:${PATH}

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        build-essential \
        clang \
        libclang-dev \
        cmake \
        pkg-config \
        libssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o /tmp/rustup-init.sh \
    && sh /tmp/rustup-init.sh -y --profile minimal --default-toolchain "${RUST_VERSION}" \
    && rm /tmp/rustup-init.sh

WORKDIR /build
COPY . .

# llama-cpp-sys-2 forwards CMAKE_* environment variables to CMake. The driver
# stub lets the linker resolve libcuda.so.1 without a GPU in the build.
RUN --mount=type=cache,id=ltengine-cargo-registry,target=/root/.cargo/registry,sharing=locked \
    --mount=type=cache,id=ltengine-cargo-git,target=/root/.cargo/git,sharing=locked \
    --mount=type=cache,id=ltengine-target-cuda${CUDA_VERSION}-sm${CUDA_ARCHITECTURES}-mmq${GGML_CUDA_FORCE_MMQ},target=/build/target \
    RUSTFLAGS="-C target-cpu=${RUST_TARGET_CPU}" \
    CMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
    GGML_CUDA_FORCE_MMQ="${GGML_CUDA_FORCE_MMQ}" \
    LIBRARY_PATH=/usr/local/cuda/lib64/stubs \
    cargo build --locked --release --features cuda -p ltengine \
    && install -Dm755 -s target/release/ltengine /out/ltengine


FROM nvidia/cuda:${CUDA_VERSION}-base-ubuntu${UBUNTU_VERSION}

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --system --uid 10001 --no-create-home --shell /usr/sbin/nologin ltengine \
    && mkdir -p /models \
    && chown ltengine /models

COPY --from=builder /out/ltengine /usr/local/bin/ltengine

# Models are downloaded once into /models (Hugging Face cache layout) or
# mounted there and selected with LTENGINE_MODEL_FILE.
ENV HF_HOME=/models \
    CUDA_CACHE_PATH=/models/.cuda-cache \
    CUDA_CACHE_MAXSIZE=4294967296 \
    LTENGINE_HOST=0.0.0.0 \
    LTENGINE_PORT=5050 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

VOLUME ["/models"]
EXPOSE 5050
USER ltengine

# Ready = model loaded. The first start may download the model, hence the
# long start period. The check is a local HTTP request, not an inference.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15m --retries=3 \
    CMD ["ltengine", "--healthcheck"]

ENTRYPOINT ["ltengine"]
