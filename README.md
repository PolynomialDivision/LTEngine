# LTEngine

Free and Open Source Local AI Machine Translation API, written in Rust, entirely self-hosted and compatible with [LibreTranslate](https://github.com/LibreTranslate/LibreTranslate). Its translation capabilities are powered by large language models (LLMs) that run locally on your machine via [llama.cpp](https://github.com/ggml-org/llama.cpp). 

![Translation](https://github.com/user-attachments/assets/37dd4e20-382b-459d-bcc1-5de3ed4b4c18)

The LLMs in LTEngine are much larger than the lightweight transformer models in [LibreTranslate](https://github.com/LibreTranslate/LibreTranslate). Thus memory usage and speed are traded off for quality of outputs, which for some languages has been reported as being [on par or better than DeepL](https://community.libretranslate.com/t/ltengine-llm-powered-local-machine-translation/1862/5).

It is possible to run LTEngine entirely on the CPU, but an accelerator will greatly improve performance. Supported accelerators currently include:

 * CUDA
 * Metal (macOS)
 * Vulkan

 The default model, [TranslateGemma 4B](https://huggingface.co/google/translategemma-4b-it), runs fully on a 4 GB GPU such as a GTX 1650. The largest model (`gemma3-27b`) fits on a single RTX 3090 with 24 GB of VRAM.

> ⚠️ LTEngine is in active development. Check the [Roadmap](#roadmap) for current limitations.


## Requirements

 * [Rust](https://www.rust-lang.org/)
 * [clang](https://clang.llvm.org/)
 * [CMake](https://cmake.org/)
 * A C++ compiler (g++, MSVC) for building the llama.cpp bindings
 * For CUDA builds: the CUDA toolkit

## Build

```bash
git clone https://github.com/LibreTranslate/LTEngine --recursive
cd LTEngine
cargo build [--features cuda,vulkan,metal] --release
```

## Run

```bash
./target/release/ltengine
```

To run a different model, or a local GGUF file:

```bash
./target/release/ltengine -m gemma3-12b
./target/release/ltengine --model-file /path/to/model.gguf
```

Every option can also be set through an environment variable (`LTENGINE_*`, see `ltengine --help`).

## Models

LTEngine supports any GGUF language model supported by [llama.cpp](https://github.com/ggml-org/llama.cpp). Models are downloaded once into the Hugging Face cache (`$HF_HOME`, `/models` in Docker) and loaded from there on later starts. Pass `--model-file` to use a local `.gguf` instead.

| Model                  | File                                | GPU memory (est.) | Notes                                            |
| ---------------------- | ----------------------------------- | ---------- | ------------------------------------------------ |
| `translategemma-4b`    | TranslateGemma 4B, Q5_K_M (2.7 GB)  | ~3.2 GB    | **Default.** Fits fully on a 4 GB GPU            |
| `translategemma-4b-q4` | TranslateGemma 4B, Q4_K_M (2.4 GB)  | ~2.9 GB    | For 4 GB GPUs shared with other workloads        |
| `translategemma-12b`   | TranslateGemma 12B, Q4_K_M (7.0 GB) | ~8 GB      | Better quality, needs ≥ 10 GB                    |
| `gemma3-1b` … `gemma3-27b`, `gemma4-e4b` | Gemma chat models  |            | Generic translation prompt                       |

GPU memory figures are estimates (weights + KV cache + compute buffers at the default 4096-token context); the startup log reports the real usage.

### TranslateGemma

[TranslateGemma](https://huggingface.co/google/translategemma-4b-it) is Google's translation fine-tune of Gemma 3 (55 languages). Its chat template doesn't take a free-form instruction. It takes a structured message with `source_lang_code`, `target_lang_code` and `text`, and renders Google's own translation prompt from it. llama.cpp's built-in template support can't express that, so LTEngine detects TranslateGemma from the chat template in the GGUF and renders that template itself (with [minijinja](https://github.com/mitsuhiko/minijinja)). If the GGUF has no template, an identical built-in prompt is used.

A plain API request is all that is needed:

```json
{ "q": "Hallo Welt", "source": "de", "target": "uk" }
```

Notes:

 * Language codes are validated against the model's list; unsupported codes return HTTP 400.
 * `"source": "auto"` detects the language first, because TranslateGemma needs an explicit source.
 * `format: "html"` is translated as Markdown and converted back, since TranslateGemma preserves Markdown but not HTML tags.
 * Decoding is greedy (deterministic). Output is capped at `3 × input tokens + 32` (at most `--max-new-tokens`), so a short message can't generate runaway text.

## GPU memory and performance

LTEngine is tuned so that if startup succeeds, requests cannot run out of GPU memory:

 * **Auto offload.** With `--gpu-layers auto` (default), llama.cpp's own fitter does a dry-run allocation of the model, KV cache and compute buffers, and offloads as much as fits while leaving `--vram-margin` MiB free (512 MiB on GPUs below 8 GB). If allocation still fails, LTEngine retries with fewer layers.
 * **One persistent context.** The context is allocated once and warmed up with a worst-case batch before the engine reports ready. Requests reuse it; nothing is allocated per request.
 * **Small buffers.** The sliding-window KV cache is sized to the window instead of the full context (Gemma 3's 5-of-6 local layers). `--ubatch-size` defaults to 128 on GPUs below 6 GB; with Gemma's 262k vocabulary, the logits scratch buffer alone is `ubatch × 1 MiB`.
 * **No silent CPU fallback.** A CUDA build refuses to start without a usable GPU unless `--allow-cpu-fallback` (or `--cpu`) is set.
 * **Serialized inference.** One translation runs at a time; up to `--queue-size` requests wait. Beyond that, or after `--queue-timeout` seconds of waiting, requests get `503 Server busy`. If a client disconnects, its translation is cancelled.

| Option (env)                                 | Default | Description                                          |
| -------------------------------------------- | ------- | ---------------------------------------------------- |
| `--model` (`LTENGINE_MODEL`)                  | `translategemma-4b` | Model to download                        |
| `--model-file` (`LTENGINE_MODEL_FILE`)        |         | Local GGUF file                                      |
| `--gpu-layers` (`LTENGINE_GPU_LAYERS`)        | `auto`  | `auto`, `all` or a number                            |
| `--vram-margin` (`LTENGINE_VRAM_MARGIN`)      | 512/1024 | MiB of VRAM to leave free                           |
| `--ctx-size` (`LTENGINE_CTX_SIZE`)            | 4096    | Tokens for prompt + translation                      |
| `--batch-size` / `--ubatch-size`              | 512 / auto | Logical / physical batch size                     |
| `--threads` (`LTENGINE_THREADS`)              | cores, ≤ 8 | CPU threads                                       |
| `--kv-cache-type` (`LTENGINE_KV_CACHE_TYPE`)  | `f16`   | `q8_0` halves KV memory                              |
| `--flash-attn` (`LTENGINE_FLASH_ATTN`)        | `auto`  | `auto`, `on`, `off`                                  |
| `--max-new-tokens` (`LTENGINE_MAX_NEW_TOKENS`)| 2048    | Hard cap on generated tokens                         |
| `--queue-size` (`LTENGINE_QUEUE_SIZE`)        | 16      | Waiting requests before `503`                        |
| `--queue-timeout` (`LTENGINE_QUEUE_TIMEOUT`)  | 60      | Seconds a request may wait                           |
| `--cache-size` (`LTENGINE_CACHE_SIZE`)        | 0       | In-memory LRU of translations (keeps text in RAM)    |
| `--char-limit` (`LTENGINE_CHAR_LIMIT`)        | 5000    | Maximum characters per request                       |
| `--api-key` (`LTENGINE_API_KEY`)              |         | Require an API key                                   |
| `--cpu` / `--allow-cpu-fallback`              | off     | CPU only / allow CPU if no GPU                       |
| `-v` (`LTENGINE_VERBOSE`), `RUST_LOG`         | `info`  | Logging (`-v` includes llama.cpp logs)               |

## Docker (NVIDIA GPU, e.g. Unraid)

The image is a multi-stage build: CUDA 12.9 devel for building, slim CUDA base image for running (the CUDA runtime and cuBLAS are linked statically, so the container only needs the host driver). It needs the NVIDIA Container Toolkit (on Unraid: the *Nvidia Driver* plugin) and a driver ≥ 525. Kernels are compiled for `CUDA_ARCHITECTURES=75` (Turing: GTX 16xx, RTX 20xx) by default. Pass e.g. `--build-arg CUDA_ARCHITECTURES="75;86;89"` for other GPUs.

On Unraid:

```bash
mkdir -p /mnt/user/appdata/ltengine/models
chown -R 99:100 /mnt/user/appdata/ltengine
cd /mnt/user/appdata/ltengine && git clone <this repository> src && cd src
docker compose up -d --build
docker logs -f ltengine        # first start downloads the model (~2.7 GB)
```

The included [`docker-compose.yml`](docker-compose.yml) reserves the GPU, stores models in `/mnt/user/appdata/ltengine/models`, and doesn't publish any port. Other containers on the `translate` network reach it at `http://ltengine:5050`. To use it from the host, uncomment the `127.0.0.1:5050` port mapping.

If the container exits with `libcuda.so.1: cannot open shared object file`, it was started without the NVIDIA runtime: the driver library is injected by the NVIDIA Container Toolkit (check the GPU reservation / `--gpus all`, and that the Nvidia Driver plugin is installed).

To use a GGUF file you downloaded yourself, put it into the models directory and set `LTENGINE_MODEL_FILE=/models/<file>.gguf`.

A healthy startup log looks like:

```
GPU 0: NVIDIA GeForce GTX 1650 (3800 MiB free / 3903 MiB total) backend=CUDA
inference settings gpu=true ctx_size=4096 batch_size=512 ubatch_size=128 threads=8 ...
model loaded name="Translategemma 4b It" architecture="gemma3" quantization="Q5_K_M" ... prompt_format="translategemma (GGUF template)"
TranslateGemma detected: using the native translation prompt with explicit language codes
GPU offload: all 35 layers on the GPU
VRAM: ... MiB used by LTEngine, ... MiB free
engine ready in ...s
```

Per request, LTEngine logs languages, token counts and queue/inference/total time. It never logs message text.

## API

LibreTranslate-compatible: `POST /translate`, `POST /detect`, `GET /languages`, `GET /frontend/settings`. Additionally:

 * `GET /health`: liveness (the process is up; reports `loading`/`ready`/`failed`)
 * `GET /health/ready`: `200` once the model is loaded, `503` before. Used by the Docker health check (`ltengine --healthcheck`)
 * `GET /metrics`: Prometheus metrics (requests, errors, busy rejections, cancellations, tokens, queue/inference/request latency histograms, queue depth, model loaded)

Errors: `400` invalid request or language, `422` the model produced an empty or over-long translation (deterministic, so don't retry), `503` loading or busy (retry later).

### Simple

Request:

```javascript
const res = await fetch("http://0.0.0.0:5050/translate", {
  method: "POST",
  body: JSON.stringify({
    q: "Hello!",
    source: "en",
    target: "es",
  }),
  headers: { "Content-Type": "application/json" },
});

console.log(await res.json());
```

Response:

```javascript
{
    "translatedText": "¡Hola!"
}
```

List of language codes: http://0.0.0.0:5050/languages

### Auto Detect Language

Request:

```javascript
const res = await fetch("http://0.0.0.0:5050/translate", {
  method: "POST",
  body: JSON.stringify({
    q: "Ciao!",
    source: "auto",
    target: "en",
  }),
  headers: { "Content-Type": "application/json" },
});

console.log(await res.json());
```

Response:

```javascript
{
    "detectedLanguage": {
        "confidence": 83,
        "language": "it"
    },
    "translatedText": "Bye!"
}
```

## Language Bindings

You can use the LTEngine API using the following bindings:

- Rust: <https://github.com/DefunctLizard/libretranslate-rs>
- Node.js: <https://github.com/franciscop/translate>
- TypeScript: <https://github.com/tderflinger/libretranslate-ts>
- .Net: <https://github.com/sigaloid/LibreTranslate.Net>
- Go: <https://github.com/SnakeSel/libretranslate>
- Python: <https://github.com/argosopentech/LibreTranslate-py>
- PHP: <https://github.com/jefs42/libretranslate>
- C++: <https://github.com/argosopentech/LibreTranslate-cpp>
- Swift: <https://github.com/wacumov/libretranslate>
- Unix: <https://github.com/argosopentech/LibreTranslate-sh>
- Shell: <https://github.com/Hayao0819/Hayao-Tools/tree/master/libretranslate-sh>
- Java: <https://github.com/suuft/libretranslate-java>
- Ruby: <https://github.com/noesya/libretranslate>
- R: <https://github.com/myanesp/libretranslateR>

## Roadmap

 - [x] Serve requests without blocking the HTTP server (bounded queue, one persistent inference context)
 - [x] Cancel inference (stop generating tokens) when HTTP connections are aborted by clients
 - [x] Native TranslateGemma support
 - [ ] Add support for `/translate_file` (ability to translate files).
 - [ ] Add support for sentence splitting. Currently text is sent to the LLM as-is, but longer texts (like documents) should be split into chunks, translated and merged back.
 - [ ] Better language detection for short texts (port [LexiLang](https://github.com/LibreTranslate/LexiLang) to Rust)
 - [ ] Batched inference of concurrent requests in one context (continuous batching)
 - [ ] Create comparative benchmarks between LTEngine and proprietary software.
 - [ ] Add support for command line inference (run `./ltengine translate` as a command line app separate from `./ltengine server`)
 - [ ] Make ltengine available as a library, possibly creating bindings for other languages like Python.
 - [x] Automated builds / CI
 - [ ] Your ideas? We welcome contributions.

## Contributing

We welcome contributions! Just open a pull request.

## Credits

This work is largely possible thanks [llama-cpp-rs](https://github.com/utilityai/llama-cpp-rs) which provide the Rust bindings to [llama.cpp](https://github.com/ggml-org/llama.cpp).

## License

[GNU Affero General Public License v3](https://www.gnu.org/licenses/agpl-3.0.en.html)

## Trademark

See [Trademark Guidelines](https://github.com/LibreTranslate/LibreTranslate/blob/main/TRADEMARK.md)
