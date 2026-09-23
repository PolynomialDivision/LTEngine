//! Model loading, GPU offload planning and context creation.
//!
//! Strategy for small GPUs (e.g. a 4 GB GTX 1650):
//! 1. llama.cpp's own `common_fit_params` runs a dry-run allocation of model,
//!    KV cache and compute buffers with our exact context parameters and picks
//!    the largest offload that leaves `vram_margin` MiB free.
//! 2. The persistent inference context is created and warmed up with a
//!    worst-case batch at startup, so every GPU allocation (weights, KV cache,
//!    compute buffers, cuBLAS/pool scratch) happens before we report ready.
//! 3. If any of that still fails, retry with fewer GPU layers.

use std::ffi::CString;
use std::num::NonZeroU32;
use std::path::Path;
use std::pin::pin;

use anyhow::{Context, Result, anyhow, bail};
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::context::params::{KvCacheType, LlamaContextParams};
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::{LlamaBackendDevice, LlamaBackendDeviceType, list_llama_ggml_backend_devices};
use tracing::{info, warn};

const MIB: usize = 1024 * 1024;

/// How many layers to offload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuLayers {
    /// Let llama.cpp fit as many layers as free VRAM allows.
    Auto,
    /// Offload exactly this many (a value above the layer count means all).
    Count(u32),
}

impl std::str::FromStr for GpuLayers {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "auto" | "" => Ok(GpuLayers::Auto),
            "all" => Ok(GpuLayers::Count(u32::MAX)),
            n => n
                .parse()
                .map(GpuLayers::Count)
                .map_err(|_| format!("expected 'auto', 'all' or a number, got '{s}'")),
        }
    }
}

/// Settings that affect model placement and context allocation.
#[derive(Debug, Clone)]
pub struct LoadConfig {
    pub cpu: bool,
    pub allow_cpu_fallback: bool,
    pub gpu_layers: GpuLayers,
    pub vram_margin_mib: Option<usize>,
    pub ctx_size: u32,
    pub batch_size: u32,
    pub ubatch_size: Option<u32>,
    pub threads: Option<u32>,
    pub kv_cache_type: KvCacheType,
    pub flash_attn: Option<bool>,
}

/// Decisions made before loading the model.
#[derive(Debug, Clone)]
pub struct Plan {
    pub use_gpu: bool,
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_ubatch: u32,
    pub threads: u32,
    gpu_total_mib: usize,
}

/// Placement of the loaded model, for logging.
#[derive(Debug, Clone, Copy)]
pub struct Offload {
    /// Layers on the GPU; `None` means all of them.
    pub gpu_layers: Option<u32>,
    /// Number of tensor groups llama.cpp's fitter kept in system RAM.
    pub cpu_overrides: usize,
}

fn gpu_devices() -> Vec<LlamaBackendDevice> {
    list_llama_ggml_backend_devices()
        .into_iter()
        .filter(|d| {
            matches!(
                d.device_type,
                LlamaBackendDeviceType::Gpu | LlamaBackendDeviceType::IntegratedGpu
            )
        })
        .collect()
}

/// Free/total VRAM in MiB summed over all GPUs.
pub fn vram_mib() -> Option<(usize, usize)> {
    let devices = gpu_devices();
    (!devices.is_empty()).then(|| {
        devices.iter().fold((0, 0), |(f, t), d| {
            (f + d.memory_free / MIB, t + d.memory_total / MIB)
        })
    })
}

/// Decide CPU vs GPU and batch sizes. Fails if a GPU was expected but none is
/// usable, unless CPU fallback was explicitly allowed.
pub fn plan(backend: &LlamaBackend, cfg: &LoadConfig) -> Result<Plan> {
    let gpu_build = cfg!(any(feature = "cuda", feature = "vulkan", feature = "metal"));
    let devices = gpu_devices();
    for d in &devices {
        info!(
            backend = %d.backend,
            "GPU {}: {} ({} MiB free / {} MiB total)",
            d.index, d.description, d.memory_free / MIB, d.memory_total / MIB
        );
    }

    let use_gpu = if cfg.cpu {
        info!("CPU mode requested, GPU offload disabled");
        false
    } else if !gpu_build {
        warn!(
            "this binary was built without GPU support (enable the cuda/vulkan/metal feature); running on CPU"
        );
        false
    } else if devices.is_empty() || !backend.supports_gpu_offload() {
        let hint = "no usable GPU found. For Docker, run the container with the NVIDIA runtime \
                    (`--gpus all` or a compose GPU reservation) and check `nvidia-smi` inside the container";
        if cfg.allow_cpu_fallback {
            warn!("{hint}; falling back to CPU because CPU fallback is allowed");
            false
        } else {
            bail!(
                "{hint}. Set LTENGINE_CPU=true or LTENGINE_ALLOW_CPU_FALLBACK=true to run on the CPU instead"
            );
        }
    } else {
        true
    };

    let gpu_total_mib = if use_gpu {
        devices
            .iter()
            .map(|d| d.memory_total / MIB)
            .min()
            .unwrap_or(0)
    } else {
        0
    };

    // The logits scratch buffer alone is n_ubatch * n_vocab * 4 bytes; with
    // Gemma's 262k vocabulary that is 128 MiB at 128 and 512 MiB at 512.
    let n_ubatch = cfg
        .ubatch_size
        .unwrap_or(if use_gpu && gpu_total_mib < 6 * 1024 {
            128
        } else {
            512
        });
    let n_ctx = cfg.ctx_size.max(64);
    let n_batch = cfg.batch_size.clamp(n_ubatch.min(n_ctx), n_ctx);
    let n_ubatch = n_ubatch.clamp(1, n_batch);

    let threads = cfg.threads.unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map_or(4, |n| u32::try_from(n.get()).unwrap_or(4))
            .clamp(1, 8)
    });

    Ok(Plan {
        use_gpu,
        n_ctx,
        n_batch,
        n_ubatch,
        threads,
        gpu_total_mib,
    })
}

/// Context parameters for the persistent inference context.
pub fn context_params(cfg: &LoadConfig, plan: &Plan) -> LlamaContextParams {
    let threads = i32::try_from(plan.threads).unwrap_or(4);
    let flash_attn = match cfg.flash_attn {
        None => llama_cpp_sys_2::LLAMA_FLASH_ATTN_TYPE_AUTO,
        Some(true) => llama_cpp_sys_2::LLAMA_FLASH_ATTN_TYPE_ENABLED,
        Some(false) => llama_cpp_sys_2::LLAMA_FLASH_ATTN_TYPE_DISABLED,
    };
    LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(plan.n_ctx))
        .with_n_batch(plan.n_batch)
        .with_n_ubatch(plan.n_ubatch)
        .with_n_seq_max(1)
        .with_n_threads(threads)
        .with_n_threads_batch(threads)
        .with_type_k(cfg.kv_cache_type)
        .with_type_v(cfg.kv_cache_type)
        .with_flash_attention_policy(flash_attn)
        // Sliding-window layers (5 of every 6 in Gemma 3) only need a window
        // sized KV cache. llama.cpp defaults to a full-size cache for them,
        // which is only useful for prompt caching; we reset per request.
        .with_swa_full(false)
        .with_no_perf(true)
}

/// Read only the vocabulary and hyperparameters (cheap) to learn the layer count.
pub fn probe_layer_count(backend: &LlamaBackend, path: &Path) -> Result<u32> {
    let model = LlamaModel::load_from_file(
        backend,
        path,
        &LlamaModelParams::default().with_vocab_only(true),
    )
    .with_context(|| format!("unable to read model metadata from {}", path.display()))?;
    Ok(model.n_layer())
}

/// Load the model. `layers == None` lets llama.cpp fit the offload to free VRAM.
pub fn load_model(
    backend: &LlamaBackend,
    path: &Path,
    cfg: &LoadConfig,
    plan: &Plan,
    layers: Option<u32>,
) -> Result<(LlamaModel, Offload)> {
    let mut params = pin!(LlamaModelParams::default());

    let offload = if !plan.use_gpu {
        params.set(LlamaModelParams::default().with_n_gpu_layers(0));
        Offload {
            gpu_layers: Some(0),
            cpu_overrides: 0,
        }
    } else if let Some(n) = layers {
        params.set(LlamaModelParams::default().with_n_gpu_layers(n));
        Offload {
            gpu_layers: Some(n),
            cpu_overrides: 0,
        }
    } else {
        let margin_mib = cfg
            .vram_margin_mib
            .unwrap_or(if plan.gpu_total_mib < 8 * 1024 {
                512
            } else {
                1024
            });
        let mut margins = vec![margin_mib * MIB; llama_cpp_2::max_devices().max(1)];
        let path_c = CString::new(path.to_str().context("model path is not valid UTF-8")?)?;
        let mut cparams = context_params(cfg, plan);
        info!(margin_mib, "fitting model to free VRAM");
        match params.as_mut().fit_params(
            &path_c,
            &mut cparams,
            &mut margins,
            plan.n_ctx,
            llama_cpp_sys_2::GGML_LOG_LEVEL_INFO,
        ) {
            Ok(_) => {}
            Err(err) => {
                warn!(
                    "llama.cpp could not fit the model into free VRAM ({err}); trying full offload"
                );
                params.set(LlamaModelParams::default());
            }
        }
        let n = params.n_gpu_layers();
        Offload {
            gpu_layers: u32::try_from(n).ok(),
            cpu_overrides: params.tensor_buft_override_patterns().len(),
        }
    };

    let model = LlamaModel::load_from_file(backend, path, &params)
        .map_err(|e| anyhow!("{e}"))
        .with_context(|| format!("unable to load model {}", path.display()))?;
    Ok((model, offload))
}

/// Create the inference context and run a worst-case warmup batch so that
/// all lazily allocated GPU memory is claimed now rather than on the first
/// real request.
pub fn create_context<'m>(
    backend: &LlamaBackend,
    model: &'m LlamaModel,
    cfg: &LoadConfig,
    plan: &Plan,
) -> Result<LlamaContext<'m>> {
    let mut ctx = model
        .new_context(backend, context_params(cfg, plan))
        .context("unable to create the llama context")?;

    let n = usize::try_from(plan.n_ubatch.min(plan.n_ctx / 2).max(1))?;
    let token = model.token_bos();
    let mut batch = LlamaBatch::new(n, 1);
    for pos in 0..n {
        batch.add(token, i32::try_from(pos)?, &[0], pos + 1 == n)?;
    }
    ctx.decode(&mut batch)
        .context("warmup prompt batch failed")?;
    batch.clear();
    batch.add(token, i32::try_from(n)?, &[0], true)?;
    ctx.decode(&mut batch)
        .context("warmup generation step failed")?;
    ctx.clear_kv_cache();
    Ok(ctx)
}

/// Next GPU layer count to try after an allocation failure, or `None` if
/// there is nothing left to reduce.
pub fn reduce_layers(current: Option<u32>, n_layer: u32) -> Option<u32> {
    let current = current.unwrap_or(u32::MAX).min(n_layer + 1);
    (current > 0).then(|| current - (current / 8).max(1))
}

/// Human-readable GGUF file type (quantization).
pub fn file_type_name(file_type: &str) -> String {
    let name = match file_type.trim() {
        "0" => "F32",
        "1" => "F16",
        "2" => "Q4_0",
        "3" => "Q4_1",
        "7" => "Q8_0",
        "8" => "Q5_0",
        "9" => "Q5_1",
        "10" => "Q2_K",
        "11" => "Q3_K_S",
        "12" => "Q3_K_M",
        "13" => "Q3_K_L",
        "14" => "Q4_K_S",
        "15" => "Q4_K_M",
        "16" => "Q5_K_S",
        "17" => "Q5_K_M",
        "18" => "Q6_K",
        "19" => "IQ2_XXS",
        "20" => "IQ2_XS",
        "21" => "Q2_K_S",
        "22" => "IQ3_XS",
        "23" => "IQ3_XXS",
        "24" => "IQ1_S",
        "25" => "IQ4_NL",
        "26" => "IQ3_S",
        "27" => "IQ3_M",
        "28" => "IQ2_S",
        "29" => "IQ2_M",
        "30" => "IQ4_XS",
        "31" => "IQ1_M",
        "32" => "BF16",
        other => return format!("type {other}"),
    };
    name.to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_layers_parse() {
        assert_eq!("auto".parse::<GpuLayers>().unwrap(), GpuLayers::Auto);
        assert_eq!(
            "all".parse::<GpuLayers>().unwrap(),
            GpuLayers::Count(u32::MAX)
        );
        assert_eq!("20".parse::<GpuLayers>().unwrap(), GpuLayers::Count(20));
        assert!("many".parse::<GpuLayers>().is_err());
    }

    #[test]
    fn layer_reduction_converges_to_zero() {
        let mut layers = None;
        let mut steps = 0;
        while let Some(next) = reduce_layers(layers, 34) {
            assert!(layers.is_none_or(|l| next < l));
            layers = Some(next);
            steps += 1;
        }
        assert_eq!(layers, Some(0));
        assert!(steps < 40);
    }
}
