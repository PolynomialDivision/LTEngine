//! Inference engine.
//!
//! A single dedicated thread owns the llama.cpp backend, the model and one
//! persistent context. Requests reach it through a bounded queue:
//!
//! * the context (KV cache + compute buffers) is allocated and warmed up once
//!   at startup, so a request can never trigger a fresh GPU allocation;
//! * inference is serialized, which is what a single small GPU wants anyway.
//!   Parallel contexts would duplicate the KV cache and compute buffers
//!   without batching work on the GPU;
//! * when the queue is full, requests are rejected immediately with
//!   "Server busy" instead of piling up;
//! * callers that go away (HTTP client disconnect, timeout) flag their job
//!   as cancelled, and the worker stops generating at the next token.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, TrySendError, sync_channel};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use anyhow::{Context, anyhow};
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::{AddBos, LlamaChatMessage, LlamaChatTemplate, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use tokio::sync::oneshot;
use tracing::{debug, error, info, warn};

use crate::formatting::clean_model_output;
use crate::languages::Language;
use crate::loader::{self, GpuLayers, LoadConfig};
use crate::metrics::Metrics;
use crate::prompt::{self, PromptFormat};

/// Give up on GPU fallback after this many load attempts.
const MAX_LOAD_ATTEMPTS: usize = 24;

/// Engine configuration.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// `--model` name to download, unless `model_file` is set.
    pub model: String,
    /// Local GGUF path.
    pub model_file: String,
    pub load: LoadConfig,
    /// Absolute cap on generated tokens per request.
    pub max_new_tokens: u32,
    /// Requests that may wait for the worker while it is busy.
    pub queue_size: usize,
    /// Requests that waited longer than this are rejected as busy.
    pub queue_timeout: Duration,
}

/// Errors surfaced to API callers.
#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("Model is loading, please try again later")]
    Loading,
    #[error("Translation engine unavailable: {0}")]
    Unavailable(String),
    #[error("Server busy, please try again later")]
    QueueFull,
    #[error("Server busy, please try again later (queue timeout)")]
    QueueTimeout,
    #[error("Request cancelled")]
    Cancelled,
    #[error("{0}")]
    UnsupportedLanguage(String),
    #[error("Input too long: {tokens} prompt tokens, context size is {n_ctx}")]
    InputTooLong { tokens: usize, n_ctx: u32 },
    #[error("Translation exceeded the output limit of {0} tokens")]
    OutputLimit(usize),
    #[error("Model produced an empty translation")]
    EmptyOutput,
    #[error("Translation failed")]
    Internal(#[source] anyhow::Error),
}

impl EngineError {
    /// HTTP status for this error. 4xx errors are deterministic (greedy
    /// decoding gives the same result on retry); 503 means "retry later".
    pub fn status(&self) -> u16 {
        match self {
            EngineError::Loading
            | EngineError::Unavailable(_)
            | EngineError::QueueFull
            | EngineError::QueueTimeout => 503,
            EngineError::Cancelled => 499,
            EngineError::UnsupportedLanguage(_) | EngineError::InputTooLong { .. } => 400,
            EngineError::OutputLimit(_) | EngineError::EmptyOutput => 422,
            EngineError::Internal(_) => 500,
        }
    }
}

/// A translation request with resolved languages.
#[derive(Debug)]
pub struct Request {
    /// `None` only for generic chat models when auto-detection failed.
    pub source: Option<&'static Language>,
    pub target: &'static Language,
    pub format: String,
    pub text: String,
}

/// A completed translation.
#[derive(Debug)]
pub struct Output {
    pub text: String,
    pub input_tokens: usize,
    pub output_tokens: usize,
    pub queue_time: Duration,
    pub inference_time: Duration,
}

/// Read-only facts about the loaded model.
#[derive(Debug)]
pub struct ModelInfo {
    pub prompt: PromptFormat,
    pub n_ctx: u32,
}

enum Status {
    Loading,
    Ready(Arc<ModelInfo>),
    Failed(String),
}

struct Shared {
    status: RwLock<Status>,
    metrics: Arc<Metrics>,
}

impl Shared {
    fn set_status(&self, status: Status) {
        *self
            .status
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = status;
    }
}

struct Job {
    request: Request,
    cancelled: Arc<AtomicBool>,
    enqueued_at: Instant,
    reply: oneshot::Sender<Result<Output, EngineError>>,
}

/// Sets the cancellation flag unless disarmed, e.g. when the awaiting HTTP
/// handler future is dropped because the client disconnected.
struct CancelOnDrop(Option<Arc<AtomicBool>>);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        if let Some(flag) = self.0.take() {
            flag.store(true, Ordering::Release);
        }
    }
}

/// Handle to the inference engine.
pub struct Engine {
    shared: Arc<Shared>,
    jobs: SyncSender<Job>,
}

impl std::fmt::Debug for Engine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Engine").finish_non_exhaustive()
    }
}

impl Engine {
    /// Start loading the model on a dedicated thread and return immediately.
    /// If loading fails, the process exits so the container restart policy
    /// can take over.
    pub fn start(config: EngineConfig, metrics: Arc<Metrics>) -> Arc<Engine> {
        let (jobs, rx) = sync_channel(config.queue_size);
        let shared = Arc::new(Shared {
            status: RwLock::new(Status::Loading),
            metrics,
        });
        let thread_shared = shared.clone();
        std::thread::Builder::new()
            .name("llama-engine".into())
            .spawn(move || {
                if let Err(err) = run(&config, &thread_shared, rx) {
                    error!("inference engine failed: {err:#}");
                    thread_shared.set_status(Status::Failed(format!("{err:#}")));
                    thread_shared
                        .metrics
                        .model_loaded
                        .store(false, Ordering::Relaxed);
                    std::process::exit(1);
                }
                info!("inference engine stopped");
            })
            .expect("failed to spawn the inference thread");
        Arc::new(Engine { shared, jobs })
    }

    /// Loaded model details, or why they are not available.
    pub fn model(&self) -> Result<Arc<ModelInfo>, EngineError> {
        match &*self
            .shared
            .status
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
        {
            Status::Loading => Err(EngineError::Loading),
            Status::Ready(info) => Ok(info.clone()),
            Status::Failed(reason) => Err(EngineError::Unavailable(reason.clone())),
        }
    }

    /// `"loading"`, `"ready"` or `"failed"`.
    pub fn state(&self) -> &'static str {
        match &*self
            .shared
            .status
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
        {
            Status::Loading => "loading",
            Status::Ready(_) => "ready",
            Status::Failed(_) => "failed",
        }
    }

    /// Queue a translation and wait for the result. Dropping the returned
    /// future cancels the job.
    pub async fn translate(&self, request: Request) -> Result<Output, EngineError> {
        self.model()?;
        let metrics = &self.shared.metrics;
        let (reply, rx) = oneshot::channel();
        let cancelled = Arc::new(AtomicBool::new(false));
        let mut guard = CancelOnDrop(Some(cancelled.clone()));

        metrics.queued.fetch_add(1, Ordering::Relaxed);
        let job = Job {
            request,
            cancelled,
            enqueued_at: Instant::now(),
            reply,
        };
        if let Err(err) = self.jobs.try_send(job) {
            metrics.queued.fetch_sub(1, Ordering::Relaxed);
            guard.0 = None;
            return Err(match err {
                TrySendError::Full(_) => EngineError::QueueFull,
                TrySendError::Disconnected(_) => {
                    EngineError::Unavailable("inference worker stopped".into())
                }
            });
        }

        let result = rx
            .await
            .map_err(|_| EngineError::Unavailable("inference worker stopped".into()))?;
        guard.0 = None;
        result
    }
}

/// Load everything, then serve jobs until the engine handle is dropped.
fn run(config: &EngineConfig, shared: &Shared, rx: Receiver<Job>) -> anyhow::Result<()> {
    let model_path = crate::models::load_model(&config.model, &config.model_file)?;
    info!(path = %model_path.display(), "model file");

    let backend = LlamaBackend::init().context("unable to initialise the llama.cpp backend")?;
    let plan = loader::plan(&backend, &config.load)?;
    info!(
        gpu = plan.use_gpu,
        ctx_size = plan.n_ctx,
        batch_size = plan.n_batch,
        ubatch_size = plan.n_ubatch,
        threads = plan.threads,
        kv_cache = ?config.load.kv_cache_type,
        flash_attn = config.load.flash_attn.map_or("auto", |on| if on { "on" } else { "off" }),
        max_new_tokens = config.max_new_tokens,
        queue_size = config.queue_size,
        "inference settings"
    );

    let n_layer = loader::probe_layer_count(&backend, &model_path)?;
    let vram_before = loader::vram_mib();
    let mut layers = match (plan.use_gpu, config.load.gpu_layers) {
        (false, _) => Some(0),
        (true, GpuLayers::Auto) => None,
        (true, GpuLayers::Count(n)) => Some(n),
    };

    for attempt in 1..=MAX_LOAD_ATTEMPTS {
        let started = Instant::now();
        let load = loader::load_model(&backend, &model_path, &config.load, &plan, layers);
        let failure = match load {
            Ok((model, offload)) => {
                let on_gpu = offload
                    .gpu_layers
                    .map_or(n_layer + 1, |n| n.min(n_layer + 1));
                match loader::create_context(&backend, &model, &config.load, &plan) {
                    Ok(ctx) => {
                        let info = describe_model(&model, ctx.n_ctx());
                        log_ready(
                            &plan,
                            n_layer,
                            on_gpu,
                            offload.cpu_overrides,
                            vram_before,
                            started.elapsed(),
                        );
                        shared.metrics.model_loaded.store(true, Ordering::Relaxed);
                        shared.set_status(Status::Ready(info.clone()));
                        let worker = Worker::new(&model, ctx, info, config, &shared.metrics);
                        worker.serve(&rx);
                        shared.metrics.model_loaded.store(false, Ordering::Relaxed);
                        return Ok(());
                    }
                    Err(err) => {
                        layers = Some(on_gpu);
                        err
                    }
                }
            }
            Err(err) => err,
        };

        if !plan.use_gpu {
            return Err(failure);
        }
        let Some(next) = loader::reduce_layers(layers, n_layer)
            .filter(|&n| n > 0 || config.load.allow_cpu_fallback)
        else {
            return Err(failure.context("model does not fit on the GPU even with minimal offload"));
        };
        warn!(
            attempt,
            "GPU allocation failed ({failure:#}); retrying with {next} GPU layers"
        );
        layers = Some(next);
    }
    Err(anyhow!(
        "giving up after {MAX_LOAD_ATTEMPTS} attempts to load the model"
    ))
}

fn describe_model(model: &LlamaModel, n_ctx: u32) -> Arc<ModelInfo> {
    let meta = |key: &str| model.meta_val_str(key).ok();
    let architecture = meta("general.architecture");
    let template = model
        .chat_template(None)
        .ok()
        .and_then(|t| t.to_string().ok());
    let prompt = PromptFormat::detect(architecture.as_deref(), template.as_deref());

    info!(
        name = meta("general.name").unwrap_or_default(),
        architecture = architecture.unwrap_or_default(),
        quantization = meta("general.file_type")
            .map(|t| loader::file_type_name(&t))
            .unwrap_or_default(),
        params_b = format!("{:.2}", model.n_params() as f64 / 1e9),
        size_mib = model.size() / (1024 * 1024),
        layers = model.n_layer(),
        n_ctx_train = model.n_ctx_train(),
        prompt_format = prompt.name(),
        "model loaded"
    );
    if matches!(prompt, PromptFormat::TranslateGemma(_)) {
        info!(
            "TranslateGemma detected: using the native translation prompt with explicit language codes"
        );
    }

    Arc::new(ModelInfo { prompt, n_ctx })
}

fn log_ready(
    plan: &loader::Plan,
    n_layer: u32,
    on_gpu: u32,
    cpu_overrides: usize,
    vram_before: Option<(usize, usize)>,
    elapsed: Duration,
) {
    let total_layers = n_layer + 1; // + output layer
    if plan.use_gpu {
        if on_gpu >= total_layers && cpu_overrides == 0 {
            info!("GPU offload: all {total_layers} layers on the GPU");
        } else {
            warn!(
                "GPU offload: {on_gpu}/{total_layers} layers on the GPU ({cpu_overrides} tensor groups kept in RAM); \
                 translations will be slower. Free VRAM or use a smaller quantization for full offload"
            );
        }
        if let (Some((free_before, _)), Some((free_after, total))) =
            (vram_before, loader::vram_mib())
        {
            info!(
                "VRAM: {} MiB used by LTEngine, {free_after} MiB of {total} MiB free",
                free_before.saturating_sub(free_after)
            );
        }
    } else {
        info!("running on the CPU");
    }
    info!("engine ready in {:.1}s", elapsed.as_secs_f32());
}

/// Output token budget: proportional to the input so a short message cannot
/// generate thousands of tokens, but generous enough for language pairs that
/// need more tokens than the source (e.g. EN → UK).
pub fn output_budget(text_tokens: usize, max_new_tokens: u32) -> usize {
    (text_tokens.saturating_mul(3) + 32).min(max_new_tokens as usize)
}

struct Worker<'m> {
    model: &'m LlamaModel,
    ctx: LlamaContext<'m>,
    info: Arc<ModelInfo>,
    chat_template: Option<LlamaChatTemplate>,
    batch: LlamaBatch<'static>,
    n_batch: usize,
    sampler: LlamaSampler,
    max_new_tokens: u32,
    queue_timeout: Duration,
    metrics: &'m Metrics,
}

impl<'m> Worker<'m> {
    fn new(
        model: &'m LlamaModel,
        ctx: LlamaContext<'m>,
        info: Arc<ModelInfo>,
        config: &EngineConfig,
        metrics: &'m Metrics,
    ) -> Self {
        let n_batch = ctx.n_batch() as usize;
        let chat_template = match info.prompt {
            PromptFormat::Chat { .. } => model.chat_template(None).ok(),
            PromptFormat::TranslateGemma(_) => None,
        };
        Worker {
            model,
            ctx,
            info,
            chat_template,
            batch: LlamaBatch::new(n_batch, 1),
            n_batch,
            // Translation wants the single most likely output: greedy decoding
            // is deterministic and needs no sampling chain.
            sampler: LlamaSampler::greedy(),
            max_new_tokens: config.max_new_tokens,
            queue_timeout: config.queue_timeout,
            metrics,
        }
    }

    fn serve(mut self, rx: &Receiver<Job>) {
        while let Ok(job) = rx.recv() {
            self.metrics.queued.fetch_sub(1, Ordering::Relaxed);
            let queue_time = job.enqueued_at.elapsed();
            self.metrics.queue_duration.observe(queue_time);

            let result = if job.cancelled.load(Ordering::Acquire) || job.reply.is_closed() {
                Err(EngineError::Cancelled)
            } else if queue_time > self.queue_timeout {
                Err(EngineError::QueueTimeout)
            } else {
                self.metrics.active.fetch_add(1, Ordering::Relaxed);
                let started = Instant::now();
                let result = catch_unwind(AssertUnwindSafe(|| {
                    self.generate(&job.request, &job.cancelled)
                }))
                .unwrap_or_else(|_| Err(EngineError::Internal(anyhow!("inference panicked"))));
                self.ctx.clear_kv_cache();
                self.metrics.active.fetch_sub(1, Ordering::Relaxed);
                let inference_time = started.elapsed();
                self.metrics.inference_duration.observe(inference_time);
                result.map(|(text, input_tokens, output_tokens)| Output {
                    text,
                    input_tokens,
                    output_tokens,
                    queue_time,
                    inference_time,
                })
            };

            if let Err(err) = &result {
                match err {
                    EngineError::Cancelled => {
                        self.metrics.cancelled.fetch_add(1, Ordering::Relaxed);
                        debug!("job cancelled by client");
                    }
                    EngineError::Internal(e) => error!("inference error: {e:#}"),
                    other => debug!("job rejected: {other}"),
                }
            }
            let _ = job.reply.send(result);
        }
    }

    fn render_prompt(&self, request: &Request) -> Result<String, EngineError> {
        match &self.info.prompt {
            PromptFormat::TranslateGemma(tg) => {
                let source = request.source.ok_or_else(|| {
                    EngineError::UnsupportedLanguage("source language is required".into())
                })?;
                tg.render(source.code, request.target.code, &request.text)
                    .map_err(|e| EngineError::UnsupportedLanguage(e.to_string()))
            }
            PromptFormat::Chat { gemma_fallback } => {
                let chat = prompt::chat_prompt(
                    &request.format,
                    request.source.map(|l| l.name),
                    request.target.name,
                    &request.text,
                );
                let message = LlamaChatMessage::new(
                    "user".into(),
                    format!("{}\n\n{}", chat.system, chat.user),
                )
                .map_err(|e| EngineError::Internal(e.into()))?;
                let applied = self
                    .chat_template
                    .as_ref()
                    .ok_or_else(|| anyhow!("model has no chat template"))
                    .and_then(|t| Ok(self.model.apply_chat_template(t, &[message], true)?));
                match applied {
                    Ok(prompt) => Ok(prompt),
                    Err(_) if *gemma_fallback => Ok(prompt::gemma_fallback(&chat)),
                    Err(err) => Err(EngineError::Internal(
                        err.context("unable to apply the chat template"),
                    )),
                }
            }
        }
    }

    /// Returns (text, prompt tokens, generated tokens).
    fn generate(
        &mut self,
        request: &Request,
        cancelled: &AtomicBool,
    ) -> Result<(String, usize, usize), EngineError> {
        let internal = |e: anyhow::Error| EngineError::Internal(e);
        let prompt = self.render_prompt(request)?;
        let tokens = self
            .model
            .str_to_token(&prompt, AddBos::Always)
            .context("failed to tokenize prompt")
            .map_err(internal)?;
        let text_tokens = self
            .model
            .str_to_token(&request.text, AddBos::Never)
            .map_or(tokens.len(), |t| t.len());

        let n_ctx = self.ctx.n_ctx() as usize;
        let room = n_ctx.saturating_sub(tokens.len());
        // Require room for an output about as long as the input.
        if room < text_tokens + 8 {
            return Err(EngineError::InputTooLong {
                tokens: tokens.len(),
                n_ctx: self.info.n_ctx,
            });
        }
        let max_new = output_budget(text_tokens, self.max_new_tokens).min(room);

        self.ctx.clear_kv_cache();
        let last = tokens.len() - 1;
        for (chunk_index, chunk) in tokens.chunks(self.n_batch).enumerate() {
            if cancelled.load(Ordering::Acquire) {
                return Err(EngineError::Cancelled);
            }
            self.batch.clear();
            for (offset, &token) in chunk.iter().enumerate() {
                let pos = chunk_index * self.n_batch + offset;
                self.batch
                    .add(
                        token,
                        i32::try_from(pos).map_err(|e| internal(e.into()))?,
                        &[0],
                        pos == last,
                    )
                    .map_err(|e| internal(e.into()))?;
            }
            self.ctx
                .decode(&mut self.batch)
                .context("prompt decode failed")
                .map_err(internal)?;
        }

        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut output = String::new();
        let mut logits_index = self.batch.n_tokens() - 1;
        let mut n_out = 0;
        loop {
            if cancelled.load(Ordering::Acquire) {
                return Err(EngineError::Cancelled);
            }
            let token = self.sampler.sample(&self.ctx, logits_index);
            if self.model.is_eog_token(token) {
                break;
            }
            if n_out == max_new {
                warn!(
                    input_tokens = tokens.len(),
                    max_new, "generation hit the output limit"
                );
                return Err(EngineError::OutputLimit(max_new));
            }
            let piece = self
                .model
                .token_to_piece(token, &mut decoder, true, None)
                .context("failed to decode token")
                .map_err(internal)?;
            output.push_str(&piece);
            n_out += 1;

            self.batch.clear();
            let pos = i32::try_from(tokens.len() + n_out - 1).map_err(|e| internal(e.into()))?;
            self.batch
                .add(token, pos, &[0], true)
                .map_err(|e| internal(e.into()))?;
            self.ctx
                .decode(&mut self.batch)
                .context("generation decode failed")
                .map_err(internal)?;
            logits_index = 0;
        }

        self.metrics
            .input_tokens
            .fetch_add(tokens.len() as u64, Ordering::Relaxed);
        self.metrics
            .output_tokens
            .fetch_add(n_out as u64, Ordering::Relaxed);

        let text = clean_model_output(&output);
        if text.is_empty() {
            return Err(EngineError::EmptyOutput);
        }
        Ok((text.to_owned(), tokens.len(), n_out))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_budget_scales_with_input_and_is_capped() {
        assert_eq!(output_budget(1, 2048), 35);
        assert_eq!(output_budget(100, 2048), 332);
        assert_eq!(output_budget(5000, 2048), 2048);
    }

    #[test]
    fn dropping_cancel_guard_sets_flag() {
        let flag = Arc::new(AtomicBool::new(false));
        drop(CancelOnDrop(Some(flag.clone())));
        assert!(flag.load(Ordering::Acquire));

        let flag = Arc::new(AtomicBool::new(false));
        let mut guard = CancelOnDrop(Some(flag.clone()));
        guard.0 = None;
        drop(guard);
        assert!(!flag.load(Ordering::Acquire));
    }

    #[test]
    fn busy_errors_mention_server_busy() {
        // The Matrix bot keys its retry classification on this phrase.
        assert!(EngineError::QueueFull.to_string().contains("Server busy"));
        assert!(
            EngineError::QueueTimeout
                .to_string()
                .contains("Server busy")
        );
    }
}
