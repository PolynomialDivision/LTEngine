//! LTEngine: local LLM machine translation with a LibreTranslate-compatible API.

use std::io::{Read, Write};
use std::net::{Ipv4Addr, SocketAddr, TcpStream};
use std::sync::Arc;
use std::time::Duration;

use actix_web::{App, HttpServer, web};
use actix_web_static_files::ResourceFiles;
use clap::Parser;
use llama_cpp_2::context::params::KvCacheType;
use llama_cpp_2::{LogOptions, send_logs_to_tracing};
use tracing::info;
use tracing_subscriber::EnvFilter;

mod api;
mod banner;
mod cache;
mod engine;
mod error_response;
mod formatting;
mod languages;
mod loader;
mod markup;
mod metrics;
mod models;
mod prompt;

use engine::{Engine, EngineConfig};
use loader::{GpuLayers, LoadConfig};
use models::{DEFAULT_MODEL, MODELS};

include!(concat!(env!("OUT_DIR"), "/generated.rs"));

/// Every option can also be set with the environment variable shown in
/// `--help` (LTENGINE_*), which is the intended way to configure containers.
#[derive(Parser, Debug, Clone)]
#[command(version, about = "Local AI machine translation API", long_about = None)]
pub struct Args {
    /// Hostname to bind to
    #[arg(long, env = "LTENGINE_HOST", default_value = "0.0.0.0")]
    host: String,

    /// Port to bind to
    #[arg(short, long, env = "LTENGINE_PORT", default_value_t = 5050)]
    port: u16,

    /// Character limit for translation requests
    #[arg(long, env = "LTENGINE_CHAR_LIMIT", default_value_t = 5000)]
    char_limit: usize,

    /// Model to download and use (ignored when --model-file is set)
    #[arg(short = 'm', long, env = "LTENGINE_MODEL", value_parser = MODELS.keys().copied().collect::<Vec<_>>(), default_value = DEFAULT_MODEL)]
    model: String,

    /// Path to a local .gguf model file
    #[arg(long, env = "LTENGINE_MODEL_FILE", default_value = "")]
    model_file: String,

    /// Require this API key for requests
    #[arg(
        long,
        env = "LTENGINE_API_KEY",
        default_value = "",
        hide_env_values = true
    )]
    api_key: String,

    /// Use the CPU only
    #[arg(long, env = "LTENGINE_CPU")]
    cpu: bool,

    /// Run on the CPU if no GPU is usable instead of refusing to start
    #[arg(long, env = "LTENGINE_ALLOW_CPU_FALLBACK")]
    allow_cpu_fallback: bool,

    /// GPU layers to offload: "auto" (fit to free VRAM), "all" or a number
    #[arg(long, env = "LTENGINE_GPU_LAYERS", default_value = "auto")]
    gpu_layers: GpuLayers,

    /// VRAM in MiB to leave free when fitting the model [default: 512 below 8 GiB, else 1024]
    #[arg(long, env = "LTENGINE_VRAM_MARGIN")]
    vram_margin: Option<usize>,

    /// Context size in tokens (prompt + translation)
    #[arg(long, env = "LTENGINE_CTX_SIZE", default_value_t = 4096)]
    ctx_size: u32,

    /// Maximum tokens submitted per decode call
    #[arg(long, env = "LTENGINE_BATCH_SIZE", default_value_t = 512)]
    batch_size: u32,

    /// Physical micro-batch size [default: 128 on GPUs below 6 GiB, else 512]
    #[arg(long, env = "LTENGINE_UBATCH_SIZE")]
    ubatch_size: Option<u32>,

    /// CPU threads [default: available cores, at most 8]
    #[arg(long, env = "LTENGINE_THREADS")]
    threads: Option<u32>,

    /// KV cache type
    #[arg(long, env = "LTENGINE_KV_CACHE_TYPE", default_value = "f16", value_parser = ["f16", "q8_0"])]
    kv_cache_type: String,

    /// Flash attention
    #[arg(long, env = "LTENGINE_FLASH_ATTN", default_value = "auto", value_parser = ["auto", "on", "off"])]
    flash_attn: String,

    /// Hard cap on generated tokens per translation (the effective limit also scales with input length)
    #[arg(long, env = "LTENGINE_MAX_NEW_TOKENS", default_value_t = 2048)]
    max_new_tokens: u32,

    /// Requests that may wait while a translation is running; more are rejected with 503
    #[arg(long, env = "LTENGINE_QUEUE_SIZE", default_value_t = 16)]
    queue_size: usize,

    /// Seconds a request may wait in the queue before it is rejected with 503
    #[arg(long, env = "LTENGINE_QUEUE_TIMEOUT", default_value_t = 60)]
    queue_timeout: u64,

    /// Cache up to this many translations in memory (0 = off; keeps message text in RAM)
    #[arg(long, env = "LTENGINE_CACHE_SIZE", default_value_t = 0)]
    cache_size: usize,

    /// Enable verbose logging, including llama.cpp's own logs
    #[arg(short = 'v', long, env = "LTENGINE_VERBOSE")]
    verbose: bool,

    /// Check whether a local instance is ready and exit (for container health checks)
    #[arg(long, hide = true)]
    healthcheck: bool,
}

impl Args {
    fn engine_config(&self) -> EngineConfig {
        EngineConfig {
            model: self.model.clone(),
            model_file: self.model_file.clone(),
            load: LoadConfig {
                cpu: self.cpu,
                allow_cpu_fallback: self.allow_cpu_fallback,
                gpu_layers: self.gpu_layers,
                vram_margin_mib: self.vram_margin,
                ctx_size: self.ctx_size,
                batch_size: self.batch_size,
                ubatch_size: self.ubatch_size,
                threads: self.threads,
                kv_cache_type: if self.kv_cache_type == "q8_0" {
                    KvCacheType::Q8_0
                } else {
                    KvCacheType::F16
                },
                flash_attn: match self.flash_attn.as_str() {
                    "on" => Some(true),
                    "off" => Some(false),
                    _ => None,
                },
            },
            max_new_tokens: self.max_new_tokens.max(1),
            queue_size: self.queue_size,
            queue_timeout: Duration::from_secs(self.queue_timeout),
        }
    }
}

fn init_logging(verbose: bool) {
    let default = if verbose {
        "debug,llama_cpp_2=info"
    } else {
        "info,llama_cpp_2=warn"
    };
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(default));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .with_ansi(std::io::IsTerminal::is_terminal(&std::io::stdout()))
        .init();
    // llama.cpp/ggml logs go through tracing and the filter above.
    send_logs_to_tracing(LogOptions::default());
}

/// Exit 0 if the local server reports ready.
fn healthcheck(port: u16) -> i32 {
    let addr = SocketAddr::from((Ipv4Addr::LOCALHOST, port));
    let check = || -> std::io::Result<bool> {
        let mut stream = TcpStream::connect_timeout(&addr, Duration::from_secs(3))?;
        stream.set_read_timeout(Some(Duration::from_secs(3)))?;
        stream.write_all(b"GET /health/ready HTTP/1.0\r\nHost: localhost\r\n\r\n")?;
        let mut response = String::new();
        stream.read_to_string(&mut response)?;
        Ok(response.starts_with("HTTP/1.1 200") || response.starts_with("HTTP/1.0 200"))
    };
    i32::from(!check().unwrap_or(false))
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let args = Arc::new(Args::parse());
    if args.healthcheck {
        std::process::exit(healthcheck(args.port));
    }
    init_logging(args.verbose);
    banner::print_banner();
    info!(version = env!("CARGO_PKG_VERSION"), "starting LTEngine");

    // The HTTP server comes up while the model downloads/loads so that health
    // checks and clients get a clear "loading" answer instead of connection errors.
    let metrics = Arc::new(metrics::Metrics::default());
    let engine = Engine::start(args.engine_config(), metrics.clone());
    let state = web::Data::new(api::AppState {
        args: args.clone(),
        engine,
        metrics,
        cache: cache::TranslationCache::new(args.cache_size),
    });

    let workers = std::thread::available_parallelism().map_or(2, |n| n.get().clamp(2, 4));
    let server = HttpServer::new(move || {
        App::new()
            .app_data(state.clone())
            .app_data(web::JsonConfig::default().limit(256 * 1024))
            .app_data(web::FormConfig::default().limit(256 * 1024))
            .configure(api::configure)
            .service(ResourceFiles::new("/", generate()))
    })
    .on_connect(api::capture_peer_socket)
    .workers(workers)
    .shutdown_timeout(10)
    .bind((args.host.as_str(), args.port))?
    .run();

    info!("listening on http://{}:{}", args.host, args.port);
    server.await
}
