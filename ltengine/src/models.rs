use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;
use std::time::{Duration, SystemTime};

use anyhow::{Context, Result, anyhow};
use hf_hub::Cache;
use hf_hub::api::sync::ApiBuilder;
use tracing::{info, warn};

#[derive(Clone, Debug)]
pub struct HuggingFace {
    pub repo: &'static str,
    pub model: &'static str,
}

/// Downloadable models, keyed by the `--model` name.
pub static MODELS: LazyLock<BTreeMap<&'static str, HuggingFace>> = LazyLock::new(|| {
    let hf = |repo, model| HuggingFace { repo, model };
    BTreeMap::from([
        // TranslateGemma (google/translategemma-*-it), native translation prompt.
        // Q5_K_M is the best quality that still fits fully on a 4 GB GPU.
        (
            "translategemma-4b",
            hf(
                "mradermacher/translategemma-4b-it-GGUF",
                "translategemma-4b-it.Q5_K_M.gguf",
            ),
        ),
        // ~320 MiB less VRAM, for GPUs shared with e.g. video transcoding.
        (
            "translategemma-4b-q4",
            hf(
                "mradermacher/translategemma-4b-it-GGUF",
                "translategemma-4b-it.Q4_K_M.gguf",
            ),
        ),
        (
            "translategemma-12b",
            hf(
                "mradermacher/translategemma-12b-it-GGUF",
                "translategemma-12b-it.Q4_K_M.gguf",
            ),
        ),
        // General-purpose chat models, generic translation prompt.
        (
            "gemma3-1b",
            hf("libretranslate/gemma3", "gemma-3-1b-it-q4_0.gguf"),
        ),
        (
            "gemma3-4b",
            hf("libretranslate/gemma3", "gemma-3-4b-it-q4_0.gguf"),
        ),
        (
            "gemma3-12b",
            hf("libretranslate/gemma3", "gemma-3-12b-it-q4_0.gguf"),
        ),
        (
            "gemma3-27b",
            hf("libretranslate/gemma3", "gemma-3-27b-it-q4_0.gguf"),
        ),
        (
            "gemma4-e4b",
            hf(
                "bartowski/google_gemma-4-E4B-it-GGUF",
                "google_gemma-4-E4B-it-Q4_0.gguf",
            ),
        ),
    ])
});

/// Name of the default `--model`.
pub const DEFAULT_MODEL: &str = "translategemma-4b";

/// Resolve the model file: a local path if `model_file` is set, otherwise the
/// cached download of `model_id` (downloading it once if needed).
pub fn load_model(model_id: &str, model_file: &str) -> Result<PathBuf> {
    if !model_file.is_empty() {
        let path = PathBuf::from(model_file);
        return if path.is_file() && path.extension().and_then(|ext| ext.to_str()) == Some("gguf") {
            Ok(path)
        } else {
            Err(anyhow!(
                "invalid path or not a .gguf file: {}",
                path.display()
            ))
        };
    }

    let hf = MODELS
        .get(model_id)
        .with_context(|| format!("unknown model: {model_id}"))?;
    let cache = Cache::default();
    if let Some(path) = cache.model(hf.repo.to_owned()).get(hf.model) {
        return Ok(path);
    }

    // hf-hub downloads into <cache>/tmp and renames the file into place when
    // complete. Interrupted downloads leave multi-GB temp files behind.
    remove_stale_downloads(&cache.path().join("tmp"));
    info!(repo = hf.repo, file = hf.model, cache = %cache.path().display(), "downloading model (one-time)");
    ApiBuilder::from_cache(cache)
        .with_progress(std::io::IsTerminal::is_terminal(&std::io::stderr()))
        .build()
        .context("unable to create Hugging Face API client")?
        .model(hf.repo.to_owned())
        .get(hf.model)
        .with_context(|| format!("unable to download {}/{}", hf.repo, hf.model))
}

fn remove_stale_downloads(tmp: &Path) {
    let Ok(entries) = std::fs::read_dir(tmp) else {
        return;
    };
    let cutoff = SystemTime::now() - Duration::from_secs(600);
    for entry in entries.flatten() {
        let stale = entry
            .metadata()
            .and_then(|m| m.modified())
            .is_ok_and(|t| t < cutoff);
        if stale {
            match std::fs::remove_file(entry.path()) {
                Ok(()) => info!(path = %entry.path().display(), "removed interrupted download"),
                Err(err) => {
                    warn!(path = %entry.path().display(), "cannot remove interrupted download: {err}")
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_remote_model_is_an_error_instead_of_a_panic() {
        assert!(load_model("missing", "").is_err());
    }

    #[test]
    fn local_model_must_be_a_gguf_file() {
        assert!(load_model(DEFAULT_MODEL, "/nonexistent/model.gguf").is_err());
        assert!(load_model(DEFAULT_MODEL, "/").is_err());
    }

    #[test]
    fn default_model_exists() {
        assert!(MODELS.contains_key(DEFAULT_MODEL));
    }
}
