//! LibreTranslate-compatible HTTP API plus health and metrics endpoints.

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Instant;

use actix_multipart::form::{MultipartForm, text::Text as MPText};
use actix_web::{FromRequest, HttpRequest, HttpResponse, Responder, get, http::header, post, web};
use serde::Deserialize;
use tracing::info;

use crate::Args;
use crate::cache::{CacheKey, TranslationCache};
use crate::engine::{Engine, EngineError, Request};
use crate::error_response::ErrorResponse;
use crate::formatting::match_source_style;
use crate::languages::{LANGUAGES, Language, detect_lang, get_language_from_code};
use crate::markup;
use crate::metrics::Metrics;

/// Shared state for all handlers.
#[derive(Debug)]
pub struct AppState {
    pub args: Arc<Args>,
    pub engine: Arc<Engine>,
    pub metrics: Arc<Metrics>,
    pub cache: TranslationCache,
}

#[derive(Debug, Deserialize)]
pub struct TranslateRequest {
    q: Option<String>,
    source: Option<String>,
    target: Option<String>,
    format: Option<String>,
    api_key: Option<String>,
    alternatives: Option<u32>,
}

#[derive(MultipartForm)]
struct MPTranslateRequest {
    q: Option<MPText<String>>,
    source: Option<MPText<String>>,
    target: Option<MPText<String>>,
    format: Option<MPText<String>>,
    api_key: Option<MPText<String>>,
    alternatives: Option<MPText<u32>>,
}

impl MPTranslateRequest {
    fn into_translate_request(self) -> TranslateRequest {
        TranslateRequest {
            q: self.q.map(MPText::into_inner),
            source: self.source.map(MPText::into_inner),
            target: self.target.map(MPText::into_inner),
            format: self.format.map(MPText::into_inner),
            api_key: self.api_key.map(MPText::into_inner),
            alternatives: self.alternatives.map(MPText::into_inner),
        }
    }
}

fn bad_request(error: impl Into<String>) -> ErrorResponse {
    ErrorResponse {
        error: error.into(),
        status: 400,
        retry_after: None,
    }
}

async fn parse_payload(
    req: HttpRequest,
    payload: web::Payload,
) -> Result<TranslateRequest, ErrorResponse> {
    let content_type = req
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|h| h.to_str().ok())
        .unwrap_or("");
    let mut payload = payload.into_inner();

    if content_type.starts_with("application/json") {
        Ok(
            web::Json::<TranslateRequest>::from_request(&req, &mut payload)
                .await?
                .into_inner(),
        )
    } else if content_type.starts_with("application/x-www-form-urlencoded") {
        Ok(
            web::Form::<TranslateRequest>::from_request(&req, &mut payload)
                .await?
                .into_inner(),
        )
    } else if content_type.starts_with("multipart/form-data") {
        Ok(
            MultipartForm::<MPTranslateRequest>::from_request(&req, &mut payload)
                .await?
                .into_inner()
                .into_translate_request(),
        )
    } else {
        Err(bad_request("Unsupported content-type"))
    }
}

/// Validate the API key and required parameters; returns `q`.
fn check_params<'a>(
    body: &'a TranslateRequest,
    args: &Args,
    required: &[(&str, &Option<String>)],
) -> Result<&'a str, ErrorResponse> {
    for (key, value) in required {
        if value.as_ref().is_none_or(|v| v.trim().is_empty()) {
            return Err(bad_request(format!(
                "Invalid request: missing {key} parameter"
            )));
        }
    }

    if !args.api_key.is_empty() && body.api_key.as_deref() != Some(args.api_key.as_str()) {
        return Err(ErrorResponse {
            error: "Invalid API key".to_string(),
            status: 403,
            retry_after: None,
        });
    }

    let q = body
        .q
        .as_deref()
        .ok_or_else(|| bad_request("Invalid request: missing q parameter"))?;
    let char_count = q.chars().count();
    if char_count > args.char_limit {
        return Err(bad_request(format!(
            "Invalid request: request ({char_count}) exceeds text limit ({})",
            args.char_limit
        )));
    }
    Ok(q)
}

fn engine_error(err: EngineError) -> ErrorResponse {
    ErrorResponse {
        error: err.to_string(),
        status: err.status(),
        retry_after: err.retry_after(),
    }
}

#[post("/detect")]
async fn detect(
    req: HttpRequest,
    payload: web::Payload,
    state: web::Data<AppState>,
) -> Result<HttpResponse, ErrorResponse> {
    let body = parse_payload(req, payload).await?;
    let q = check_params(&body, &state.args, &[("q", &body.q)])?;
    let (language, confidence) =
        detect_lang(q).map_or((LANGUAGES[0].code, 0), |d| (d.language.code, d.confidence));
    Ok(HttpResponse::Ok()
        .json(serde_json::json!([{ "language": language, "confidence": confidence }])))
}

#[post("/translate")]
async fn translate(
    req: HttpRequest,
    payload: web::Payload,
    state: web::Data<AppState>,
) -> Result<HttpResponse, ErrorResponse> {
    let started = Instant::now();
    let peer = req.conn_data::<PeerSocket>().copied();
    let body = parse_payload(req, payload).await?;
    let q = check_params(
        &body,
        &state.args,
        &[
            ("q", &body.q),
            ("source", &body.source),
            ("target", &body.target),
        ],
    )?;
    let source = body.source.as_deref().unwrap_or_default();
    let target = body.target.as_deref().unwrap_or_default();
    let format = body.format.as_deref().unwrap_or("text");
    if !matches!(format, "text" | "html") {
        return Err(bad_request("Invalid format. Supported formats: text, html"));
    }

    state.metrics.requests.fetch_add(1, Ordering::Relaxed);
    // Dropping the translation future cancels the job in the engine.
    let result = tokio::select! {
        result = run_translation(&state, q, source, target, format, started) => result,
        () = client_disconnected(peer) => Err(engine_error(EngineError::Cancelled)),
    };
    if let Err(err) = &result {
        match err.status {
            499 => 0,
            503 => state.metrics.rejected_busy.fetch_add(1, Ordering::Relaxed),
            _ => state.metrics.errors.fetch_add(1, Ordering::Relaxed),
        };
        info!(
            source,
            target,
            status = err.status,
            "translation failed: {}",
            err.error
        );
    }
    state.metrics.request_duration.observe(started.elapsed());
    let (translated, detected) = result?;

    let mut response = serde_json::json!({ "translatedText": translated });
    // Compatibility only: alternatives are not generated.
    if body.alternatives.is_some_and(|v| v > 0) {
        response["alternatives"] = serde_json::json!([]);
    }
    if source == "auto" {
        let (language, confidence) = detected.map_or((LANGUAGES[0].code, 0), |(l, c)| (l.code, c));
        response["detectedLanguage"] =
            serde_json::json!({ "language": language, "confidence": confidence });
    }
    Ok(HttpResponse::Ok().json(response))
}

/// Socket of the HTTP connection, captured in `on_connect`.
#[derive(Debug, Clone, Copy)]
pub struct PeerSocket(#[cfg_attr(not(unix), allow(dead_code))] i32);

/// `HttpServer::on_connect` hook: remember the connection's socket so a
/// handler can notice when the client goes away. actix-web keeps running an
/// HTTP/1 handler after the client disconnects, which would leave the GPU
/// translating for nobody.
pub fn capture_peer_socket(connection: &dyn std::any::Any, data: &mut actix_web::dev::Extensions) {
    #[cfg(unix)]
    if let Some(stream) = connection.downcast_ref::<actix_web::rt::net::TcpStream>() {
        use std::os::fd::AsRawFd;
        data.insert(PeerSocket(stream.as_raw_fd()));
    }
    #[cfg(not(unix))]
    let _ = (connection, data);
}

/// Resolves once the client has closed its connection; never resolves if the
/// socket is unknown. The request body has been read by then, so the socket
/// only becomes readable on EOF (or a pipelined request, treated as alive).
async fn client_disconnected(peer: Option<PeerSocket>) {
    let Some(PeerSocket(fd)) = peer else {
        return std::future::pending().await;
    };
    loop {
        actix_web::rt::time::sleep(std::time::Duration::from_millis(250)).await;
        #[cfg(unix)]
        {
            let mut byte = 0u8;
            // SAFETY: non-blocking peek of one byte into a valid buffer. The
            // fd belongs to this request's connection, which the dispatcher
            // keeps open while the handler runs.
            let n = unsafe {
                libc::recv(
                    fd,
                    (&raw mut byte).cast(),
                    1,
                    libc::MSG_PEEK | libc::MSG_DONTWAIT,
                )
            };
            let closed = n == 0
                || (n < 0
                    && !matches!(
                        std::io::Error::last_os_error().kind(),
                        std::io::ErrorKind::WouldBlock | std::io::ErrorKind::Interrupted
                    ));
            if closed {
                return;
            }
        }
        #[cfg(not(unix))]
        let _ = fd;
    }
}

type Detected = Option<(&'static Language, i32)>;

async fn run_translation(
    state: &AppState,
    q: &str,
    source: &str,
    target: &str,
    format: &str,
    started: Instant,
) -> Result<(String, Detected), ErrorResponse> {
    let model = state.engine.model().map_err(engine_error)?;
    let supported = |language: &'static Language| {
        if model.prompt.supports_language(language.code) {
            Ok(language)
        } else {
            Err(bad_request(format!(
                "{} is not supported by the loaded model",
                language.code
            )))
        }
    };

    let target_lang = supported(
        get_language_from_code(target)
            .ok_or_else(|| bad_request(format!("{target} is not supported")))?,
    )?;
    let detected = (source == "auto")
        .then(|| detect_lang(q))
        .flatten()
        .map(|d| (d.language, d.confidence));
    let source_lang = if source == "auto" {
        detected.map(|(language, _)| language)
    } else {
        Some(
            get_language_from_code(source)
                .ok_or_else(|| bad_request(format!("{source} is not supported")))?,
        )
    };
    let source_lang = source_lang.map(supported).transpose()?;

    if source_lang.is_some_and(|s| s.code == target_lang.code) {
        return Ok((q.to_owned(), detected));
    }

    let cache_key = CacheKey {
        source: source_lang.map_or("auto", |l| l.code).to_owned(),
        target: target_lang.code.to_owned(),
        format: format.to_owned(),
        text: q.to_owned(),
    };
    if let Some(hit) = state.cache.get(&cache_key) {
        state.metrics.cache_hits.fetch_add(1, Ordering::Relaxed);
        return Ok((hit, detected));
    }

    let as_markdown = format == "html" && model.prompt.translates_html_as_markdown();
    let text = if as_markdown {
        markup::html_to_markdown(q)
    } else {
        q.to_owned()
    };
    let output = state
        .engine
        .translate(Request {
            source: source_lang,
            target: target_lang,
            format: format.to_owned(),
            text,
        })
        .await
        .map_err(engine_error)?;

    info!(
        source = source_lang.map_or("?", |l| l.code),
        target = target_lang.code,
        format,
        input_tokens = output.input_tokens,
        output_tokens = output.output_tokens,
        queue_ms = output.queue_time.as_millis() as u64,
        inference_ms = output.inference_time.as_millis() as u64,
        total_ms = started.elapsed().as_millis() as u64,
        "translated"
    );

    let text = if format == "text" {
        match_source_style(q, &output.text)
    } else if as_markdown {
        markup::markdown_to_html(&output.text, q)
    } else {
        output.text
    };
    state.cache.insert(cache_key, text.clone());
    Ok((text, detected))
}

#[post("/translate_file")]
async fn translate_file() -> Result<HttpResponse, ErrorResponse> {
    Err(ErrorResponse {
        error: "Not implemented".to_string(),
        status: 501,
        retry_after: None,
    })
}

#[post("/suggest")]
async fn suggest() -> Result<HttpResponse, ErrorResponse> {
    Err(ErrorResponse {
        error: "Not implemented".to_string(),
        status: 501,
        retry_after: None,
    })
}

#[get("/languages")]
async fn get_languages(state: web::Data<AppState>) -> impl Responder {
    let Ok(model) = state.engine.model() else {
        return HttpResponse::Ok().json(&*LANGUAGES);
    };
    let supported: Vec<&Language> = LANGUAGES
        .iter()
        .filter(|l| model.prompt.supports_language(l.code))
        .collect();
    let targets: Vec<&str> = supported.iter().map(|l| l.code).collect();
    let body: Vec<_> = supported
        .iter()
        .map(|l| serde_json::json!({ "code": l.code, "name": l.name, "targets": targets }))
        .collect();
    HttpResponse::Ok().json(body)
}

#[get("/frontend/settings")]
async fn get_frontend_settings(state: web::Data<AppState>) -> impl Responder {
    HttpResponse::Ok().json(serde_json::json!({
        "apiKeys": false,
        "charLimit": state.args.char_limit,
        "filesTranslation": false,
        "frontendTimeout": 1000,
        "keyRequired": !state.args.api_key.is_empty(),
        "language": {
            "source": { "code": "auto", "name": "Auto Detect" },
            "target": { "code": "en", "name": "English" }
        },
        "suggestions": false,
        "supportedFilesFormat": []
    }))
}

/// Liveness: the process is up and serving HTTP.
#[get("/health")]
async fn health(state: web::Data<AppState>) -> impl Responder {
    HttpResponse::Ok().json(serde_json::json!({ "status": "ok", "engine": state.engine.state() }))
}

/// Readiness: the model is loaded and translations can be served.
#[get("/health/ready")]
async fn ready(state: web::Data<AppState>) -> impl Responder {
    let engine = state.engine.state();
    let body = serde_json::json!({ "status": engine });
    if engine == "ready" {
        HttpResponse::Ok().json(body)
    } else {
        HttpResponse::ServiceUnavailable().json(body)
    }
}

#[get("/metrics")]
async fn metrics(state: web::Data<AppState>) -> impl Responder {
    HttpResponse::Ok()
        .content_type("text/plain; version=0.0.4")
        .body(state.metrics.render())
}

/// Register all API routes.
pub fn configure(cfg: &mut web::ServiceConfig) {
    cfg.service(get_languages)
        .service(get_frontend_settings)
        .service(translate)
        .service(translate_file)
        .service(detect)
        .service(suggest)
        .service(health)
        .service(ready)
        .service(metrics);
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    fn args(char_limit: usize) -> Args {
        let mut args = Args::parse_from(["ltengine"]);
        args.char_limit = char_limit;
        args
    }

    fn request(q: Option<&str>) -> TranslateRequest {
        TranslateRequest {
            q: q.map(str::to_owned),
            source: None,
            target: None,
            format: None,
            api_key: None,
            alternatives: None,
        }
    }

    #[test]
    fn missing_q_is_a_400_not_a_panic() {
        let body = request(None);
        assert_eq!(check_params(&body, &args(2), &[]).unwrap_err().status, 400);
    }

    #[test]
    fn request_limit_counts_characters_not_bytes() {
        let body = request(Some("你好"));
        assert!(check_params(&body, &args(2), &[("q", &body.q)]).is_ok());
        let body = request(Some("你好!"));
        assert!(check_params(&body, &args(2), &[("q", &body.q)]).is_err());
    }

    #[test]
    fn api_key_is_enforced() {
        let mut args = args(100);
        args.api_key = "secret".into();
        let mut body = request(Some("hi"));
        assert_eq!(check_params(&body, &args, &[]).unwrap_err().status, 403);
        body.api_key = Some("secret".into());
        assert!(check_params(&body, &args, &[]).is_ok());
    }
}
