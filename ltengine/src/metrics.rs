//! Minimal Prometheus metrics, rendered in the text exposition format.

use std::fmt::Write;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::time::Duration;

const DURATION_BUCKETS: [f64; 12] = [
    0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0,
];

/// Fixed-bucket histogram of durations in seconds.
#[derive(Debug, Default)]
pub struct Histogram {
    buckets: [AtomicU64; DURATION_BUCKETS.len()],
    count: AtomicU64,
    sum_micros: AtomicU64,
}

impl Histogram {
    /// Record one observation.
    pub fn observe(&self, value: Duration) {
        let secs = value.as_secs_f64();
        for (bucket, bound) in self.buckets.iter().zip(DURATION_BUCKETS) {
            if secs <= bound {
                bucket.fetch_add(1, Ordering::Relaxed);
            }
        }
        self.count.fetch_add(1, Ordering::Relaxed);
        self.sum_micros.fetch_add(
            u64::try_from(value.as_micros()).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
    }

    fn render(&self, out: &mut String, name: &str, help: &str) {
        let _ = writeln!(out, "# HELP {name} {help}\n# TYPE {name} histogram");
        for (bucket, bound) in self.buckets.iter().zip(DURATION_BUCKETS) {
            let _ = writeln!(
                out,
                "{name}_bucket{{le=\"{bound}\"}} {}",
                bucket.load(Ordering::Relaxed)
            );
        }
        let count = self.count.load(Ordering::Relaxed);
        let sum = self.sum_micros.load(Ordering::Relaxed) as f64 / 1e6;
        let _ = writeln!(
            out,
            "{name}_bucket{{le=\"+Inf\"}} {count}\n{name}_sum {sum}\n{name}_count {count}"
        );
    }
}

/// Process-wide counters.
#[derive(Debug, Default)]
pub struct Metrics {
    pub requests: AtomicU64,
    pub errors: AtomicU64,
    pub rejected_busy: AtomicU64,
    pub cancelled: AtomicU64,
    pub cache_hits: AtomicU64,
    pub input_tokens: AtomicU64,
    pub output_tokens: AtomicU64,
    pub active: AtomicI64,
    pub queued: AtomicI64,
    pub model_loaded: AtomicBool,
    pub request_duration: Histogram,
    pub queue_duration: Histogram,
    pub inference_duration: Histogram,
}

impl Metrics {
    /// Render all metrics.
    pub fn render(&self) -> String {
        let mut out = String::with_capacity(4096);
        let counter = |out: &mut String, name: &str, help: &str, value: u64| {
            let _ = writeln!(
                out,
                "# HELP {name} {help}\n# TYPE {name} counter\n{name} {value}"
            );
        };
        let gauge = |out: &mut String, name: &str, help: &str, value: i64| {
            let _ = writeln!(
                out,
                "# HELP {name} {help}\n# TYPE {name} gauge\n{name} {value}"
            );
        };
        let load = |a: &AtomicU64| a.load(Ordering::Relaxed);

        counter(
            &mut out,
            "ltengine_translate_requests_total",
            "Translation requests received.",
            load(&self.requests),
        );
        counter(
            &mut out,
            "ltengine_translate_errors_total",
            "Translation requests that failed.",
            load(&self.errors),
        );
        counter(
            &mut out,
            "ltengine_translate_rejected_busy_total",
            "Requests rejected because the queue was full or timed out.",
            load(&self.rejected_busy),
        );
        counter(
            &mut out,
            "ltengine_translate_cancelled_total",
            "Requests abandoned by the client before completion.",
            load(&self.cancelled),
        );
        counter(
            &mut out,
            "ltengine_translate_cache_hits_total",
            "Translations served from the in-memory cache.",
            load(&self.cache_hits),
        );
        counter(
            &mut out,
            "ltengine_input_tokens_total",
            "Prompt tokens processed.",
            load(&self.input_tokens),
        );
        counter(
            &mut out,
            "ltengine_output_tokens_total",
            "Tokens generated.",
            load(&self.output_tokens),
        );
        gauge(
            &mut out,
            "ltengine_active_inferences",
            "Inferences currently running.",
            self.active.load(Ordering::Relaxed),
        );
        gauge(
            &mut out,
            "ltengine_queued_requests",
            "Requests waiting for the inference worker.",
            self.queued.load(Ordering::Relaxed),
        );
        gauge(
            &mut out,
            "ltengine_model_loaded",
            "1 when the model is loaded and ready.",
            i64::from(self.model_loaded.load(Ordering::Relaxed)),
        );
        self.request_duration.render(
            &mut out,
            "ltengine_request_duration_seconds",
            "End-to-end /translate latency.",
        );
        self.queue_duration.render(
            &mut out,
            "ltengine_queue_duration_seconds",
            "Time spent waiting for the inference worker.",
        );
        self.inference_duration.render(
            &mut out,
            "ltengine_inference_duration_seconds",
            "Model inference time.",
        );
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn histogram_buckets_are_cumulative() {
        let h = Histogram::default();
        h.observe(Duration::from_millis(300));
        h.observe(Duration::from_secs(4));
        let mut out = String::new();
        h.render(&mut out, "x", "help");
        assert!(out.contains("x_bucket{le=\"0.25\"} 0"));
        assert!(out.contains("x_bucket{le=\"0.5\"} 1"));
        assert!(out.contains("x_bucket{le=\"5\"} 2"));
        assert!(out.contains("x_count 2"));
    }
}
