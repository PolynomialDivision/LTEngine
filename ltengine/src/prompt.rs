//! Model-specific prompt construction.
//!
//! LTEngine talks to two families of models:
//!
//! * **TranslateGemma**: a translation fine-tune of Gemma 3 whose chat template
//!   takes a structured user message (`source_lang_code`, `target_lang_code`,
//!   `text`) and renders Google's own translation instruction. llama.cpp's
//!   built-in template engine cannot express that structure. It only sees
//!   `<start_of_turn>` and falls back to a plain Gemma wrapper. So we render
//!   the Jinja template embedded in the GGUF ourselves with minijinja.
//! * **Generic chat models** (Gemma 3/4, …): a system + user prompt rendered
//!   through llama.cpp's chat template support.

use std::collections::HashMap;

use anyhow::{Context, Result, anyhow};
use minijinja::{Environment, ErrorKind, context};

const TEMPLATE_NAME: &str = "chat";

/// How the loaded model expects translation requests to be phrased.
#[derive(Debug)]
pub enum PromptFormat {
    /// Native TranslateGemma prompt with explicit language codes.
    TranslateGemma(Box<TranslateGemma>),
    /// Instruction prompt rendered through the model's chat template.
    Chat {
        /// Use a hardcoded Gemma turn format if the GGUF template cannot be
        /// applied. Only safe for models whose architecture is Gemma.
        gemma_fallback: bool,
    },
}

impl PromptFormat {
    /// Pick the prompt format from GGUF metadata.
    pub fn detect(architecture: Option<&str>, chat_template: Option<&str>) -> Self {
        if let Some(template) = chat_template.filter(|t| is_translategemma_template(t)) {
            match TranslateGemma::from_template(template) {
                Ok(tg) => return PromptFormat::TranslateGemma(Box::new(tg)),
                Err(err) => tracing::warn!(
                    "GGUF looks like TranslateGemma but its chat template could not be compiled ({err:#}); \
                     using the built-in TranslateGemma prompt"
                ),
            }
            return PromptFormat::TranslateGemma(Box::new(TranslateGemma::builtin()));
        }

        PromptFormat::Chat {
            gemma_fallback: architecture.is_some_and(|a| a.starts_with("gemma")),
        }
    }

    /// Short human-readable name for logs.
    pub fn name(&self) -> &'static str {
        match self {
            PromptFormat::TranslateGemma(tg) if tg.env.is_some() => {
                "translategemma (GGUF template)"
            }
            PromptFormat::TranslateGemma(_) => "translategemma (built-in template)",
            PromptFormat::Chat { .. } => "generic chat",
        }
    }

    /// Whether HTML must be translated as Markdown. TranslateGemma has no
    /// HTML instruction and turns tags into Markdown on its own.
    pub fn translates_html_as_markdown(&self) -> bool {
        matches!(self, PromptFormat::TranslateGemma(_))
    }

    /// Whether the model accepts `code` as a source/target language.
    /// Generic chat models are prompted with language names and accept every
    /// language LTEngine knows.
    pub fn supports_language(&self, code: &str) -> bool {
        match self {
            PromptFormat::TranslateGemma(tg) => tg.language_name(code).is_some(),
            PromptFormat::Chat { .. } => true,
        }
    }
}

/// The TranslateGemma chat template keys its user message on these fields.
fn is_translategemma_template(template: &str) -> bool {
    template.contains("source_lang_code") && template.contains("target_lang_code")
}

/// TranslateGemma prompt renderer.
#[derive(Debug)]
pub struct TranslateGemma {
    /// Compiled GGUF chat template, if the model carries one.
    env: Option<Environment<'static>>,
    /// Language code → English name, as used inside the prompt.
    languages: HashMap<String, String>,
}

impl TranslateGemma {
    /// Compile the chat template shipped in the GGUF.
    pub fn from_template(template: &str) -> Result<Self> {
        let languages = parse_language_map(template);
        if languages.is_empty() {
            return Err(anyhow!("no language map found in chat template"));
        }

        let mut env = Environment::new();
        env.add_function(
            "raise_exception",
            |msg: String| -> Result<String, minijinja::Error> {
                Err(minijinja::Error::new(ErrorKind::InvalidOperation, msg))
            },
        );
        env.add_template_owned(TEMPLATE_NAME, template.to_owned())
            .context("invalid chat template")?;

        let tg = TranslateGemma {
            env: Some(env),
            languages,
        };
        // Fail early (and fall back to the built-in renderer) if the template
        // does not render for a trivial request.
        tg.render("en", "de", "test")
            .context("chat template does not render")?;
        Ok(tg)
    }

    /// Renderer used when the GGUF has no usable template. Produces exactly
    /// what the official template renders.
    pub fn builtin() -> Self {
        TranslateGemma {
            env: None,
            languages: BUILTIN_LANGUAGES
                .iter()
                .map(|&(code, name)| (code.to_owned(), name.to_owned()))
                .collect(),
        }
    }

    /// English language name for a code (`de`, `pt-BR`, `pt_BR`, …).
    pub fn language_name(&self, code: &str) -> Option<&str> {
        self.languages
            .get(&code.replace('_', "-"))
            .map(String::as_str)
    }

    /// Render the complete prompt (without BOS, which the tokenizer adds).
    pub fn render(&self, source: &str, target: &str, text: &str) -> Result<String> {
        let source_name = self
            .language_name(source)
            .ok_or_else(|| anyhow!("language '{source}' is not supported by this model"))?;
        let target_name = self
            .language_name(target)
            .ok_or_else(|| anyhow!("language '{target}' is not supported by this model"))?;

        match &self.env {
            Some(env) => {
                let content = context! {
                    type => "text",
                    source_lang_code => source,
                    target_lang_code => target,
                    text => text,
                };
                let messages = vec![context! { role => "user", content => vec![content] }];
                Ok(env.get_template(TEMPLATE_NAME)?.render(context! {
                    messages => messages,
                    bos_token => "",
                    add_generation_prompt => true,
                })?)
            }
            None => Ok(render_builtin(
                source,
                source_name,
                target,
                target_name,
                text,
            )),
        }
    }
}

fn render_builtin(
    source: &str,
    source_name: &str,
    target: &str,
    target_name: &str,
    text: &str,
) -> String {
    let source = source.replace('_', "-");
    let target = target.replace('_', "-");
    format!(
        "<start_of_turn>user\n\
         You are a professional {source_name} ({source}) to {target_name} ({target}) translator. \
         Your goal is to accurately convey the meaning and nuances of the original {source_name} text \
         while adhering to {target_name} grammar, vocabulary, and cultural sensitivities.\n\
         Produce only the {target_name} translation, without any additional explanations or commentary. \
         Please translate the following {source_name} text into {target_name}:\n\n\n{text}<end_of_turn>\n\
         <start_of_turn>model\n",
        text = text.trim(),
    )
}

/// Extract `"code": "Name"` pairs from the `{% set languages = {...} %}` block.
fn parse_language_map(template: &str) -> HashMap<String, String> {
    let Some(start) = template.find("set languages") else {
        return HashMap::new();
    };
    let block = &template[start..];
    let block = &block[..block.find('}').unwrap_or(block.len())];

    block
        .lines()
        .filter_map(|line| {
            let (key, value) = line.trim().trim_end_matches(',').split_once(':')?;
            let key = key.trim().strip_prefix('"')?.strip_suffix('"')?;
            let value = value.trim().strip_prefix('"')?.strip_suffix('"')?;
            Some((key.to_owned(), value.to_owned()))
        })
        .collect()
}

/// Language names used by the official template for LTEngine's languages.
const BUILTIN_LANGUAGES: &[(&str, &str)] = &[
    ("ar", "Arabic"),
    ("az", "Azerbaijani"),
    ("bg", "Bulgarian"),
    ("bn", "Bengali"),
    ("ca", "Catalan"),
    ("cs", "Czech"),
    ("da", "Danish"),
    ("de", "German"),
    ("el", "Greek"),
    ("en", "English"),
    ("eo", "Esperanto"),
    ("es", "Spanish"),
    ("et", "Estonian"),
    ("eu", "Basque"),
    ("fa", "Persian"),
    ("fi", "Finnish"),
    ("fr", "French"),
    ("ga", "Irish"),
    ("gl", "Galician"),
    ("he", "Hebrew"),
    ("hi", "Hindi"),
    ("hu", "Hungarian"),
    ("id", "Indonesian"),
    ("it", "Italian"),
    ("ja", "Japanese"),
    ("ko", "Korean"),
    ("lt", "Lithuanian"),
    ("lv", "Latvian"),
    ("ms", "Malay"),
    ("nb", "Norwegian Bokmål"),
    ("nl", "Dutch"),
    ("pl", "Polish"),
    ("pt", "Portuguese"),
    ("pt-BR", "Portuguese"),
    ("ro", "Romanian"),
    ("ru", "Russian"),
    ("sk", "Slovak"),
    ("sl", "Slovenian"),
    ("sq", "Albanian"),
    ("sr", "Serbian"),
    ("sv", "Swedish"),
    ("th", "Thai"),
    ("tl", "Tagalog"),
    ("tr", "Turkish"),
    ("uk", "Ukrainian"),
    ("ur", "Urdu"),
    ("vi", "Vietnamese"),
    ("zh", "Chinese"),
    ("zh-Hans", "Chinese"),
    ("zh-Hant", "Chinese"),
];

/// Instruction prompt for generic chat models.
pub struct ChatPrompt {
    /// System instructions.
    pub system: &'static str,
    /// User turn containing the text.
    pub user: String,
}

/// Build the instruction prompt for generic chat models. `source_name` is
/// `None` when the source language is unknown (auto-detect failed).
pub fn chat_prompt(
    format: &str,
    source_name: Option<&str>,
    target_name: &str,
    text: &str,
) -> ChatPrompt {
    let system = if format == "html" {
        "You are an expert linguist, specializing in translation. You are able to capture the nuances of the languages you translate. You pay attention to masculine/feminine/plural and proper use of articles and grammar. You always provide natural sounding translations that fully preserve the meaning of the original text. You never provide explanations for your work. You must preserve all HTML tags and elements in the translation. You always answer with the translated text and nothing else."
    } else {
        "You are an expert linguist, specializing in translation. You are able to capture the nuances of the languages you translate. You pay attention to masculine/feminine/plural and proper use of articles and grammar. You always provide natural sounding translations that fully preserve the meaning of the original text. You never provide explanations for your work. You always answer with the translated text and nothing else."
    };

    let user = match source_name {
        Some(source) => format!(
            "Translate the text below from {source} to {target_name}.\n\n{source}: {text}\n\n{target_name}:\n"
        ),
        None => format!(
            "Translate the text below to {target_name}.\n\nText: {text}\n\n{target_name}:\n"
        ),
    };

    ChatPrompt { system, user }
}

/// Hardcoded Gemma turn format for Gemma models whose GGUF template llama.cpp
/// cannot apply.
pub fn gemma_fallback(prompt: &ChatPrompt) -> String {
    format!(
        "<start_of_turn>user\n{}\n\n{}<end_of_turn>\n<start_of_turn>model\n",
        prompt.system, prompt.user
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEMPLATE: &str = include_str!("../tests/fixtures/translategemma_chat_template.jinja");

    #[test]
    fn detects_translategemma_from_gguf_template() {
        let format = PromptFormat::detect(Some("gemma3"), Some(TEMPLATE));
        assert!(matches!(format, PromptFormat::TranslateGemma(ref tg) if tg.env.is_some()));
        for code in ["de", "en", "uk"] {
            assert!(format.supports_language(code), "{code}");
        }
        assert!(!format.supports_language("xx"));
    }

    #[test]
    fn plain_gemma_uses_chat_prompt() {
        let format =
            PromptFormat::detect(Some("gemma3"), Some("{{ bos_token }}<start_of_turn>user"));
        assert!(matches!(
            format,
            PromptFormat::Chat {
                gemma_fallback: true
            }
        ));
        let format = PromptFormat::detect(Some("llama"), None);
        assert!(matches!(
            format,
            PromptFormat::Chat {
                gemma_fallback: false
            }
        ));
    }

    #[test]
    fn gguf_template_renders_native_prompt() {
        let tg = TranslateGemma::from_template(TEMPLATE).unwrap();
        let prompt = tg.render("de", "uk", "  Hallo Welt  ").unwrap();
        assert_eq!(
            prompt,
            "<start_of_turn>user\nYou are a professional German (de) to Ukrainian (uk) translator. \
             Your goal is to accurately convey the meaning and nuances of the original German text \
             while adhering to Ukrainian grammar, vocabulary, and cultural sensitivities.\n\
             Produce only the Ukrainian translation, without any additional explanations or commentary. \
             Please translate the following German text into Ukrainian:\n\n\nHallo Welt<end_of_turn>\n\
             <start_of_turn>model\n"
        );
    }

    #[test]
    fn builtin_renderer_matches_gguf_template() {
        let gguf = TranslateGemma::from_template(TEMPLATE).unwrap();
        let builtin = TranslateGemma::builtin();
        let text = "Line one.\n\nLine <b>two</b> & \"three\" {{ not jinja }}\n";
        for (source, target) in [
            ("de", "en"),
            ("en", "de"),
            ("de", "uk"),
            ("uk", "de"),
            ("en", "uk"),
            ("uk", "en"),
            ("pt-BR", "zh-Hant"),
        ] {
            assert_eq!(
                gguf.render(source, target, text).unwrap(),
                builtin.render(source, target, text).unwrap(),
                "{source} -> {target}"
            );
        }
    }

    #[test]
    fn builtin_languages_match_gguf_names() {
        let gguf = TranslateGemma::from_template(TEMPLATE).unwrap();
        for (code, name) in BUILTIN_LANGUAGES {
            assert_eq!(gguf.language_name(code), Some(*name), "{code}");
        }
    }

    #[test]
    fn unsupported_language_is_an_error() {
        let tg = TranslateGemma::builtin();
        assert!(tg.render("de", "tlh", "Hallo").is_err());
    }
}
