use serde::Serialize;
use std::collections::HashMap;
use std::sync::LazyLock;
use whatlang::{Detector, Lang};

const LANGS: &[(&str, &str, &str)] = &[
    ("en", "", "English"),
    ("sq", "", "Albanian"),
    ("ar", "", "Arabic"),
    ("az", "", "Azerbaijani"),
    ("eu", "", "Basque"),
    ("bn", "", "Bengali"),
    ("bg", "", "Bulgarian"),
    ("ca", "", "Catalan"),
    ("zh", "zh-Hans", "Chinese"),
    ("zt", "zh-Hant", "Chinese (traditional)"),
    ("cs", "", "Czech"),
    ("da", "", "Danish"),
    ("nl", "", "Dutch"),
    ("eo", "", "Esperanto"),
    ("et", "", "Estonian"),
    ("fi", "", "Finnish"),
    ("fr", "", "French"),
    ("gl", "", "Galician"),
    ("de", "", "German"),
    ("el", "", "Greek"),
    ("he", "", "Hebrew"),
    ("hi", "", "Hindi"),
    ("hu", "", "Hungarian"),
    ("id", "", "Indonesian"),
    ("ga", "", "Irish"),
    ("it", "", "Italian"),
    ("ja", "", "Japanese"),
    ("ko", "", "Korean"),
    ("lv", "", "Latvian"),
    ("lt", "", "Lithuanian"),
    ("ms", "", "Malay"),
    ("nb", "", "Norwegian"),
    ("fa", "", "Persian"),
    ("pl", "", "Polish"),
    ("pt", "", "Portuguese"),
    ("pb", "pt-BR", "Portuguese (Brazil)"),
    ("ro", "", "Romanian"),
    ("ru", "", "Russian"),
    ("sr", "", "Serbian"),
    ("sk", "", "Slovak"),
    ("sl", "", "Slovenian"),
    ("es", "", "Spanish"),
    ("sv", "", "Swedish"),
    ("tl", "", "Tagalog"),
    ("th", "", "Thai"),
    ("tr", "", "Turkish"),
    ("uk", "", "Ukrainian"),
    ("ur", "", "Urdu"),
    ("vi", "", "Vietnamese"),
];

#[derive(Debug, Serialize)]
pub struct Language {
    pub code: &'static str,
    pub name: &'static str,
    pub targets: &'static [&'static str],

    #[serde(skip)]
    pub lang_detect: Option<Lang>,

    #[serde(skip)]
    pub internal_code: &'static str,
}

pub static LANGUAGES: LazyLock<Vec<Language>> = LazyLock::new(|| {
    // From whatlang names to our names
    let eng_name_map: HashMap<&'static str, &'static str> =
        HashMap::from([("Mandarin", "Chinese")]);

    let lang_detect_map: HashMap<&'static str, Lang> = Lang::all()
        .iter()
        .map(|lang| {
            let eng_name = lang.eng_name();
            (*eng_name_map.get(eng_name).unwrap_or(&eng_name), *lang)
        })
        .collect();

    let targets: &'static [&'static str] = Box::leak(
        LANGS
            .iter()
            .map(|&(code, alias, _)| if alias.is_empty() { code } else { alias })
            .collect::<Vec<_>>()
            .into_boxed_slice(),
    );

    LANGS
        .iter()
        .map(|&(code, alias, name)| Language {
            code: if alias.is_empty() { code } else { alias },
            name,
            targets,
            lang_detect: lang_detect_map.get(name).copied(),
            internal_code: code,
        })
        .collect()
});

static LANGUAGES_MAP: LazyLock<HashMap<&'static str, &'static Language>> = LazyLock::new(|| {
    LANGUAGES
        .iter()
        .map(|lang| (lang.internal_code, lang))
        .collect()
});

static CODE_TO_INTERNAL_CODE_MAP: LazyLock<HashMap<&'static str, &'static str>> =
    LazyLock::new(|| {
        LANGUAGES
            .iter()
            .map(|lang| (lang.code, lang.internal_code))
            .collect()
    });

/// Look up a language by its public code (`pt-BR`) or internal code (`pb`).
pub fn get_language_from_code(code: &str) -> Option<&'static Language> {
    let internal_code = CODE_TO_INTERNAL_CODE_MAP.get(code).unwrap_or(&code);
    LANGUAGES_MAP.get(internal_code).copied()
}

pub struct LangDetect {
    pub language: &'static Language,
    pub confidence: i32,
}

static LANGUAGE_DETECTOR: LazyLock<Detector> = LazyLock::new(|| {
    Detector::with_allowlist(
        LANGUAGES
            .iter()
            .filter_map(|language| language.lang_detect)
            .collect(),
    )
});

/// Detect the language of `q`. `None` when whatlang cannot tell.
pub fn detect_lang(q: &str) -> Option<LangDetect> {
    let info = LANGUAGE_DETECTOR.detect(q)?;
    let language = LANGUAGES
        .iter()
        .find(|l| l.lang_detect == Some(info.lang()))?;
    Some(LangDetect {
        language,
        confidence: (info.confidence() * 100.0) as i32,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deployment_languages_resolve() {
        for code in ["de", "en", "uk"] {
            assert_eq!(get_language_from_code(code).unwrap().code, code);
        }
        assert_eq!(get_language_from_code("pb").unwrap().code, "pt-BR");
        assert_eq!(
            get_language_from_code("pt-BR").unwrap().name,
            "Portuguese (Brazil)"
        );
        assert!(get_language_from_code("xx").is_none());
    }

    #[test]
    fn detects_deployment_languages() {
        let cases = [
            (
                "Das ist ein ganz normaler deutscher Satz über das Wetter.",
                "de",
            ),
            (
                "This is a perfectly normal English sentence about the weather.",
                "en",
            ),
            ("Це звичайне українське речення про погоду сьогодні.", "uk"),
        ];
        for (text, code) in cases {
            assert_eq!(detect_lang(text).unwrap().language.code, code, "{text}");
        }
    }
}
