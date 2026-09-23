//! Post-processing of generated text.
//!
//! Kept deliberately conservative: a valid translation should come out
//! unchanged apart from surrounding whitespace.

/// Strip model control markers from raw generated text.
///
/// Only removes markers in the positions where models emit them (a Gemma 4
/// thinking prefix, trailing end-of-turn markers) so that literal occurrences
/// inside a translation survive.
pub fn clean_model_output(output: &str) -> &str {
    let mut output = output;
    if let Some(rest) = output.strip_prefix("<|channel>thought") {
        output = match rest.split_once("<channel|>") {
            Some((_, answer)) => answer,
            None => rest.trim_start_matches(['\n', ' ']),
        };
    }

    let mut output = output.trim();
    while let Some(rest) = output
        .strip_suffix("<end_of_turn>")
        .or_else(|| output.strip_suffix("<eos>"))
    {
        output = rest.trim_end();
    }
    output
}

const TERMINAL_PUNCTUATION: [char; 6] = ['!', '?', '.', ',', ';', '。'];

/// Align trailing punctuation and letter case of `translation` with the
/// source text. Chat messages often omit a final full stop, and LLMs tend to
/// add one.
pub fn match_source_style(source: &str, translation: &str) -> String {
    let source = source.trim();
    let mut result = translation.trim().to_owned();
    let (Some(source_last), Some(result_last)) =
        (source.chars().next_back(), result.chars().next_back())
    else {
        return result;
    };

    let source_punct = TERMINAL_PUNCTUATION.contains(&source_last);
    let result_punct = TERMINAL_PUNCTUATION.contains(&result_last);
    if source_punct && !result_punct {
        result.push(source_last);
    } else if !source_punct && result_punct {
        result.pop();
        result.truncate(result.trim_end().len());
    } else if source_punct
        && source_last != result_last
        && source_last.is_ascii()
        && result_last.is_ascii()
    {
        // e.g. "!" flattened to "."; leave script-specific marks like "。" alone.
        result.pop();
        result.push(source_last);
    }

    // Preserve SHOUTING. All-lowercase input is deliberately not forced onto
    // the output: that would break German noun capitalisation and "I".
    let shouting = source.chars().filter(|c| c.is_uppercase()).count() >= 2
        && !source.chars().any(char::is_lowercase);
    if shouting {
        result = result.to_uppercase();
    }

    // Sentence-initial capital: only upgrade, never lowercase.
    if let (Some(s0), Some(r0)) = (source.chars().next(), result.chars().next())
        && s0.is_uppercase()
        && r0.is_lowercase()
    {
        let upper: String = r0.to_uppercase().collect();
        result.replace_range(..r0.len_utf8(), &upper);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cleanup_only_removes_control_markers_at_the_edges() {
        assert_eq!(
            clean_model_output("literal <channel|> content"),
            "literal <channel|> content"
        );
        assert_eq!(clean_model_output("a <end_of_turn> b"), "a <end_of_turn> b");
        assert_eq!(
            clean_model_output(" translation<end_of_turn>\n"),
            "translation"
        );
        assert_eq!(
            clean_model_output("<|channel>thought\nreasoning<channel|>answer"),
            "answer"
        );
        assert_eq!(clean_model_output("<|channel>thought answer"), "answer");
    }

    #[test]
    fn trailing_punctuation_follows_source() {
        assert_eq!(
            match_source_style("Hallo Welt", "Hello world."),
            "Hello world"
        );
        assert_eq!(
            match_source_style("Hallo Welt!", "Hello world"),
            "Hello world!"
        );
        assert_eq!(
            match_source_style("Wie geht's?", "How are you?"),
            "How are you?"
        );
        // Different but valid punctuation is left alone.
        assert_eq!(match_source_style("你好。", "Hello."), "Hello.");
    }

    #[test]
    fn keeps_german_noun_capitalisation_for_lowercase_input() {
        assert_eq!(
            match_source_style("the house is big", "Das Haus ist groß"),
            "Das Haus ist groß"
        );
        assert_eq!(match_source_style("ich gehe", "I'm going"), "I'm going");
    }

    #[test]
    fn preserves_all_caps_across_spaces_and_punctuation() {
        assert_eq!(
            match_source_style("HELLO WORLD!", "Hallo Welt."),
            "HALLO WELT!"
        );
        assert_eq!(match_source_style("OK 123", "Добре 123"), "ДОБРЕ 123");
        assert_eq!(match_source_style("123", "123"), "123");
    }

    #[test]
    fn capitalises_first_letter_like_source() {
        assert_eq!(match_source_style("Hallo", "привіт"), "Привіт");
    }
}
