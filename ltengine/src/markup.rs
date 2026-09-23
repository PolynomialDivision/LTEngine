//! HTML ↔ Markdown conversion for models that keep Markdown formatting but
//! not HTML tags. TranslateGemma rewrites `<b>x</b>` as `**x**`, so HTML
//! requests are translated as Markdown and converted back.

use pulldown_cmark::{Options, Parser, html::push_html};

/// Convert (Matrix-style) HTML to Markdown.
pub fn html_to_markdown(html: &str) -> String {
    htmd::convert(html).unwrap_or_else(|_| html.to_owned())
}

/// Convert translated Markdown back to HTML. A single paragraph is returned
/// without a `<p>` wrapper unless the original HTML had one, matching how
/// Matrix clients format one-line messages.
pub fn markdown_to_html(markdown: &str, original_html: &str) -> String {
    let options = Options::ENABLE_STRIKETHROUGH | Options::ENABLE_TABLES;
    let mut html = String::with_capacity(markdown.len() * 2);
    push_html(&mut html, Parser::new_ext(markdown, options));
    let html = html.trim_end();

    if !original_html.contains("<p")
        && let Some(inner) = html
            .strip_prefix("<p>")
            .and_then(|h| h.strip_suffix("</p>"))
        && !inner.contains("<p>")
    {
        return inner.to_owned();
    }
    html.to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inline_formatting_round_trips() {
        let md = html_to_markdown("Das ist <strong>wichtig</strong> und <em>schön</em>!");
        assert_eq!(md, "Das ist **wichtig** und *schön*!");
        assert_eq!(
            markdown_to_html(
                "This is **important** and _nice_!",
                "Das ist <strong>wichtig</strong>"
            ),
            "This is <strong>important</strong> and <em>nice</em>!"
        );
    }

    #[test]
    fn links_code_and_lists_survive() {
        let html = r#"Siehe <a href="https://example.org">Doku</a> und <code>cargo build</code><ul><li>eins</li><li>zwei</li></ul>"#;
        let md = html_to_markdown(html);
        assert!(md.contains("[Doku](https://example.org)"), "{md}");
        assert!(md.contains("`cargo build`"), "{md}");
        let back = markdown_to_html(&md, html);
        assert!(
            back.contains(r#"<a href="https://example.org">Doku</a>"#),
            "{back}"
        );
        assert!(back.contains("<li>eins</li>"), "{back}");
    }

    #[test]
    fn paragraph_wrapper_follows_original() {
        assert_eq!(markdown_to_html("Hallo", "<p>Hi</p>"), "<p>Hallo</p>");
        assert_eq!(markdown_to_html("Hallo", "<b>Hi</b>"), "Hallo");
    }
}
