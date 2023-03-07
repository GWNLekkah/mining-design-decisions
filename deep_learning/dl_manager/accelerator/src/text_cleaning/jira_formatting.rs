use lazy_static::lazy_static;
use regex::{Captures, Regex, RegexSet};
use crate::text_cleaning::FormattingHandling;
use crate::text_cleaning::heuristic_class_names::determine_type;
use crate::text_cleaning::markers::Marker;
use crate::text_cleaning::regex_util::is_full_match;


pub fn remove_simple_jira_formatting(mut text: String, handling: FormattingHandling) -> String {
    lazy_static! {
        static ref PATTERNS: [Regex; 14] = [
            Regex::new(r"\[\~[^\s]+\]").unwrap(),
            Regex::new(r"!.*?!").unwrap(),
            Regex::new(r"h[1-6]\.").unwrap(),
            Regex::new(r"\|").unwrap(),
            Regex::new(r"bq\.").unwrap(),
            Regex::new(r"<.*?>").unwrap(),
            Regex::new(r"\{quote\}").unwrap(),
            Regex::new(r"\*(?P<content>.+?)\*").unwrap(),
            Regex::new(r"_(?P<content>.+?)_").unwrap(),
            Regex::new(r"\?\?(?P<content>.+?)\?\?").unwrap(),
            Regex::new(r"\+(?P<content>.+?)\+").unwrap(),
            Regex::new(r"\^(?P<content>.+?)\^").unwrap(),
            Regex::new(r"~(?P<content>.+?)~").unwrap(),
            Regex::new(r"\{color:.+?\}(?P<content>.*?)\{color\}").unwrap(),
        ];
        static ref U: String = Marker::UserProfileLink.string_marker();
        static ref I: String = Marker::ImageAttachment.string_marker();
        static ref REPLACEMENTS: [&'static str; 14] = [
            &U,
            &I,
            "",
            " | ",
            "",
            "",
            " ",
            "$content",
            "$content",
            "$content",
            "$content",
            "$content",
            "$content",
            "$content"
        ];
        static ref INLINE_CODE_PATTERN: Regex = Regex::new(
            r"\{\{(?P<code>.*?)\}\}"
        ).unwrap();
        static ref CLASS_NAME_PATTERN: Regex = Regex::new(
            r"[a-zA-Z\d_\-\.:#]+(\(.*\))?"
        ).unwrap();
    }
    for i in 0..PATTERNS.len() {
        let pattern = &PATTERNS[i];
        let repl = REPLACEMENTS[i];
        text = pattern.replace_all(&text, repl).into();
    }
    match handling {
        FormattingHandling::Keep => {
            text = INLINE_CODE_PATTERN.replace_all(&text, "$code").into();
        }
        FormattingHandling::Remove => {
            text = INLINE_CODE_PATTERN.replace_all(&text, "").into();
        }
        FormattingHandling::Markers => {
            text = INLINE_CODE_PATTERN.replace_all(
                &text,
                |c: &Captures| {
                    let match_text = c.get(1).expect("Expected a match").as_str();
                    let is_maybe_name = is_full_match(&CLASS_NAME_PATTERN, match_text);
                    if is_maybe_name {
                        determine_type(match_text, handling)
                    } else {
                        Marker::InlineCode.string_marker()
                    }
                }
            ).into();
        }
    }
    text
}


pub fn remove_lists_from_text(text: String) -> String {
    let mut parts: Vec<String> = Vec::new();
    let mut capitalize_next = false;
    for in_line in text.lines() {
        let capitalize =  capitalize_next;
        let (mut line, capitalize_next) = remove_list_item(in_line);
        if capitalize {
            line = capitalize_string(line);
        }
        if !line.trim().is_empty() {
            parts.push(line);
        }
    }
    parts.join("\n")
}

fn remove_list_item(line: &str) -> (String, bool) {
    let stripped = line.trim();
    let mut remainder = stripped
        .chars()
        .skip_while(|c| *c == '*' || *c == '-' || *c == '#')
        .collect::<String>();
    if remainder.len() == stripped.len() { // No symbols removed
        remove_list_item_heuristic(line)
    } else {
        remainder = capitalize_string(remainder);
        if !remainder.ends_with('.') {
            (remainder + ".", true)
        } else {
            (remainder, true)
        }
    }
}

fn remove_list_item_heuristic(line: &str) -> (String, bool) {
    lazy_static! {
        static ref PATTERN: Regex = Regex::new(r"\d+[.)](?P<payload>.+)").unwrap();
    }
    let m = PATTERN.find_at(line, 0);
    if let Some(mat) = m {
        if mat.start() == 0 && mat.end() == line.len() - 1 {
            let mut remainder = mat
                .as_str()
                .chars()
                .skip_while(|c| c.is_numeric())
                .collect::<String>();
            remainder = if let Some(x) = remainder.strip_prefix(')') {
                x.to_string()
            } else {
                remainder
            };
            if !remainder.ends_with('.') {
                return (remainder + ".", true);
            } else {
                return (remainder, true);
            }
        }
    }
    (line.to_string(), false)
}


fn capitalize_string(s: String) -> String {
    let mut c = s.chars();
    match c.next() {
        None => String::new(),
        Some(f) => f.to_uppercase().chain(c).collect(),
    }
}
