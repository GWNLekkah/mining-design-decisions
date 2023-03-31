use lazy_static::lazy_static;
use regex::{Captures, Regex, RegexSet};
use crate::text_cleaning::FormattingHandling;
use crate::text_cleaning::markers::Marker;
use crate::text_cleaning::regex_util::is_full_match;


pub fn remove_empty_lines(text: String) -> String {
    text
        .lines()
        .filter(|line| !is_useless_line(*line))
        .collect::<Vec<_>>()
        .join("\n")
}

fn is_useless_line(line: &str) -> bool {
    line.chars().all(
        |c| c.is_whitespace() || c.is_ascii_punctuation()
    )
}


pub fn remove_dates(text: String, handling: FormattingHandling) -> String {
    if handling == FormattingHandling::Keep {
        return text;
    }
    lazy_static! {
        static ref PATTERN: Regex = Regex::new(
            r"(\d\d\d\d[./]\d\d?[./]\d\d?)|(\d\d?[./]\d\d?[./]\d\d\d\d)"
        ).unwrap();
    }
    PATTERN.replace_all(
        &text,
        |_: &Captures| match handling {
            FormattingHandling::Keep => unreachable!(),
            FormattingHandling::Remove => "".into(),
            FormattingHandling::Markers => Marker::Date.string_marker()
        }
    ).into()
}

pub fn remove_ip_addresses(text: String, handling: FormattingHandling) -> String {
    if handling == FormattingHandling::Keep {
        return text;
    }
    lazy_static! {
        static ref PATTERN: Regex = Regex::new(
            r"(\d{2,3}|xx)\.(\d{2,3}|xx)\.(\d{2,3}|xx)\.(\d{1,3}|xx)"
        ).unwrap();
    }
    PATTERN.replace_all(
        &text,
        |_: &Captures| match handling {
            FormattingHandling::Keep => unreachable!(),
            FormattingHandling::Remove => "".into(),
            FormattingHandling::Markers => Marker::IPAddress.string_marker()
        }
    ).into()
}

pub fn remove_links(mut text: String, handling: FormattingHandling) -> String {
    lazy_static! {
        static ref LINK_1: Regex = Regex::new(
            r"\[(?P<linkpart>(http|https|mailto|file).*?\|.*?)\]"
        ).unwrap();
        static ref LINK_2: Regex = Regex::new(
            r"\[(?P<linkpart>(#|\^|http|https|mailto|file).*?)\]"
        ).unwrap();
        static ref LINK_3: Regex = Regex::new(
            r"https?://\S+"
        ).unwrap();
    }
    text = LINK_1.replace_all(
        &text,
        |c: &Captures| {
            remove_link(c.get(1).expect("Expected a match").as_str(), handling)
        }
    ).into();
    text = LINK_2.replace_all(
        &text,
        |c: &Captures| {
            remove_link(c.get(1).expect("Expected a match").as_str(), handling)
        }
    ).into();
    text = LINK_3.replace_all(
        &text,
        |c: &Captures| {
            remove_link(c.get(0).expect("Expected a match").as_str(), handling)
        }
    ).into();
    text
}

fn remove_link(link: &str, handling: FormattingHandling) -> String {
    lazy_static! {
        static ref HTTP_VERSION_PATTERN: Regex = Regex::new(r"http/\d.\d").unwrap();
    }
    match handling {
        FormattingHandling::Keep => link.to_string(),
        FormattingHandling::Remove => "".to_string(),
        FormattingHandling::Markers => {
            if link.starts_with('^') {
                Marker::Attachment.string_marker()
            } else if is_full_match(&HTTP_VERSION_PATTERN, link) {
                link.to_string()
            } else if link.starts_with("https://github.com") {
                Marker::GithubLink.string_marker()
            } else if link.starts_with("https://issues.apache.org/jira/browse/") {
                Marker::IssueLink.string_marker()
            } else {
                Marker::WebLink.string_marker()
            }
        }
    }
}
