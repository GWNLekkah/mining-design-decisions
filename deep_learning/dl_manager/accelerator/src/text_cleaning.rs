use lazy_static::lazy_static;
use regex::Regex;
use crate::text_cleaning::blocks::{remove_code_blocks, remove_no_format_blocks};
use crate::text_cleaning::heuristic_class_names::{remove_class_names_heuristically, remove_file_paths_heuristically, replace_class_names_no_path_heuristically};
use crate::text_cleaning::heuristic_logs::remove_unformatted_lines;
use crate::text_cleaning::jira_formatting::{remove_lists_from_text, remove_simple_jira_formatting};
use crate::text_cleaning::misc::{remove_dates, remove_empty_lines, remove_ip_addresses, remove_links};

mod jira_formatting;
mod markers;
mod heuristic_logs;
mod misc;
mod regex_util;
mod heuristic_class_names;
mod blocks;


#[derive(PartialEq, Eq)]
#[derive(Copy, Clone)]
pub enum FormattingHandling {
    Keep, Remove, Markers
}

pub fn clean_text(mut text: String, handling: FormattingHandling) -> String {
    lazy_static! {
        static ref PATTERN: Regex = Regex::new(r"\{code:\w+\}").unwrap();
    }
    if handling != FormattingHandling::Keep {
        text = remove_unformatted_lines(text, handling);
    }
    text = remove_dates(text, handling);
    text = remove_ip_addresses(text, handling);
    text = remove_links(text, handling);

    if handling == FormattingHandling::Keep {
        text = text.replace("{code}", "");
        text = PATTERN.replace(&text, "").into();
        text = text.replace("{noformat}", "");
    } else {
        text = remove_code_blocks(text, handling);
        text = remove_no_format_blocks(text, handling);
    }

    text = remove_simple_jira_formatting(text, handling);
    text = remove_empty_lines(text);
    text = remove_lists_from_text(text);
    text = remove_file_paths_heuristically(text, handling);
    text = remove_class_names_heuristically(text, handling);
    text = replace_class_names_no_path_heuristically(text, handling);
    text = remove_empty_lines(text);
    text
}
