use lazy_static::lazy_static;
use regex::{Regex, RegexSet};
use crate::text_cleaning::FormattingHandling;
use crate::text_cleaning::markers::Marker;


fn remove_unformatted_lines(text: String, handling: FormattingHandling) -> String {
    if handling == FormattingHandling::Keep {
        return text;
    }
    lazy_static! {
        static ref LOG_PATTERN: Regex = Regex::new(
            r"LLLOG(\s*LLLOG)*"     // Marker::Log
        ).unwrap();
        static ref TB_PATTERN: Regex = Regex::new(
            r"TTTRACEBACK(\s*TTTRACEBACK)*"     // Marker::Traceback
        ).unwrap();
    }
    let trimmed = text
        .lines()
        .map(|line| handle_log_or_traceback(line.to_string(), handling))
        .collect::<Vec<_>>()
        .join( "\n");
    if handling == FormattingHandling::Markers {
        let replaced = LOG_PATTERN.replace(&trimmed,
                                           Marker::UnformattedLog.string_marker());
        replaced = TB_PATTERN.replace(&replaced,
                                      Marker::UnformattedTraceback.string_marker());
        replaced.into()
    } else {
        trimmed
    }
}

fn handle_log_or_traceback(line: String, handling: FormattingHandling) -> String {
     if line_is_log(&line) {
         match handling {
             FormattingHandling::Keep => line,
             FormattingHandling::Remove => "".into(),
             FormattingHandling::Markers => Marker::Log.string_marker()
         }
     } else if line_is_traceback(&line) {
         match handling {
             FormattingHandling::Keep => line,
             FormattingHandling::Remove => "".into(),
             FormattingHandling::Markers => Marker::Traceback.string_marker()
         }
     } else {
         line
     }
}

fn line_is_log(line: &String) -> bool {
    lazy_static! {
        static ref PATTERN_SET: RegexSet = RegexSet::new(
            [
                r"\s*\d\d/\d\d/\d\d \d\d:\d\d:\d\d (DEBUG|INFO|WARNING|WARN|ERROR|CRITICAL|SEVERE) .*\s*",
                r"\s*\[(DEBUG|INFO|WARNING|WARN|ERROR|CRITICAL|SEVERE)\] .*\s*",
                r"\s*\d\d\d\d-\d\d-\d\d \d\d:\d\d:\d\d,\d\d\d (DEBUG|INFO|WARNING|WARN|ERROR|CRITICAL|SEVERE) .*\s*",
                r"\s*(DEBUG|INFO|WARNING|WARN|ERROR|CRITICAL|SEVERE) .*? \d\d\d\d-\d\d-\d\d \d\d:\d\d:\d\d,\d\d\d .*\s*",
                r"\s*[A-Z][a-z]{,2} \d\d?, \d\d\d\d \d\d?:\d\d?:\d\d? (AM|PM) .*?\s*",
                r"\s*(DEBUG|INFO|WARNING|WARN|ERROR|CRITICAL|SEVERE): .*\s*",
                r"\s*ERROR - .*\s*",
                r"\s*INFO .*\s*",
            ]
        ).unwrap();
    }
    PATTERN_SET.is_match(line)
}

fn line_is_traceback(line: &String) -> bool {
    lazy_static! {
        static ref PATTERN_SET: RegexSet = RegexSet::new(
            [
                r"\s*at [$#]?\w+([\.$#]\w+)*\(.*\)\s*",
                r"\s*Caused by: \w+(\.\w+)*: .*\s*",
                r"\s*(\w+\.)*\w+(Error|Exception)(: (\w+\.)*\w+(Error|Exception))*\s*"
            ]
        ).unwrap();
    }
    PATTERN_SET.is_match(line)
}