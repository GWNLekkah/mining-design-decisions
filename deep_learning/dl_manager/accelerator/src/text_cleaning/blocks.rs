use lazy_static::lazy_static;
use regex::Regex;
use crate::text_cleaning::FormattingHandling;
use crate::text_cleaning::heuristic_logs::remove_unformatted_lines;
use crate::text_cleaning::markers::Marker;


pub fn remove_code_blocks(mut text: String, handling: FormattingHandling) -> String {
    lazy_static! {
        static ref START_PATTERN: Regex = Regex::new(r"\{code:.*?\}").unwrap();
        static ref GENERIC_PATTERN: Regex = Regex::new(r"\{code\}").unwrap();
    }
    let start_markers = START_PATTERN.find_iter(&text);
    let generic_markers = GENERIC_PATTERN.find_iter(&text);
    let mut markers = start_markers
        .into_iter()
        .map(|m| (m, true))
        .chain(
            generic_markers
                .into_iter()
                .map(|m| (m, false)))
        .collect::<Vec<_>>();
    markers.sort_by_key(|(m, _) | m.start());
    let mut index: usize = 0;
    let mut removals: Vec<(usize, usize)> = Vec::new();
    while index + 1 < markers.len() {
        let (start, start_is_pure) = markers[index];
        let (end, end_is_pure) = markers[index + 1];
        if end_is_pure {
            index += 1;
            markers[index] = (start, start_is_pure);
            continue;
        }
        removals.push((start.start(), end.end()));
        index += 2;
    }
    if index + 1 < markers.len() {
        let (marker, is_pure) = markers[index+1];
        removals.push((marker.start(), text.len()));
    }
    removals.reverse();
    for (start, stop) in removals {
        if handling == FormattingHandling::Remove {
            if stop + 1 >= text.len() {
                text = text[0..start].to_string();
            } else {
                text = text[0..start].to_string() + &text[stop+1..text.len()];
            }
        } else {
            assert!(handling == FormattingHandling::Markers);
            let marker = guess_marker(&text[start..stop], Marker::StructuredCodeBlock);
            if stop + 1 >= text.len() {
                text = text[0..start].to_string() + " " + &marker;
            } else {
                text = text[0..start].to_string() + " " + &marker + " " + &text[stop+1..text.len()];
            }
        }
    }
    text
}


pub fn remove_no_format_blocks(mut text: String, handling: FormattingHandling) -> String {
    lazy_static! {
        static ref PATTERN: Regex = Regex::new(r"\{noformat\}").unwrap();
    }
    let mut markers = PATTERN
        .find_iter(&text)
        .into_iter()
        .enumerate()
        .map(|(i, m)| if i % 2 == 0 { m.start() } else { m.end() })
        .collect::<Vec<_>>();
    if markers.len() % 2 == 1 {
        markers.push(text.len());
    }
    let mut nums = (0..markers.len() / 2).collect::<Vec<usize>>();
    nums.reverse();
    for i in nums {
        let start = markers[2*i];
        let stop = markers[2*i + 1];
        if handling == FormattingHandling::Remove {
            if stop + 1 >= text.len() {
                text = text[0..start].to_string();
            } else {
                text = text[0..start].to_string() + &text[stop+1..text.len()];
            }
        } else {
            assert!(handling == FormattingHandling::Markers);
            if stop > text.len() {
                println!("YO WHAT THE FUCK. LEN IS {}, but stop is {}", text.len(), stop);
            }
            let marker = guess_marker(&text[start..stop], Marker::NoFormatBlock);
            if stop + 1 >= text.len() {
                text = text[0..start].to_string() + " " + &marker;
            } else {
                text = text[0..start].to_string() + " " + &marker + " " + &text[stop+1..text.len()];
            }
        }
    }
    text
}


fn guess_marker(text: &str, default: Marker) -> String {
    let stripped = remove_unformatted_lines(text.to_string(), FormattingHandling::Markers);
    if stripped.contains(&Marker::UnformattedLog.string_marker()) {
        Marker::FormattedLogging.string_marker()
    } else if stripped.contains(&Marker::UnformattedTraceback.string_marker()) {
        Marker::FormattedTraceback.string_marker()
    } else {
        default.string_marker()
    }
}
