mod jira_formatting;
mod markers;
mod heuristic_logs;


#[derive(PartialEq, Eq)]
pub enum FormattingHandling {
    Keep, Remove, Markers
}

pub fn clean_text(text: String, handling: FormattingHandling) -> String {
    text
}

fn fix_punctuation(text: String) -> String {

}