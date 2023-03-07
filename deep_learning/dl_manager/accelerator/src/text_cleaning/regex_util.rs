use regex::{Captures, Regex};


#[inline(always)]
pub fn is_full_match(p: &Regex, text: &str) -> bool {
    let m = p.find_at(text, 0);
    if let Some(mat) = m {
        mat.end() == text.len() - 1
    } else {
        false
    }
}
