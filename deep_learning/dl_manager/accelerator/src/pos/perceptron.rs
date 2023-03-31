use std::collections::{HashMap, HashSet};
use super::averaged_perceptron::AveragedPerceptron;

pub struct PerceptronTagger {
    model: AveragedPerceptron,
    tagdict: HashMap<String, String>,
    classes: HashSet<String>
}

macro_rules! add {
    ($f:ident, $($item:expr),+) => {
        {
            let key = vec![$($item),*].join(" ");
            $f.insert(key, 1.0);
        }
    };
}

macro_rules! suffix {
    ($string:expr, $size:expr) => {
        {
            let i = $size;
            let text = $string;
            if text.len() <= i {
                text.to_string()
            } else {
                text[text.len() - i..].to_string()
            }
        }
    }
}

impl PerceptronTagger {
    pub fn from_weights_and_classes(weights: HashMap<String, HashMap<String, f64>>,
                                    classes: HashSet<String>,
                                    tagdict: HashMap<String, String>) -> Self {
        PerceptronTagger{
            model: AveragedPerceptron::from_weights_and_classes(weights, classes.clone()),
            classes,
            tagdict
        }
    }

    pub fn tag(&self, tokens: Vec<String>) -> Vec<(String, String)> {
        let mut prev: String = "-START-".into();
        let mut prev2: String = "-START2-".into();
        let mut output = Vec::with_capacity(tokens.len());
        let mut context: Vec<String> = Vec::with_capacity(tokens.len() + 4);
        context.extend_from_slice(&["-START-".into(), "-START2-".into()]);
        context.extend(tokens.iter().map(|t| self.normalize(t)));
        context.extend_from_slice(&["-END-".into(), "-END2-".into()]);
        for (i, word) in tokens.iter().cloned().enumerate() {
            let tag = match self.tagdict.get(&word) {
                Some(x) => x.clone(),
                None => {
                    let features = self.make_features(i, word, &context, &prev, &prev2);
                    self.model.predict(features)
                }
            };
            output.push(tag.clone());
            prev2 = prev;
            prev = tag;
        }
        tokens.into_iter().zip(output.into_iter()).collect()
    }

    fn normalize(&self, word: &String) -> String {
        if word.contains('-') && !word.starts_with('-') {
            "!HYPEN".into()
        } else if word.chars().all(char::is_numeric) {
            if word.len() == 4 {
                "!YEAR".into()
            } else {
                "!DIGITS".into()
            }
        } else {
            word.to_lowercase()
        }
    }

    fn make_features(&self, mut i: usize, word: String, context: &[String], prev: &String, prev2: &String) -> HashMap<String, f64> {
        let mut features: HashMap<String, f64> = HashMap::new();
        i += 2;
        add!(features, "bias");
        //add!(features, "i suffix", if word.len() > 3 { word[word.len()-3..].into() } else { &word });
        add!(features, "i suffix", &suffix!(&word, 3));
        add!(features, "i pref1", if !word.is_empty() { word[0..1].into() } else { "" });
        add!(features, "i-1 tag", prev);
        add!(features, "i-2 tag", prev2);
        add!(features, "i tag+i-2 tag", prev, prev2);
        add!(features, "i word", &context[i]);
        add!(features, "i-1 tag+i word", prev, &context[i]);
        add!(features, "i-1 word", &context[i-1]);
        //add!(features, "i-1 suffix", if context[i-1].len() > 3 { &context[i-1][context[i-1].len()-3..] } else { &context[i-1] });
        add!(features, "i-1 suffix", &suffix!(&context[i-1], 3));
        add!(features, "i-2 word", &context[i-2]);
        add!(features, "i+1 word", &context[i+1]);
        //add!(features, "i+1 suffix", if context[i+1].len() > 3 { &context[i+1][context[i+1].len()-3..] } else { &context[i+1] });
        add!(features, "i+1 suffix", &suffix!(&context[i+1], 3));
        add!(features, "i+2 word", &context[i+2]);
        features
    }
}

// fn suffix(text: &str, size: usize) -> &str {
//     if text.len() <= size {
//         text
//     } else {
//         text[text.len() - size..]
//     }
// }

