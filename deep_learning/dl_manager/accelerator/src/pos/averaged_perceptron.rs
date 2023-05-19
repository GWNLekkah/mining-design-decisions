use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::collections::hash_map::Entry;

pub struct AveragedPerceptron {
    weights: HashMap<String, HashMap<String, f64>>,
    classes: HashSet<String>,
    totals: HashMap<(String, String), i64>,
    timestamps: HashMap<(String, String), i64>,
    i: i64
}

impl AveragedPerceptron {

    pub fn from_weights_and_classes(weights: HashMap<String, HashMap<String, f64>>,
                                    classes: HashSet<String>) -> Self {
        AveragedPerceptron{
            weights,
            classes,
            totals: HashMap::new(),
            timestamps: HashMap::new(),
            i: 0
        }
    }

    fn softmax(scores: Vec<f64>) -> Vec<f64> {
        let exps = scores.iter().map(|f| f.exp()).collect::<Vec<_>>();
        let denom = exps.iter().sum::<f64>();
        exps.into_iter().map(|f| f / denom).collect()
    }

    fn predict_internal(&self, features: HashMap<String, f64>) -> (String, HashMap<String, f64>) {
        let mut scores: HashMap<String, f64> = self.classes
            .iter()
            .map(|c| (c.clone(), 0.0))
            .collect();
        for (feature, value) in features.into_iter() {
            if !self.weights.contains_key(&feature) || value == 0.0 {
                continue;
            }
            for (label, weight) in self.weights.get(&feature).expect("Weight key not present") {
                match scores.entry(label.clone()) {
                    Entry::Occupied(mut e) => {
                        e.insert(*e.get() + value*weight);
                    },
                    _ => unreachable!("Entry must be occupied")
                }
            }
        }
        let best_label = self.classes
            .iter()
            .max_by(
                |a, b| {
                    let x = scores.get(*a).expect("!");
                    let y = scores.get(*b).expect("!");
                    // For backwards compatability with habrok,
                    // we copy the implementation for f64::total_cmp
                    match total_cmp_floats(*x, *y) {
                        Ordering::Less => Ordering::Less,
                        Ordering::Greater => Ordering::Greater,
                        Ordering::Equal => a.cmp(b)
                    }
                })
            .expect("Maximum should be present")
            .clone();

        (best_label, scores)
    }

    pub fn predict(&self, features: HashMap<String, f64>) -> String {
        let (cls, _) = self.predict_internal(features);
        cls
    }

    fn predict_with_confidence(&self, features: HashMap<String, f64>) -> (String, f64) {
        let (cls, scores) = self.predict_internal(features);
        let best_score = scores
            .values()
            .into_iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        (cls, best_score)
    }
}


fn total_cmp_floats(x: f64, y: f64) -> Ordering {
    let mut left = x.to_bits() as i64;
    let mut right = y.to_bits() as i64;
    left ^= (((left >> 63) as u64) >> 1) as i64;
    right ^= (((right >> 63) as u64) >> 1) as i64;
    left.cmp(&right)
}
