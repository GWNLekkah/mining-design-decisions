use pyo3::prelude::*;
use std::collections::HashMap;
use std::collections::HashSet;
use std::error::Error;
use pyo3::exceptions::PyValueError;
use rayon::prelude::*;

mod pos;
mod text_cleaning;

use pos::PerceptronTagger;
use crate::text_cleaning::{clean_text, FormattingHandling};

#[pyclass]
pub struct Tagger {
    p: PerceptronTagger
}

pub fn create_pool(num_threads: usize) -> PyResult<rayon::ThreadPool> {
   match rayon::ThreadPoolBuilder::new()
      .num_threads(num_threads)
      .build()
   {
      Err(e) => Err(PyValueError::new_err(e.to_string())),
      Ok(pool) => Ok(pool),
   }
}

#[pymethods]
impl Tagger {
    #[new]
    pub fn __new__(weights: HashMap<String, HashMap<String, f64>>,
                   classes: HashSet<String>,
                   tagdict: HashMap<String, String>) -> Self {
        Tagger{p: PerceptronTagger::from_weights_and_classes(weights, classes, tagdict)}
    }

    pub fn tag(&self, sentence: Vec<String>) -> Vec<(String, String)> {
        self.p.tag(sentence)
    }

    pub fn bulk_tag(&self, documents: Vec<Vec<Vec<String>>>) -> Vec<Vec<Vec<(String, String)>>> {
        documents
            .into_iter()
            .map(
                |document|
                    document
                        .into_iter()
                        .map(|sentence| self.p.tag(sentence))
                        .collect())
            .collect()
    }

    pub fn bulk_tag_parallel(&self, documents: Vec<Vec<Vec<String>>>, num_threads: usize) -> PyResult<Vec<Vec<Vec<(String, String)>>>> {
        Ok(
            create_pool(num_threads)?.install(|| {
                documents.into_par_iter().map(
                |document|
                    document
                        .into_iter()
                        .map(|sentence| self.p.tag(sentence))
                        .collect())
                .collect()
            })
        )
    }
}

#[pyfunction]
fn bulk_clean_text_parallel(documents: Vec<String>,
                            formatting_handling: String,
                            num_threads: usize) -> PyResult<Vec<String>> {
    let handling = if formatting_handling == "keep" {
        FormattingHandling::Keep
    } else if formatting_handling == "remove" {
        FormattingHandling::Remove
    } else if formatting_handling == "markers" {
        FormattingHandling::Markers
    } else {
        return Err(PyValueError::new_err("Invalid formatting handling mode"));
    };
    Ok(
        create_pool(num_threads)?.install(|| {
            documents.into_par_iter().map(
                |document| clean_text(document, handling)
            )
        }).collect()
    )
}

#[pymodule]
fn accelerator(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<Tagger>()?;
    m.add_function(wrap_pyfunction!(bulk_clean_text_parallel, m)?)?;
    Ok(())
}
