use pyo3::prelude::*;


#[pymodule]
fn accelerator(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    Ok(())
}