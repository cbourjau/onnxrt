use std::collections::HashMap;

use numpy::IntoPyArray;
use numpy::PyArrayDyn;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::opset::get_opset;
use crate::{Session, Value};

#[pymodule]
pub fn onnxruntime_rs(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<Session>()?;
    Ok(())
}

#[pymethods]
impl Session {
    #[new]
    pub fn py_new(path: &str) -> PyResult<Self> {
        Self::from_file(path, &get_opset()).map_err(|_e| PyValueError::new_err("cannot be zero"))
    }

    #[pyo3(name = "run")]
    pub fn py_run(&self, inputs: &PyDict) -> HashMap<String, PyObject> {
        let py = inputs.py();
        let inputs: HashMap<&str, Value> = inputs.extract().unwrap();

        self.run(&inputs)
            .unwrap()
            .iter()
            .map(|(k, v)| match v {
                Value::F64(v) => (k.to_string(), v.to_owned().into_pyarray(py).into_py(py)),
                _ => panic!(),
            })
            .collect()
    }
}

// This is a bit wasteful since we are making copies of all the inputs...
impl<'source> FromPyObject<'source> for Value {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        // Get the GIL marker.
        let py = ob.py();
        Ok(if let Ok(arr) = ob.extract::<&PyArrayDyn<f32>>() {
            let arr = arr.readonly();
            Value::F32(arr.as_array().to_owned().into())
        } else if let Ok(arr) = ob.extract::<&PyArrayDyn<f64>>() {
            let arr = arr.readonly();
            Value::F64(arr.as_array().to_owned().into())
        } else if let Ok(arr) = ob.extract::<&PyArrayDyn<i32>>() {
            let arr = arr.readonly();
            Value::I32(arr.as_array().to_owned().into())
        } else if let Ok(arr) = ob.extract::<&PyArrayDyn<i64>>() {
            let arr = arr.readonly();
            Value::I64(arr.as_array().to_owned().into())
        } else if let Ok(arr) = ob.extract::<&PyArrayDyn<PyObject>>() {
            let arr = arr.readonly().as_array().mapv(|el| el.extract(py).unwrap());
            Value::String(arr.into())
        } else if let Ok(arr) = ob.extract::<&PyArrayDyn<bool>>() {
            let arr = arr.readonly();
            Value::Bool(arr.as_array().to_owned().into())
        } else {
            return Err(PyTypeError::new_err(format!(
                "Cannot convert `{:?}` to Rust type.",
                ob
            )));
        })
    }
}
