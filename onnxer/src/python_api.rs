use std::collections::HashMap;

use numpy::IntoPyArray;
use numpy::PyArrayDyn;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::opset::get_opset;
use crate::{Session, Value};

#[pymodule]
pub fn onnxer(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<Session>()?;
    Ok(())
}

#[pymethods]
impl Session {
    #[new]
    pub fn py_new(path: &str) -> PyResult<Self> {
        Self::from_file(path, &get_opset()).map_err(|e| PyValueError::new_err(format!("{:?}", e)))
    }

    #[pyo3(name = "run")]
    pub fn py_run<'py>(&self, inputs: &'py PyDict) -> PyResult<&'py PyDict> {
        let py = inputs.py();
        let inputs: HashMap<String, Value> = inputs.extract().unwrap();

        let mut outs = self.run(&inputs).unwrap();

        let py_outs: Vec<(String, PyObject)> = self
            .outputs
            .iter()
            .map(|(k, _)| (k, outs.remove(k).unwrap()))
            .map(|(k, v)| match v {
                Value::F64(v) => (k.to_string(), v.to_owned().into_pyarray(py).into_py(py)),
                Value::F32(v) => (k.to_string(), v.to_owned().into_pyarray(py).into_py(py)),
                Value::I64(v) => (k.to_string(), v.to_owned().into_pyarray(py).into_py(py)),
                Value::I32(v) => (k.to_string(), v.to_owned().into_pyarray(py).into_py(py)),
                Value::U64(v) => (k.to_string(), v.to_owned().into_pyarray(py).into_py(py)),
                Value::U32(v) => (k.to_string(), v.to_owned().into_pyarray(py).into_py(py)),
                Value::String(v) => (
                    k.to_string(),
                    v.mapv(|el| el.into_py(py))
                        .to_owned()
                        .into_pyarray(py)
                        .into_py(py),
                ),
                Value::Bool(v) => (k.to_string(), v.to_owned().into_pyarray(py).into_py(py)),
            })
            .collect();
        PyDict::from_sequence(py, py_outs.into_py(py))
    }

    #[pyo3(name = "input_types")]
    pub fn py_input_types(&self) -> HashMap<String, String> {
        let hm: HashMap<String, String> = self
            .input_dtypes()
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_standard_name().to_string()))
            .collect();
        hm
    }
}

// This is a bit wasteful since we are making copies of all the inputs...
impl<'source> FromPyObject<'source> for Value {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        // Get the GIL marker.
        // TODO: convert scalars and strings
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
            let py = ob.py();
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
