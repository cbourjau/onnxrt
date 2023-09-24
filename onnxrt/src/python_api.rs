use std::collections::HashMap;

use itertools::Itertools;
use numpy::PyReadonlyArrayDyn;
use numpy::ToPyArray;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::opset::get_opset;
use crate::{Session, Value};

#[pymodule]
pub fn onnxrt(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<Session>()?;
    Ok(())
}

#[pymethods]
impl Session {
    #[new]
    pub fn py_new(path: &str) -> PyResult<Self> {
        Self::from_file(path, &get_opset()).map_err(|e| PyValueError::new_err(format!("{:?}", e)))
    }

    #[staticmethod]
    pub fn py_from_bytes(model_bytes: Vec<u8>) -> PyResult<Self> {
        Self::from_bytes(&model_bytes, &get_opset())
            .map_err(|e| PyValueError::new_err(format!("{:?}", e)))
    }

    #[pyo3(name = "run")]
    pub fn py_run<'py>(&self, inputs: &'py PyDict) -> PyResult<&'py PyDict> {
        let py = inputs.py();
        let inputs: HashMap<String, Value> = inputs.extract().unwrap();

        let mut outs = self.run_named(&inputs).unwrap();
        let out_dict = PyDict::new(py);

        for (k, _) in &self.outputs {
            let v = match outs.remove(k).unwrap() {
                Value::F32(v) => v.view().to_pyarray(py).to_object(py),
                Value::F64(v) => v.view().to_pyarray(py).to_object(py),

                Value::I8(v) => v.view().to_pyarray(py).to_object(py),
                Value::I16(v) => v.view().to_pyarray(py).to_object(py),
                Value::I32(v) => v.view().to_pyarray(py).to_object(py),
                Value::I64(v) => v.view().to_pyarray(py).to_object(py),

                Value::U8(v) => v.view().to_pyarray(py).to_object(py),
                Value::U16(v) => v.view().to_pyarray(py).to_object(py),
                Value::U32(v) => v.view().to_pyarray(py).to_object(py),
                Value::U64(v) => v.view().to_pyarray(py).to_object(py),

                Value::String(v) => v
                    .mapv(|el| el.to_object(py))
                    .view()
                    .to_pyarray(py)
                    .to_object(py),
                Value::Bool(v) => v.view().to_pyarray(py).to_object(py),
            };
            out_dict.set_item(k, v)?;
        }

        Ok(out_dict)
    }

    #[pyo3(name = "input_types")]
    pub fn py_input_types(&self) -> HashMap<String, String> {
        let hm: HashMap<String, String> = self
            .input_dtypes()
            .iter()
            .map(|(k, v)| (k.to_string(), v.standard_name().to_string()))
            .collect();
        hm
    }

    #[getter(input_names)]
    pub fn py_input_names(&self) -> Vec<String> {
       self.inputs.0.keys().map(|k| k.to_string()).collect()
    }

    #[getter(output_names)]
    pub fn py_output_names(&self) -> Vec<String> {
        self
            .outputs
            .iter()
            .map(|(k, _)| k.to_string())
            .collect()
    }
}

// This is a bit wasteful since we are making copies of all the inputs...
impl<'source> FromPyObject<'source> for Value {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        // Get the GIL marker.
        // TODO: convert scalars and strings
        Ok(if let Ok(arr) = ob.extract::<PyReadonlyArrayDyn<f32>>() {
            Value::F32(arr.as_array().to_owned().into())
        } else if let Ok(arr) = ob.extract::<PyReadonlyArrayDyn<f64>>() {
            Value::F64(arr.as_array().to_owned().into())
        } else if let Ok(arr) = ob.extract::<PyReadonlyArrayDyn<i32>>() {
            Value::I32(arr.as_array().to_owned().into())
        } else if let Ok(arr) = ob.extract::<PyReadonlyArrayDyn<i64>>() {
            Value::I64(arr.as_array().to_owned().into())
        } else if let Ok(arr) = ob.extract::<PyReadonlyArrayDyn<PyObject>>() {
            let py = ob.py();
            let arr = arr.as_array().mapv(|el| el.extract(py).unwrap());
            Value::String(arr.into())
        } else if let Ok(arr) = ob.extract::<PyReadonlyArrayDyn<bool>>() {
            Value::Bool(arr.as_array().to_owned().into())
        } else {
            return Err(PyTypeError::new_err(format!(
                "Cannot convert `{:?}` to Rust type.",
                ob
            )));
        })
    }
}
