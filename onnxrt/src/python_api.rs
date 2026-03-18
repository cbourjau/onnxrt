// Disable some lints until pyo3 is up-to-date
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(non_local_definitions)]

use std::collections::HashMap;

use half::f16;
use numpy::{IntoPyArray, PyArrayMethods, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use numpy::{PyUntypedArray, ToPyArray};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::types::PyDict;
use pyo3::types::PyString;
use pyo3::{IntoPyObjectExt, prelude::*};

use crate::opset::get_opset;
use crate::{Session, Value};

#[pymodule]
mod onnxrt {
    #[pymodule_export]
    use super::Session;
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
    pub fn py_run<'py>(&'py self, inputs: &Bound<'py, PyDict>) -> PyResult<Bound<'py, PyDict>> {
        let py = inputs.py();
        let inputs: HashMap<String, Value> = inputs.extract().unwrap();

        let mut outs = self.run_named(&inputs).unwrap();
        let out_dict = PyDict::new(py);

        for (k, _) in &self.outputs {
            let value = outs.remove(k).unwrap();
            let v = value.into_pyobject(py)?;
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
        self.outputs.iter().map(|(k, _)| k.to_string()).collect()
    }
}
// This is a bit wasteful since we are making copies of all the inputs...
impl<'py> FromPyObject<'_, 'py> for Value {
    type Error = PyErr;
    fn extract(ob: pyo3::Borrowed<'_, 'py, pyo3::PyAny>) -> Result<Self, Self::Error> {
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
        // } else if let Ok(arr) = ob.extract::<PyReadonlyArrayDyn<String>>() {
        // let py = ob.py();
        // let arr = arr.as_array().mapv(|el| el.extract(py).unwrap());
        // Value::String(arr.into())
        // todo!()
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

impl<'py> IntoPyObject<'py> for Value {
    type Target = PyAny; // the Python type
    type Output = Bound<'py, Self::Target>; // in most cases this will be `Bound`
    type Error = std::convert::Infallible;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let res = match self {
            Value::Bool(v) => v.to_owned().into_pyarray(py).into_any(),
            Value::U8(v) => v.to_owned().into_pyarray(py).into_any(),
            Value::U16(v) => v.to_owned().into_pyarray(py).into_any(),
            Value::U32(v) => v.to_owned().into_pyarray(py).into_any(),
            Value::U64(v) => v.to_owned().into_pyarray(py).into_any(),
            Value::I8(v) => v.to_owned().into_pyarray(py).into_any(),
            Value::I16(v) => v.to_owned().into_pyarray(py).into_any(),
            Value::I32(v) => v.to_owned().into_pyarray(py).into_any(),
            Value::I64(v) => v.to_owned().into_pyarray(py).into_any(),
            Value::F16(v) => to_np_float16(py, v)?.into_any(),
            Value::F32(v) => v.to_owned().into_pyarray(py).into_any(),
            Value::F64(v) => v.to_owned().into_pyarray(py).into_any(),
            Value::String(v) => v
                .mapv(|item| item.into_py_any(py).unwrap())
                .into_pyarray(py)
                .into_any(),
        };
        Ok(res)
    }
}

fn to_np_float16<'py>(
    py: Python<'py>,
    arr: ndarray::ArcArray<f16, ndarray::IxDyn>,
) -> Result<Bound<'py, PyAny>, std::convert::Infallible> {
    let raw = arr.mapv(|x| x.to_bits());
    let py_arr = raw.to_pyarray(py);

    let np = py.import("numpy").unwrap();
    let res = py_arr
        .call_method1("view", (np.getattr("float16").unwrap(),))
        .unwrap();

    Ok(res)
}
