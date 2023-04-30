use std::collections::HashMap;

use anyhow::{anyhow, bail, Result};
use ndarray::Array;
use protobuf::MessageField;

use crate::{ArcArrayD, Value};
use onnx_protos::onnx_proto3::{
    attribute_proto::AttributeType, AttributeProto, GraphProto, NodeProto, TensorProto,
};

// Vectors should be ArcArrayD
pub enum Attribute {
    F32s(ArcArrayD<f32>),
    F64s(ArcArrayD<f64>),
    I64s(ArcArrayD<i64>),
    I32s(ArcArrayD<i32>),
    Bools(ArcArrayD<bool>),

    ScalarF32(f32),
    ScalarI64(i64),
    ScalarString(String),
    Strings(ArcArrayD<String>),
    TensorUnknown(Vec<u8>),
    // Graphs may have inputs from the enclosing scope; Outputs are
    // handled through the node.
    Graph(GraphProto),
}

impl std::fmt::Debug for Attribute {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::result::Result<(), std::fmt::Error> {
        macro_rules! vec_debug {
            ($variant:literal, $values:expr) => {{
                fmt.write_str($variant)?;
                fmt.write_str("(")?;
                let mut dbg_list = fmt.debug_list();
                dbg_list.entries($values.iter().take(10));
                if $values.len() > 10 {
                    dbg_list.entry(&"...");
                }
                dbg_list.finish()?;
                fmt.write_str(")")?;
            }};
        }
        match self {
            Self::Bools(values) => vec_debug!("Attribute::Bools", values),
            Self::F32s(values) => vec_debug!("Attribute::F32s", values),
            Self::F64s(values) => vec_debug!("Attribute::F64s", values),
            Self::Graph(_) => write!(fmt, "Attribute::Graph")?,
            Self::I32s(values) => vec_debug!("Attribute::I32s", values),
            Self::I64s(values) => vec_debug!("Attribute::I64s", values),
            Self::ScalarF32(val) => write!(fmt, "Attribute::ScalarF32({})", val)?,
            Self::ScalarI64(val) => write!(fmt, "Attribute::ScalarI64({})", val)?,
            Self::ScalarString(val) => write!(fmt, "Attribute::ScalarString({})", val)?,
            Self::Strings(values) => vec_debug!("Attribute::Strings", values),
            Self::TensorUnknown(_) => write!(fmt, "Attribute::TensorUnknown")?,
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct Attributes(HashMap<String, Attribute>);

impl Attributes {
    /// Panics if the node proto is in an invalid state.
    pub fn from_node(node: &NodeProto) -> Result<Self> {
        let mut hm = HashMap::new();
        for attr_proto in &node.attribute {
            let attr = attr_proto.try_into()?;
            hm.insert(attr_proto.name.to_string(), attr);
        }
        Ok(Attributes(hm))
    }

    pub fn get_with_default<'s, T>(&'s self, name: &str, default: T) -> Result<T>
    where
        T: TryFrom<&'s Attribute, Error = anyhow::Error>,
    {
        if let Some(attr) = self.0.get(name) {
            attr.try_into()
        } else {
            Ok(default)
        }
    }

    pub fn get<'s>(&'s self, name: &str) -> Result<&'s Attribute> {
        self.0
            .get(name)
            .ok_or_else(|| anyhow!("No attribute of name {}", name))
    }

    /// Try to get an optional attribute.
    /// An `Err` is returned if the attribute exists but is of the wrong type.
    pub fn get_optional<'s, T>(&'s self, name: &str) -> Result<Option<T>>
    where
        T: TryFrom<&'s Attribute, Error = anyhow::Error>,
    {
        match self.0.get(name) {
            Some(attr) => attr.try_into().map(Some),
            None => Ok(None),
        }
    }

    /// Get graph attributes
    pub fn get_graphs(&self) -> Vec<&GraphProto> {
        self.0
            .iter()
            .filter_map(|(_, v)| match v {
                Attribute::Graph(g) => Some(g),
                _ => None,
            })
            .collect()
    }
}

macro_rules! try_scalar_from_attr {
    ($variant:tt, $ty:ty, $ty_name:literal) => {
        impl TryFrom<&Attribute> for $ty {
            type Error = anyhow::Error;

            fn try_from(value: &Attribute) -> Result<Self> {
                if let Attribute::$variant(out) = value {
                    Ok(out.to_owned())
                } else {
                    Err(anyhow!("Attribute is not of type: {}", $ty_name))
                }
            }
        }
    };
}

macro_rules! try_array_from_attr {
    ($variant:tt, $ty:ty, $ty_name:literal) => {
        impl TryFrom<&Attribute> for $ty {
            type Error = anyhow::Error;

            fn try_from(value: &Attribute) -> Result<Self> {
                if let Attribute::$variant(out) = value {
                    Ok(out.clone())
                } else {
                    Err(anyhow!("Attribute is not of type: {}", $ty_name))
                }
            }
        }
    };
}

macro_rules! try_vec_from_attr {
    ($variant:tt, $ty:ty, $ty_name:literal) => {
        impl TryFrom<&Attribute> for $ty {
            type Error = anyhow::Error;

            fn try_from(value: &Attribute) -> Result<Self> {
                if let Attribute::$variant(out) = value {
                    Ok(out.clone().into_owned().into_raw_vec())
                } else {
                    Err(anyhow!("Attribute is not of type: {}", $ty_name))
                }
            }
        }
    };
}

try_scalar_from_attr!(ScalarI64, i64, "i64");
try_scalar_from_attr!(ScalarF32, f32, "f32");
try_scalar_from_attr!(ScalarString, String, "String");
try_vec_from_attr!(F32s, Vec<f32>, "Vec<f32>");
try_vec_from_attr!(F64s, Vec<f64>, "Vec<f64>");
try_vec_from_attr!(I32s, Vec<i32>, "Vec<i32>");
try_vec_from_attr!(I64s, Vec<i64>, "Vec<i64>");
try_vec_from_attr!(Strings, Vec<String>, "Vec<String>");
try_array_from_attr!(F32s, ArcArrayD<f32>, "Vec<f32>");
try_array_from_attr!(I64s, ArcArrayD<i64>, "Vec<i64>");
try_array_from_attr!(Strings, ArcArrayD<String>, "Vec<String>");
try_scalar_from_attr!(Graph, GraphProto, "GraphProto");

impl TryFrom<&AttributeProto> for Attribute {
    type Error = anyhow::Error;

    fn try_from(ap: &AttributeProto) -> Result<Self> {
        let ty = ap.type_.unwrap();
        Ok(match (ty, ap) {
            (AttributeType::FLOAT, AttributeProto { f, .. }) => Attribute::ScalarF32(*f),
            (AttributeType::INT, AttributeProto { i, .. }) => Attribute::ScalarI64(*i),
            (AttributeType::STRING, AttributeProto { s, .. }) => {
                Attribute::ScalarString(String::from_utf8(s.clone())?)
            }
            (AttributeType::INTS, AttributeProto { ints, .. }) => {
                Attribute::I64s(Array::from_vec(ints.clone()).into_dyn().into())
            }
            (AttributeType::FLOATS, AttributeProto { floats, .. }) => {
                Attribute::F32s(Array::from_vec(floats.clone()).into_dyn().into())
            }
            (AttributeType::STRINGS, AttributeProto { strings, .. }) => {
                let strings = strings
                    .iter()
                    .map(|chunk| Ok(String::from_utf8(chunk.to_owned())?))
                    .collect::<Result<Vec<_>>>()?;
                Attribute::Strings(Array::from_vec(strings).into_dyn().into())
            }
            (
                AttributeType::TENSOR,
                AttributeProto {
                    t: MessageField(Some(tp)),
                    ..
                },
            ) => tp.as_ref().try_into()?,
            (
                AttributeType::GRAPH,
                AttributeProto {
                    g: MessageField(Some(gp)),
                    ..
                },
            ) => Attribute::Graph(*gp.clone()),
            (ty, ap) => bail!("Cannot handle type `{:#?}` for `{:#?}`", ty, ap),
        })
    }
}

macro_rules! attr_from_vec {
    ($ty:ty, $variant:tt) => {
        impl From<Vec<$ty>> for Attribute {
            fn from(vec: Vec<$ty>) -> Self {
                Attribute::$variant(Array::from_vec(vec.clone()).into_dyn().into())
            }
        }
    };
}
attr_from_vec!(f32, F32s);
attr_from_vec!(f64, F64s);
attr_from_vec!(i32, I32s);
attr_from_vec!(i64, I64s);

impl TryFrom<&TensorProto> for Attribute {
    type Error = anyhow::Error;

    fn try_from(tensor: &TensorProto) -> Result<Self, Self::Error> {
        let value = tensor.try_into()?;
        match value {
            Value::F32(v) => Ok(Attribute::F32s(v)),
            Value::F64(v) => Ok(Attribute::F64s(v)),
            Value::I32(v) => Ok(Attribute::I32s(v)),
            Value::I64(v) => Ok(Attribute::I64s(v)),
            Value::String(v) => Ok(Attribute::Strings(v)),
            Value::Bool(v) => Ok(Attribute::Bools(v)),
        }
    }
}

impl TryFrom<&TensorProto> for Value {
    type Error = anyhow::Error;

    fn try_from(tensor: &TensorProto) -> Result<Self, Self::Error> {
        use crate::DataType;
        use std::mem::size_of;

        macro_rules! read_raw {
            ($ty:ty) => {{
                assert_eq!(tensor.raw_data.len() % size_of::<$ty>(), 0);
                tensor
                    .raw_data
                    .chunks_exact(size_of::<$ty>())
                    .map(|slice| slice.try_into().unwrap())
                    .map(<$ty>::from_le_bytes)
            }};
        }
        let shape: Vec<_> = tensor.dims.iter().map(|v| *v as usize).collect();

        match DataType::from_i32(tensor.data_type)? {
            DataType::F32 => {
                let explicit_field = &tensor.float_data;
                let vec: Vec<_> = if !explicit_field.is_empty() {
                    explicit_field.clone()
                } else {
                    read_raw!(f32).collect::<Vec<_>>()
                };
                Ok(Value::F32(
                    Array::from_vec(vec)
                        .into_dyn()
                        .into_shape(shape)
                        .unwrap()
                        .into(),
                ))
            }
            DataType::F64 => {
                let explicit_field = &tensor.double_data;
                let vec: Vec<_> = if !explicit_field.is_empty() {
                    explicit_field.clone()
                } else {
                    read_raw!(f64).collect::<Vec<_>>()
                };
                Ok(Value::F64(
                    Array::from_vec(vec)
                        .into_dyn()
                        .into_shape(shape)
                        .unwrap()
                        .into(),
                ))
            }
            DataType::I32 => {
                let explicit_field = &tensor.int32_data;
                let vec: Vec<_> = if !explicit_field.is_empty() {
                    explicit_field.clone()
                } else {
                    read_raw!(i32).collect::<Vec<_>>()
                };
                Ok(Value::I32(
                    Array::from_vec(vec)
                        .into_dyn()
                        .into_shape(shape)
                        .unwrap()
                        .into(),
                ))
            }
            DataType::I64 => {
                let explicit_field = &tensor.int64_data;
                let vec: Vec<_> = if !explicit_field.is_empty() {
                    explicit_field.clone()
                } else {
                    read_raw!(i64).collect::<Vec<_>>()
                };
                Ok(Value::I64(
                    Array::from_vec(vec)
                        .into_dyn()
                        .into_shape(shape)
                        .unwrap()
                        .into(),
                ))
            }
            DataType::Bool => {
                let explicit_field = &tensor.int32_data;
                let vec: Vec<_> = if !explicit_field.is_empty() {
                    explicit_field.iter().map(|&el| el != 0).collect()
                } else {
                    read_raw!(u8).map(|v| v > 0).collect()
                };
                Ok(Value::Bool(
                    Array::from_vec(vec)
                        .into_dyn()
                        .into_shape(shape)
                        .unwrap()
                        .into(),
                ))
            }
            DataType::String => {
                let explicit_field = &tensor.string_data;
                // Strings are always in the explicit field (from proto3 source)
                let vec: Vec<_> = explicit_field
                    .clone()
                    .into_iter()
                    .map(|u8_vec| String::from_utf8(u8_vec).unwrap())
                    .collect();
                Ok(Value::String(
                    Array::from_vec(vec)
                        .into_dyn()
                        .into_shape(shape)
                        .unwrap()
                        .into(),
                ))
            }
        }
    }
}

// enum TensorData {
//     F32(Vec<f32>),
//     F64(Vec<f64>),
//     I32(Vec<i32>),
//     I64(Vec<i64>),
//     U64(Vec<u64>),
//     String(Vec<String>),
//     Raw(Vec<u8>),
//     External,
// }

// fn parse(tp: TensorProto) -> Result<TensorData> {
//     if !tp.float_data.is_empty() {
//         Ok(TensorData::F32(tp.float_data))
//     } else if !tp.double_data.is_empty() {
//         Ok(TensorData::F64(tp.double_data))
//     } else if !tp.int32_data.is_empty() {
//         Ok(TensorData::I32(tp.int32_data))
//     } else if !tp.int64_data.is_empty() {
//         Ok(TensorData::I64(tp.int64_data))
//     } else if !tp.uint64_data.is_empty() {
//         Ok(TensorData::U64(tp.uint64_data))
//     } else if !tp.string_data.is_empty() {
//         Ok(TensorData::String(
//             tp.string_data
//                 .into_iter()
//                 .map(|v| String::from_utf8(v).map_err(|e| anyhow!("{:?}", e)))
//                 .collect::<Result<_>>()?,
//         ))
//     } else if !tp.raw_data.is_empty() {
//         Ok(TensorData::Raw(tp.raw_data))
//     } else if !tp.external_data.is_empty() {
//         bail!("External data is not supported.")
//     } else {
//         bail!("Empty tensor proto")
//     }
// }
