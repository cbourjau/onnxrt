use anyhow::{Result, anyhow, bail};
use onnx_protos::onnx_proto3::{ValueInfoProto, tensor_proto, type_proto};
use protobuf::Enum;

/// Data types which may be exchange between nodes.
#[derive(Debug, PartialEq, Copy, Clone)]
pub enum DataType {
    Bool,
    F16,
    F32,
    F64,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    String,
}

#[derive(Debug, PartialEq, Copy, Clone)]
pub struct TensorType(pub DataType);

#[derive(Debug, PartialEq, Copy, Clone)]
pub enum SequenceType {
    Tensor(TensorType),
}

#[derive(Debug, PartialEq, Copy, Clone)]
pub struct MapType {
    pub key: DataType,
    pub value: DataType,
}

#[derive(Debug, PartialEq, Copy, Clone)]
pub enum ValueType {
    Tensor(TensorType),
    Sequence(SequenceType),
    Map(MapType),
}

impl DataType {
    pub fn from_i32(type_number: i32) -> Result<Self> {
        use tensor_proto::DataType::*;
        let dtype_proto = tensor_proto::DataType::from_i32(type_number)
            .ok_or_else(|| anyhow!("Invalid type number: {}", type_number))?;
        Ok(match dtype_proto {
            // Defined in tensor_proto::DataType (cf onnx.proto)
            BOOL => DataType::Bool,
            DOUBLE => DataType::F64,
            FLOAT16 => DataType::F16,
            FLOAT => DataType::F32,
            INT8 => DataType::I8,
            INT16 => DataType::I16,
            INT32 => DataType::I32,
            INT64 => DataType::I64,
            STRING => DataType::String,
            UINT8 => DataType::U8,
            UINT16 => DataType::U16,
            UINT32 => DataType::U32,
            UINT64 => DataType::U64,
            val => bail!("Unsupported type `{:?}`", val),
        })
    }
}

impl ValueType {
    pub fn from_value_info(value_info: &ValueInfoProto) -> Result<Self> {
        let err_fn = || {
            anyhow!(
                "Type information is missing from `ValueInfo`: [information hidden for brevity]",
            )
        };
        let val_type_info = value_info
            .type_
            .as_ref()
            .ok_or_else(err_fn)?
            .value
            .as_ref()
            .ok_or_else(err_fn)?;

        use type_proto as typro;
        match val_type_info {
            typro::Value::TensorType(typro::Tensor {
                elem_type: type_number,
                ..
            }) => Ok(Self::Tensor(TensorType(DataType::from_i32(*type_number)?))),

            // typro::Value::SequenceType(typro::Sequence { elem_type, .. }) => {
            //     match elem_type.0.as_ref().map(|el| el.value.clone()).flatten() {
            //         Some(typro::Value::TensorType(typro::Tensor {
            //             elem_type: type_number,
            //             ..
            //         })) => Ok(Self::Tensor(TensorType(DataType::from_i32(type_number)?))),
            //         _ => bail!("'Sequence' value type is not supported"),
            //     }
            // }

            // typro::Value::OptionalType(_) => {
            //     bail!("'Optional' value type is not supported")
            // }

            // typro::Value::MapType(typro::Map { .. }) => {
            //     bail!("'Map' value type is not supported")
            // }
            val_type_info => bail!("Unsupported input type: {:?}", val_type_info),
        }
    }

    pub fn standard_name(&self) -> &str {
        match self {
            Self::Tensor(TensorType(dtype)) => match dtype {
                DataType::Bool => "tensor(bool)",

                DataType::I8 => "tensor(int8)",
                DataType::I16 => "tensor(int16)",
                DataType::I32 => "tensor(int32)",
                DataType::I64 => "tensor(int64)",

                DataType::U8 => "tensor(uint8)",
                DataType::U16 => "tensor(uint16)",
                DataType::U32 => "tensor(uint32)",
                DataType::U64 => "tensor(uint64)",

                DataType::F16 => "tensor(half)",
                DataType::F32 => "tensor(float)",
                DataType::F64 => "tensor(double)",

                DataType::String => "tensor(string)",
            },
            _ => panic!(),
        }
    }
}
