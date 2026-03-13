# onnxrt

A proof-of-concept runtime for ONNX models written in Rust.

## Goals

- Competitive performance compared to other CPU based inference implementations
- Readability over minor optimizations
- Easy to contribute
- If possible, operator implementation should only depend on [`ndarray`](https://docs.rs/ndarray/0.15.6/ndarray/index.html)
- Operators are implemented for all types mandated by the standard which are also supported by onnxrt

## Out of scope for now

- Any kind of API stability
- Multi-threading
- C-API
- Multiple execution providers (and thus GPU support)

## Standard coverage

The coverage is still fairly small. Onnxrt passes around 40% of the upstream test, but that metric is not very useful.

### Supported types

Currently, onnxrt only supports a subset of the [standard types](https://github.com/onnx/onnx/blob/main/docs/IR.md#input--output-data-types):

- Tensors
  - `String`
  - `bool`
  - `f32`
  - `f64`
  - `i16`
  - `i32`
  - `i64`
  - `i8`
  - `u16`
  - `u32`
  - `u64`
  - `u8`
- Maps
  - Unsupported
- Optional
  - Unsupported
- Sequence
  - Unsupported
