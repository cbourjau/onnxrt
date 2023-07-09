use ndarray::{ArcArray, IxDyn};

mod attribute;
mod edges;
mod opset;
mod python_api;
mod session;
mod utils;

pub(crate) use attribute::{Attribute, Attributes};
pub use edges::Value;
pub(crate) use edges::{
    channel, value_channel, DataType, Dispatchable, Dispatcher, EnumReceiver, ValueSender,
};
pub use opset::get_opset;
pub use session::Session;

pub type ArcArrayD<T> = ArcArray<T, IxDyn>;

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::{get_opset, Session, Value};
    use ndarray::array;

    #[test]
    fn test_tree_ensemble_regressor() {
        let wire_dict = get_opset();
        let session = Session::from_file(
            "../../../onnxruntime/onnxruntime/test/testdata/tree_ensemble_as_tensor.onnx",
            &wire_dict,
        )
        .unwrap();

        let input = array![
            [0., 1., 1.7999999523162842, 3.],
            [0., 1., 1.7999999523, 3.],
            [0., 1., 1.79999995232, 3.],
        ]
        .into_dyn()
        .into();
        let inputs = HashMap::from_iter([("X".to_string(), Value::F64(input))]);

        let expected: HashMap<_, _> = [(
            "variable".to_string(),
            Value::F64(
                array![
                    [0.7284910678863525],
                    [0.7284910678863525],
                    [0.9134243130683899]
                ]
                .into_dyn()
                .into(),
            ),
        )]
        .into_iter()
        .collect();
        let got = session.run(&inputs).unwrap();
        assert_eq!(got, expected);
    }
}
