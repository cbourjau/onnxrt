use crate::opset::utils::make_binary_numeric;

make_binary_numeric!(setup_add, |&a, &b| a + b);
