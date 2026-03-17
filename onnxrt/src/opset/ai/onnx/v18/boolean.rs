use crate::opset::utils::*;

make_binary_boolean!(setup_and, |&a, &b| a & b);
make_binary_boolean!(setup_or, |&a, &b| a | b);
make_binary_boolean!(setup_xor, |&a, &b| a ^ b);
