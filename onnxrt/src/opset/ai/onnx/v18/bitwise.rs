use crate::opset::utils::make_binary_integer;

make_binary_integer!(setup_bitwise_and, |&a, &b| a & b);
make_binary_integer!(setup_bitwise_or, |&a, &b| a | b);
make_binary_integer!(setup_bitwise_xor, |&a, &b| a ^ b);
