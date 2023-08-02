#[macro_export]
macro_rules! setup_all {
    (TaggedReceiver, $tagged_rec:expr, $rec_name:tt, $setup:expr) => {
        setup_all!(
            TaggedReceiver, $tagged_rec, $rec_name, $setup,
            F32, F64, I8, I16, I32, I64, U8, U16, U32, U64, Bool, String
        )
    };
    (Numeric, $tagged_rec:expr, $rec_name:tt, $setup:expr) => {
        setup_all!(
            Numeric, $tagged_rec, $rec_name, $setup,
            F32, F64, I8, I16, I32, I64, U8, U16, U32, U64
        )
    };
    (Homogeneous, $tagged_rec:expr, $rec_name:tt, $setup:expr) => {
        setup_all!(
            Homogeneous, $tagged_rec, $rec_name, $setup,
            F32, F64, I8, I16, I32, I64, U8, U16, U32, U64, Bool, String
        )
    };
    (HomogeneousFixed, $tagged_rec:expr, $rec_name:tt, $setup:expr) => {
        setup_all!(
            HomogeneousFixed, $tagged_rec, $rec_name, $setup,
            F32, F64, I8, I16, I32, I64, U8, U16, U32, U64, Bool, String
        )
    };
    (HomogeneousNumeric, $tagged_rec:expr, $rec_name:tt, $setup:expr) => {
        setup_all!(
            HomogeneousNumeric, $tagged_rec, $rec_name, $setup,
            F32, F64, I8, I16, I32, I64, U8, U16, U32, U64
        )
    };
    (HomogeneousNumericFixed, $tagged_rec:expr, $rec_name:tt, $setup:expr) => {
        setup_all!(
            HomogeneousNumericFixed, $tagged_rec, $rec_name, $setup,
            F32, F64, I8, I16, I32, I64, U8, U16, U32, U64
        )
    };
    // Common implementation
    ($enum:tt, $tagged_rec:expr, $rec_name:tt, $setup:expr, $($variant:tt),*) => {
        match $tagged_rec {
            $($enum::$variant($rec_name) => $setup,)*
        }
    };
}

/// Create a setup function for a simple binary operation which takes
/// two inputs of identical type.
#[macro_export]
macro_rules! binary_op_factory {
    // Special case booleans for now since the other varians use
    // HomogeneousNumericFixed under the hood
    (boolean, $fun:expr) => {
        Box::new(move |ctx: OpContext| -> Result<ExecutionStep> {
            let (a_ch, b_ch) = ctx.subscribe_to_inputs().narrow()?;
            use TaggedReceiver::*;
            match (a_ch, b_ch) {
                (Bool(a_ch), Bool(b_ch)) => setup((a_ch, b_ch), move |(a, b)| $fun(a, b)),
                // Internal error. This should never happen!
                _ => panic!(),
            }
        })
    };
    (int, $fun:expr) => {
        binary_op_factory!(
            $fun,
            I8, I16, I32, I64, U8, U16, U32, U64 |
            rec => bail!("Expected integer input, found: '{:?}'", rec)
        )
    };
    (numeric, $fun:expr) => {
        binary_op_factory!(
            $fun,
            F32, F64, I8, I16, I32, I64, U8, U16, U32, U64 |
        )
    };
    (float, $fun:expr) => {
        binary_op_factory!(
            $fun,
            F32, F64 |
            rec => bail!("Expected floating point input, found: '{:?}'", rec)
        )
    };
    (signed, $fun:expr) => {
        binary_op_factory!(
            $fun,
            I32, I64, F32, F64 |
            rec => bail!("Expected signed input, found: '{:?}'", rec)
        )
    };
    ($fun:expr, $($variant:tt),* | $($rec:tt => $bail:expr)?) => {
        Box::new(move |ctx: OpContext| -> Result<ExecutionStep> {
            let inputs: HomogeneousNumericFixed<2> = ctx.subscribe_to_inputs().narrow()?;
            use HomogeneousNumericFixed::*;
            match inputs {
                $($variant([a_ch, b_ch]) => {
                    setup((a_ch, b_ch), move |(a, b)| $fun(a, b))
                })*,
                $($rec => $bail)?
            }
        })
    };
}

#[macro_export]
macro_rules! bin_numeric_with_co_broadcast {
    ($fun:expr) => {
        binary_op_factory!(numeric, |a, b| zip_map_with_co_broadcast(a, b, $fun))
    };
}
