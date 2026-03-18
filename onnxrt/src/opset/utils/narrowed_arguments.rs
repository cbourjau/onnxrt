use anyhow::{Result, anyhow, bail};
/// Some helper types for validating inputs.
use flume::Receiver;
use half::f16;

use crate::{ArcArrayD, DataType, TaggedReceiver, TaggedTensorReceiver};

/// Narrow inputs into a smaller subset of expected types
/// This trait should not be implemented direct. Instead,
/// `PartialNarrow` or `PartialNarrowLast` should be implemented which
/// are used by `Narrow` via generic implementations.
pub trait Narrow<To> {
    fn narrow(self) -> Result<To>
    where
        Self: Sized;
}

/// Narrow into a smaller type set consuming a statically known number
/// of input items (i.e. all but variadic input type sets).
trait PartialNarrow<Item> {
    /// Number of items that will be consumed by parse
    const N_ITEMS: usize = 1;

    fn parse<I>(it: &mut I) -> Result<Self>
    where
        Self: Sized,
        I: Iterator<Item = Item>;
}

/// Greedily parse all remaining input items. All types that implement
/// `PartialNarrow` also implement `PartialNarrowLast`, but not vice
/// versa. This trait should be implemented directly for variadic type
/// sets.
trait PartialNarrowLast<Item> {
    fn parse<I>(it: I) -> Result<Self>
    where
        Self: Sized,
        I: Iterator<Item = Item>;
}

///////////// Implementations of `Narrow`

impl<A> Narrow<A> for Vec<TaggedReceiver>
where
    A: PartialNarrowLast<TaggedReceiver>,
{
    fn narrow(self) -> Result<A>
    where
        Self: Sized,
    {
        let mut it = self.into_iter();
        A::parse(&mut it)
    }
}

impl<A> Narrow<Option<A>> for Vec<Option<TaggedReceiver>>
where
    Option<A>: PartialNarrowLast<Option<TaggedReceiver>>,
{
    fn narrow(self) -> Result<Option<A>>
    where
        Self: Sized,
    {
        let mut it = self.into_iter();
        Option::<A>::parse(&mut it)
    }
}

/// Implementation macro for `Narrow` on tuples. Currently up to
/// 6-tuples are supported.
macro_rules! impl_Narrow_for_tuples {
    ($($ty:ident),* | Z) => {
        impl<$($ty,)* Z, T> Narrow<($($ty,)* Z,)> for Vec<T>
        where
            $($ty: PartialNarrow<T>,)*
            Z: PartialNarrowLast<T>,
        {
            #[allow(unused_mut)]
            fn narrow(self) -> Result<($($ty,)* Z,)>
            where
                Self: Sized,
            {
                let mut it = self.into_iter();
                Ok(($($ty::parse(&mut it)?,)* Z::parse(it)?,))
            }
        }
    };
}
impl_Narrow_for_tuples!(| Z);
impl_Narrow_for_tuples!(A | Z);
impl_Narrow_for_tuples!(A, B | Z);
impl_Narrow_for_tuples!(A, B, C | Z);
impl_Narrow_for_tuples!(A, B, C, D | Z);
impl_Narrow_for_tuples!(A, B, C, D, E | Z);

//////////////// Implementations of `PartialNarrow`

impl<T> PartialNarrow<Option<TaggedReceiver>> for T
where
    T: PartialNarrow<TaggedReceiver>,
{
    fn parse<I>(it: &mut I) -> Result<Self>
    where
        I: Iterator<Item = Option<TaggedReceiver>>,
    {
        let tmp: Option<Vec<_>> = it.by_ref().take(T::N_ITEMS).collect();

        let tmp = tmp.ok_or_else(|| anyhow!("Required input was missing"))?;

        if tmp.len() != T::N_ITEMS {
            bail!("insufficient number of inputs A")
        }
        T::parse(&mut tmp.into_iter())
    }
}

impl<T> PartialNarrow<Option<TaggedReceiver>> for Option<T>
where
    T: PartialNarrow<TaggedReceiver>,
{
    fn parse<I>(it: &mut I) -> Result<Self>
    where
        I: Iterator<Item = Option<TaggedReceiver>>,
    {
        let mut items = vec![];
        for _ in 0..T::N_ITEMS {
            // It might be that we exhaust the optional inputs early
            let item = it.next().unwrap_or_else(|| None);
            items.push(item);
        }

        let all_some = items.iter().all(|el| el.is_some());
        let all_none = items.iter().all(|el| el.is_none());

        if all_none || items.is_empty() {
            Ok(None)
        } else if all_some {
            let mut iter = items.into_iter().flatten();
            Ok(Some(T::parse(&mut iter)?))
        } else {
            bail!("Optional inputs parsed as a block must either all be missing or present")
        }
    }
}

impl PartialNarrow<TaggedReceiver> for TaggedTensorReceiver {
    fn parse<I>(it: &mut I) -> Result<Self>
    where
        I: Iterator<Item = TaggedReceiver>,
    {
        let item = it
            .next()
            .ok_or_else(|| anyhow!("insufficient number of inputs B"))?;
        match item {
            TaggedReceiver::Tensor(t) => Ok(t),
            TaggedReceiver::Sequence(_) => bail!("Expected 'tensor' found 'sequence'"),
            TaggedReceiver::Map => bail!("Expected 'tensor' found 'map'"),
        }
    }
}

impl PartialNarrow<TaggedReceiver> for TaggedReceiver {
    fn parse<I>(it: &mut I) -> Result<Self>
    where
        I: Iterator<Item = TaggedReceiver>,
    {
        let item = it
            .next()
            .ok_or_else(|| anyhow!("insufficient number of inputs C"))?;
        Ok(item)
    }
}

macro_rules! impl_PartialNarrow_for_Receiver {
    ($variant:ident, $ty:ty) => {
        impl PartialNarrow<TaggedReceiver> for Receiver<ArcArrayD<$ty>> {
            fn parse<I>(it: &mut I) -> Result<Self>
            where
                I: Iterator<Item = TaggedReceiver>,
            {
                let item = <TaggedReceiver as PartialNarrow<_>>::parse(it)?;
                match item {
                    TaggedReceiver::Tensor(t) => match t {
                        TaggedTensorReceiver::$variant(value) => Ok(value),
                        tr => bail!(
                            "Expected data type '{:?}', found '{:?}'",
                            DataType::$variant,
                            tr.dtype()
                        ),
                    },
                    TaggedReceiver::Sequence(_) => bail!("expected 'tensor' found 'sequence'"),
                    TaggedReceiver::Map => bail!("expected 'tensor' found 'map'"),
                }
            }
        }
    };
}

impl_PartialNarrow_for_Receiver!(Bool, bool);
impl_PartialNarrow_for_Receiver!(F16, f16);
impl_PartialNarrow_for_Receiver!(F32, f32);
impl_PartialNarrow_for_Receiver!(F64, f64);
impl_PartialNarrow_for_Receiver!(I16, i16);
impl_PartialNarrow_for_Receiver!(I32, i32);
impl_PartialNarrow_for_Receiver!(I64, i64);
impl_PartialNarrow_for_Receiver!(I8, i8);
impl_PartialNarrow_for_Receiver!(String, String);
impl_PartialNarrow_for_Receiver!(U16, u16);
impl_PartialNarrow_for_Receiver!(U32, u32);
impl_PartialNarrow_for_Receiver!(U64, u64);
impl_PartialNarrow_for_Receiver!(U8, u8);

/// Map to a subset of tensor types
macro_rules! map_to_sub_set {
    ($item:expr, $from:tt, $to:ty, $msg:expr, $($variant:ident),*) => {
        match $item {
            $(
                $from::$variant(rec) => <$to>::$variant(rec),
            )*
                item => bail!("{}. Found '{:?}'", $msg, item.dtype()),
        }
    }
}

macro_rules! map_to_sub_set_fixed_array {
    ($item:expr, $from:tt, $to:ty, $($msg:literal,)? $($variant:ident),*) => {
        match $item {
            $(
                $from::$variant(rec) => <$to>::$variant(rec.try_into().map_err(|e| anyhow!("{:?}", e))?),
            )*
            $(
                item => bail!("{}. Found '{:?}'", $msg, item.dtype()),
            )?
        }
    }
}

impl PartialNarrow<TaggedReceiver> for Numeric {
    fn parse<I>(it: &mut I) -> Result<Self>
    where
        I: Iterator<Item = TaggedReceiver>,
    {
        let item = <TaggedReceiver as PartialNarrow<_>>::parse(it)?;
        Ok(map_to_sub_set!(
            item.unpack_tensor_receiver()?,
            TaggedTensorReceiver,
            Self,
            "Expected numeric input",
            U8,
            U16,
            U32,
            U64,
            I8,
            I16,
            I32,
            I64,
            F16,
            F32,
            F64
        ))
    }
}

impl PartialNarrow<TaggedReceiver> for Integer {
    fn parse<I>(it: &mut I) -> Result<Self>
    where
        I: Iterator<Item = TaggedReceiver>,
    {
        let item = <TaggedReceiver as PartialNarrow<_>>::parse(it)?;
        Ok(map_to_sub_set!(
            item.unpack_tensor_receiver()?,
            TaggedTensorReceiver,
            Self,
            "Expected integer point input",
            U8,
            U16,
            U32,
            U64,
            I8,
            I16,
            I32,
            I64
        ))
    }
}

impl PartialNarrow<TaggedReceiver> for FloatingPoint {
    fn parse<I>(it: &mut I) -> Result<Self>
    where
        I: Iterator<Item = TaggedReceiver>,
    {
        let item = <TaggedReceiver as PartialNarrow<_>>::parse(it)?;
        Ok(map_to_sub_set!(
            item.unpack_tensor_receiver()?,
            TaggedTensorReceiver,
            Self,
            "Expected floating point input",
            F16,
            F32,
            F64
        ))
    }
}

impl PartialNarrow<TaggedReceiver> for OnnxIndex {
    fn parse<I>(it: &mut I) -> Result<Self>
    where
        I: Iterator<Item = TaggedReceiver>,
    {
        let item = <TaggedReceiver as PartialNarrow<_>>::parse(it)?;
        Ok(map_to_sub_set!(
            item.unpack_tensor_receiver()?,
            TaggedTensorReceiver,
            Self,
            "Expected floating point input",
            I32,
            I64
        ))
    }
}

impl<const N: usize> PartialNarrow<TaggedReceiver> for HomogeneousFixed<N> {
    const N_ITEMS: usize = N;

    fn parse<I>(it: &mut I) -> Result<Self>
    where
        I: Iterator<Item = TaggedReceiver>,
    {
        let mut items = Vec::with_capacity(N);
        for _ in 0..N {
            items.push(
                it.next()
                    .ok_or_else(|| anyhow!("insufficient number of inputs D"))?,
            );
        }
        let item = <Homogeneous as PartialNarrowLast<_>>::parse(items.into_iter())?;
        Ok(map_to_sub_set_fixed_array!(
            item,
            Homogeneous,
            Self,
            U8,
            U16,
            U32,
            U64,
            I8,
            I16,
            I32,
            I64,
            F16,
            F32,
            F64,
            Bool,
            String
        ))
    }
}

impl<const N: usize> PartialNarrow<TaggedReceiver> for HomogeneousNumericFixed<N> {
    const N_ITEMS: usize = N;

    fn parse<I>(it: &mut I) -> Result<Self>
    where
        I: Iterator<Item = TaggedReceiver>,
    {
        let item = <HomogeneousFixed<N> as PartialNarrowLast<_>>::parse(it)?;
        Ok(map_to_sub_set_fixed_array!(
            item,
            HomogeneousFixed,
            Self,
            "Expected numerical input",
            U8,
            U16,
            U32,
            U64,
            I8,
            I16,
            I32,
            I64,
            F16,
            F32,
            F64
        ))
    }
}

impl<const N: usize> PartialNarrow<TaggedReceiver> for HomogeneousPrimIntFixed<N> {
    const N_ITEMS: usize = N;

    fn parse<I>(it: &mut I) -> Result<Self>
    where
        I: Iterator<Item = TaggedReceiver>,
    {
        let item = <HomogeneousFixed<N> as PartialNarrowLast<_>>::parse(it)?;
        Ok(map_to_sub_set_fixed_array!(
            item,
            HomogeneousFixed,
            Self,
            "Expected numerical input",
            U8,
            U16,
            U32,
            U64,
            I8,
            I16,
            I32,
            I64
        ))
    }
}

//////////////// Implementations of `PartialNarrowLast`

// Parse mandatory to mandatory
impl<T> PartialNarrowLast<TaggedReceiver> for T
where
    T: PartialNarrow<TaggedReceiver>,
{
    fn parse<I>(mut it: I) -> Result<Self>
    where
        Self: Sized,
        I: Iterator<Item = TaggedReceiver>,
    {
        let out = T::parse(&mut it)?;
        if it.count() != 0 {
            bail!("All subscribed inputs must be used")
        }
        Ok(out)
    }
}

// Parse optional to optional
impl<T> PartialNarrowLast<Option<TaggedReceiver>> for Option<T>
where
    T: PartialNarrowLast<TaggedReceiver>,
{
    fn parse<I>(it: I) -> Result<Self>
    where
        Self: Sized,
        I: Iterator<Item = Option<TaggedReceiver>>,
    {
        let items: Vec<_> = it.collect();

        let all_some = items.iter().all(|el| el.is_some());
        let all_none = items.iter().all(|el| el.is_none());

        if all_none || items.is_empty() {
            Ok(None)
        } else if all_some {
            let it = items.into_iter().flatten();
            Ok(Some(T::parse(it)?))
        } else {
            bail!("Missing mandatory inputs")
        }
    }
}

impl PartialNarrowLast<Option<TaggedReceiver>> for Vec<Option<TaggedReceiver>> {
    fn parse<I>(it: I) -> Result<Self>
    where
        Self: Sized,
        I: Iterator<Item = Option<TaggedReceiver>>,
    {
        Ok(it.collect())
    }
}

impl PartialNarrowLast<Option<TaggedReceiver>> for Vec<TaggedReceiver> {
    fn parse<I>(it: I) -> Result<Self>
    where
        Self: Sized,
        I: Iterator<Item = Option<TaggedReceiver>>,
    {
        let mut res = vec![];
        for (i, item) in it.enumerate() {
            let item = item
                .ok_or_else(|| anyhow!("The '{}'th variadic input is required but missing", i))?;
            res.push(item);
        }
        Ok(res)
    }
}

impl PartialNarrowLast<Option<TaggedReceiver>> for Vec<TaggedTensorReceiver> {
    fn parse<I>(it: I) -> Result<Self>
    where
        Self: Sized,
        I: Iterator<Item = Option<TaggedReceiver>>,
    {
        let mut res = vec![];
        for (i, item) in it.enumerate() {
            let item = item
                .ok_or_else(|| anyhow!("The '{}'th variadic input is required but missing", i))?;
            res.push(item.unpack_tensor_receiver()?);
        }
        Ok(res)
    }
}

impl PartialNarrowLast<TaggedReceiver> for Homogeneous {
    fn parse<I>(mut it: I) -> Result<Self>
    where
        I: Iterator<Item = TaggedReceiver>,
    {
        let first = it
            .next()
            .ok_or_else(|| anyhow!("insufficient number of inputs E"))?;

        macro_rules! match_all_variants {
            ($($variant:tt),* ) => {
                match first.unpack_tensor_receiver()? {
                    $(
                    TaggedTensorReceiver::$variant(first) => {
                        let mut items = vec![first];
                        for item in it {
                            let item = item.unpack_tensor_receiver()?;
                            if let TaggedTensorReceiver::$variant(item) = item {
                                items.push(item);
                            } else {
                                bail!(
                                    "Expected homogeneous input of data type '{:?}' found '{:?}'",
                                    stringify!($variant),
                                    item.dtype()
                                )
                            }
                        }
                        Ok(Self::$variant(items))
                    },
                )*}
            };
        }
        match_all_variants!(
            U8, U16, U32, U64, I8, I16, I32, I64, F16, F32, F64, Bool, String
        )
    }
}

impl PartialNarrowLast<TaggedReceiver> for HomogeneousNumeric {
    fn parse<I>(it: I) -> Result<Self>
    where
        I: Iterator<Item = TaggedReceiver>,
    {
        let item = <Homogeneous as PartialNarrowLast<_>>::parse(it)?;

        Ok(map_to_sub_set_fixed_array!(
            item,
            Homogeneous,
            Self,
            "Expected numerical input",
            U8,
            U16,
            U32,
            U64,
            I8,
            I16,
            I32,
            I64,
            F16,
            F32,
            F64
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn narrow_optionals() {
        let dummy: Vec<Option<TaggedReceiver>> = vec![];

        // Test compilation
        let _: Result<(
            Option<HomogeneousFixed<2>>,
            Option<TaggedReceiver>,
            Option<Homogeneous>,
        )> = dummy.narrow();
    }

    #[test]
    fn narrow_required() {
        let dummy: Vec<TaggedReceiver> = vec![];

        // Test compilation
        let _: Result<(HomogeneousFixed<2>, TaggedReceiver, Homogeneous)> = dummy.narrow();
    }
}

/// Homogeneous, numeric variadic inputs
#[derive(Debug)]
pub enum Numeric {
    F16(Receiver<ArcArrayD<f16>>),
    F32(Receiver<ArcArrayD<f32>>),
    F64(Receiver<ArcArrayD<f64>>),

    I8(Receiver<ArcArrayD<i8>>),
    I16(Receiver<ArcArrayD<i16>>),
    I32(Receiver<ArcArrayD<i32>>),
    I64(Receiver<ArcArrayD<i64>>),

    U8(Receiver<ArcArrayD<u8>>),
    U16(Receiver<ArcArrayD<u16>>),
    U32(Receiver<ArcArrayD<u32>>),
    U64(Receiver<ArcArrayD<u64>>),
}

/// Homogeneous, numeric variadic inputs
#[derive(Debug)]
pub enum Integer {
    I8(Receiver<ArcArrayD<i8>>),
    I16(Receiver<ArcArrayD<i16>>),
    I32(Receiver<ArcArrayD<i32>>),
    I64(Receiver<ArcArrayD<i64>>),

    U8(Receiver<ArcArrayD<u8>>),
    U16(Receiver<ArcArrayD<u16>>),
    U32(Receiver<ArcArrayD<u32>>),
    U64(Receiver<ArcArrayD<u64>>),
}

/// Homogeneous, numeric variadic inputs
#[derive(Debug)]
pub enum FloatingPoint {
    F16(Receiver<ArcArrayD<f16>>),
    F32(Receiver<ArcArrayD<f32>>),
    F64(Receiver<ArcArrayD<f64>>),
}

/// Homogeneous, numeric variadic inputs
#[derive(Debug)]
pub enum OnnxIndex {
    I32(Receiver<ArcArrayD<i32>>),
    I64(Receiver<ArcArrayD<i64>>),
}

/// Homogeneous, numeric variadic inputs
#[derive(Debug)]
pub enum Homogeneous {
    F16(Vec<Receiver<ArcArrayD<f16>>>),
    F32(Vec<Receiver<ArcArrayD<f32>>>),
    F64(Vec<Receiver<ArcArrayD<f64>>>),

    I8(Vec<Receiver<ArcArrayD<i8>>>),
    I16(Vec<Receiver<ArcArrayD<i16>>>),
    I32(Vec<Receiver<ArcArrayD<i32>>>),
    I64(Vec<Receiver<ArcArrayD<i64>>>),

    U8(Vec<Receiver<ArcArrayD<u8>>>),
    U16(Vec<Receiver<ArcArrayD<u16>>>),
    U32(Vec<Receiver<ArcArrayD<u32>>>),
    U64(Vec<Receiver<ArcArrayD<u64>>>),

    String(Vec<Receiver<ArcArrayD<String>>>),
    Bool(Vec<Receiver<ArcArrayD<bool>>>),
}

/// Homogeneous, numeric variadic inputs
#[derive(Debug)]
pub enum HomogeneousNumeric {
    F16(Vec<Receiver<ArcArrayD<f16>>>),
    F32(Vec<Receiver<ArcArrayD<f32>>>),
    F64(Vec<Receiver<ArcArrayD<f64>>>),

    I8(Vec<Receiver<ArcArrayD<i8>>>),
    I16(Vec<Receiver<ArcArrayD<i16>>>),
    I32(Vec<Receiver<ArcArrayD<i32>>>),
    I64(Vec<Receiver<ArcArrayD<i64>>>),

    U8(Vec<Receiver<ArcArrayD<u8>>>),
    U16(Vec<Receiver<ArcArrayD<u16>>>),
    U32(Vec<Receiver<ArcArrayD<u32>>>),
    U64(Vec<Receiver<ArcArrayD<u64>>>),
}

/// Homogeneous variadic inputs
#[derive(Debug)]
pub enum HomogeneousFixed<const N: usize> {
    F16([Receiver<ArcArrayD<f16>>; N]),
    F32([Receiver<ArcArrayD<f32>>; N]),
    F64([Receiver<ArcArrayD<f64>>; N]),

    I8([Receiver<ArcArrayD<i8>>; N]),
    I16([Receiver<ArcArrayD<i16>>; N]),
    I32([Receiver<ArcArrayD<i32>>; N]),
    I64([Receiver<ArcArrayD<i64>>; N]),

    U8([Receiver<ArcArrayD<u8>>; N]),
    U16([Receiver<ArcArrayD<u16>>; N]),
    U32([Receiver<ArcArrayD<u32>>; N]),
    U64([Receiver<ArcArrayD<u64>>; N]),

    Bool([Receiver<ArcArrayD<bool>>; N]),
    String([Receiver<ArcArrayD<String>>; N]),
}

/// Homogeneous, numeric variadic inputs
#[derive(Debug)]
pub enum HomogeneousNumericFixed<const N: usize> {
    F16([Receiver<ArcArrayD<f16>>; N]),
    F32([Receiver<ArcArrayD<f32>>; N]),
    F64([Receiver<ArcArrayD<f64>>; N]),

    I8([Receiver<ArcArrayD<i8>>; N]),
    I16([Receiver<ArcArrayD<i16>>; N]),
    I32([Receiver<ArcArrayD<i32>>; N]),
    I64([Receiver<ArcArrayD<i64>>; N]),

    U8([Receiver<ArcArrayD<u8>>; N]),
    U16([Receiver<ArcArrayD<u16>>; N]),
    U32([Receiver<ArcArrayD<u32>>; N]),
    U64([Receiver<ArcArrayD<u64>>; N]),
}

#[derive(Debug)]
pub enum HomogeneousPrimIntFixed<const N: usize> {
    I8([Receiver<ArcArrayD<i8>>; N]),
    I16([Receiver<ArcArrayD<i16>>; N]),
    I32([Receiver<ArcArrayD<i32>>; N]),
    I64([Receiver<ArcArrayD<i64>>; N]),

    U8([Receiver<ArcArrayD<u8>>; N]),
    U16([Receiver<ArcArrayD<u16>>; N]),
    U32([Receiver<ArcArrayD<u32>>; N]),
    U64([Receiver<ArcArrayD<u64>>; N]),
}

impl Homogeneous {
    fn dtype(&self) -> DataType {
        use Homogeneous::*;
        match self {
            U8(_) => DataType::U8,
            U16(_) => DataType::U16,
            U32(_) => DataType::U32,
            U64(_) => DataType::U64,
            I8(_) => DataType::I8,
            I16(_) => DataType::I16,
            I32(_) => DataType::I32,
            I64(_) => DataType::I64,
            F16(_) => DataType::F16,
            F32(_) => DataType::F32,
            F64(_) => DataType::F64,
            Bool(_) => DataType::Bool,
            String(_) => DataType::String,
        }
    }
}

impl<const N: usize> HomogeneousFixed<N> {
    fn dtype(&self) -> DataType {
        use HomogeneousFixed::*;
        match self {
            U8(_) => DataType::U8,
            U16(_) => DataType::U16,
            U32(_) => DataType::U32,
            U64(_) => DataType::U64,
            I8(_) => DataType::I8,
            I16(_) => DataType::I16,
            I32(_) => DataType::I32,
            I64(_) => DataType::I64,
            F16(_) => DataType::F16,
            F32(_) => DataType::F32,
            F64(_) => DataType::F64,
            Bool(_) => DataType::Bool,
            String(_) => DataType::String,
        }
    }
}
