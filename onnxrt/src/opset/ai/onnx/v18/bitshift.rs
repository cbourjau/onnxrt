use crate::{Attributes, Executable, TaggedReceiver, opset::utils::*, session::OpContext};
use anyhow::bail;

struct BitShift {
    direction: Direction,
}

enum Direction {
    Left,
    Right,
}

impl BitShift {
    fn from_attrs(attrs: &Attributes) -> Result<Self> {
        let direction: String = attrs.get("direction")?.try_into()?;
        let direction = match direction.as_str() {
            "RIGHT" => Direction::Right,
            "LEFT" => Direction::Left,
            dir => bail!("BitShift: Invalid direction '{}'", dir),
        };

        Ok(Self { direction })
    }
}

impl<T> BinaryKernel<T> for BitShift
where
    T: std::ops::Shl<Output = T> + std::ops::Shr<Output = T> + Copy,
{
    fn compute(&self, a: ArcArrayD<T>, b: ArcArrayD<T>) -> Result<ArcArrayD<T>> {
        Ok(match self.direction {
            Direction::Left => zip_map_with_co_broadcast(a.view(), b.view(), |&a, &b| a << b),
            Direction::Right => zip_map_with_co_broadcast(a.view(), b.view(), |&a, &b| a >> b),
        }?
        .into())
    }
}

pub fn setup_bishift(
    ctx: OpContext,
) -> Result<(Box<dyn Executable + Send + Sync>, Vec<TaggedReceiver>)> {
    use HomogeneousFixed::*;
    let state = BitShift::from_attrs(&ctx.attrs)?;
    let inputs = ctx.subscribe_to_inputs().narrow()?;

    match inputs {
        U8([a, b]) => {
            let (c, c_rec) = crate::channels::unbounded();
            Ok((Box::new(BinaryWrapper { a, b, c, state }), vec![c_rec]))
        }
        U16([a, b]) => {
            let (c, c_rec) = crate::channels::unbounded();
            Ok((Box::new(BinaryWrapper { a, b, c, state }), vec![c_rec]))
        }
        U32([a, b]) => {
            let (c, c_rec) = crate::channels::unbounded();
            Ok((Box::new(BinaryWrapper { a, b, c, state }), vec![c_rec]))
        }
        U64([a, b]) => {
            let (c, c_rec) = crate::channels::unbounded();
            Ok((Box::new(BinaryWrapper { a, b, c, state }), vec![c_rec]))
        }
        _ => panic!(),
    }
}
