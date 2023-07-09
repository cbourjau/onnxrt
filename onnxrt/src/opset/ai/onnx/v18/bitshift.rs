use crate::opset::utils::*;

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

    fn compute<T>(&self, a: ArcArrayD<T>, b: ArcArrayD<T>) -> Result<ArcArrayD<T>>
    where
        T: Clone + std::ops::Shl<Output = T> + std::ops::Shr<Output = T> + Copy,
    {
        match self.direction {
            Direction::Left => zip_map_with_co_broadcast(a, b, |&a, &b| a << b),
            Direction::Right => zip_map_with_co_broadcast(a, b, |&a, &b| a >> b),
        }
    }
}

pub fn setup_bishift(ctx: OpContext) -> Result<ExecutionStep> {
    let state = BitShift::from_attrs(&ctx.attrs)?;
    let inputs = ctx.subscribe_to_inputs().narrow()?;

    use HomogeneousNumericFixed::*;
    match inputs {
        U8([a, b]) => setup((a, b), move |(a, b)| state.compute(a, b)),
        U16([a, b]) => setup((a, b), move |(a, b)| state.compute(a, b)),
        U32([a, b]) => setup((a, b), move |(a, b)| state.compute(a, b)),
        U64([a, b]) => setup((a, b), move |(a, b)| state.compute(a, b)),
        _ => bail!("BitShift: Expected unsigned input found: {:?}", inputs),
    }
}
