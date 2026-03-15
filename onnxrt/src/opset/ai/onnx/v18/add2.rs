// use crate::opset::utils::*;
use flume::{Receiver, SendError, Sender, unbounded};
use num::{Float, Integer};

use crate::{ArcArrayD, opset::utils::HomogeneousFixed, session::OpContext};

trait Executable {
    fn execute(&self) -> Result<(), ()>;
}

struct Broadcaster<T> {
    sender: Sender<T>,
}

impl<T: Clone> Broadcaster<T> {
    fn send(&self, value: T) -> Result<(), SendError<T>> {
        for _ in 0..self.sender.receiver_count() {
            self.sender.send(value.clone())?
        }
        Ok(())
    }
}

// NOTE: These must not exist when we execute
enum TaggedReceiver {
    F32(Receiver<ArcArrayD<f32>>),
    I32(Receiver<ArcArrayD<i32>>),
}

struct BinaryWrapper<T, S> {
    a: Receiver<ArcArrayD<T>>,
    b: Receiver<ArcArrayD<T>>,
    c: Broadcaster<ArcArrayD<T>>,
    state: S,
}

trait BinaryKernel<T> {
    fn compute(&self, a: ArcArrayD<T>, b: ArcArrayD<T>) -> ArcArrayD<T>;
}

impl<T, S> Executable for BinaryWrapper<T, S>
where
    S: BinaryKernel<T>,
{
    /// Consume inputs and produce outputs
    fn execute(&self) -> Result<(), ()> {
        let a = self.a.recv().unwrap();
        let b = self.b.recv().unwrap();
        let res = self.state.compute(a, b);
        self.c.send(res).unwrap();
        Ok(())
    }
}

// Assume that we want to specialize for some data types
struct AddInt;
struct AddFloat;

impl<T> BinaryKernel<T> for AddInt
where
    T: Integer + Clone,
{
    fn compute(&self, a: ArcArrayD<T>, b: ArcArrayD<T>) -> ArcArrayD<T> {
        a + &b
    }
}
impl<T> BinaryKernel<T> for AddFloat
where
    T: Float,
{
    fn compute(&self, a: ArcArrayD<T>, b: ArcArrayD<T>) -> ArcArrayD<T> {
        a + &b
    }
}

macro_rules! match_binary {
    ($data:expr => {$([$($variant:ident),*] => $state:expr),* $(,)?}) => {
        match $data {
            $(
                $(
                    super::HomogeneousFixed::$variant([a, b]) => {
                        let (c, c_rec) = unbounded();
                        let c = Broadcaster{sender: c};
                        let state = $state;
                        (Box::new(BinaryWrapper { a, b, c, state }), vec![TaggedReceiver::$variant(c_rec)])
                    }
                ),*
            ),*
                _ => panic!()
        }
    };
}

fn mock_narrow(_recs: Vec<crate::TaggedReceiver>) -> HomogeneousFixed<2> {
    todo!()
}

fn setup_add(ctx: OpContext) -> (Box<dyn Executable>, Vec<TaggedReceiver>) {
    let data = mock_narrow(ctx.subscribe_to_inputs());

    match_binary!(
        data => {
            [I32] => AddInt,
            [F32] => AddFloat
        }
    )
}
