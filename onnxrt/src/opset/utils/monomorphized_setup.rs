use anyhow::{anyhow, bail};
use flume::Receiver;

use crate::{Executable, TaggedReceiver, channels::unbounded, opset::utils::*, session::OpContext};

pub fn setup_unary<T1, T4>(
    inp: Receiver<ArcArrayD<T1>>,
    fun: impl Fn(ArcArrayD<T1>) -> Result<ArrayD<T4>> + Send + Sync + 'static + Copy,
) -> anyhow::Result<(Box<dyn crate::Executable>, Vec<crate::TaggedReceiver>)>
where
    T1: Sync + Send + 'static,
    T4: Sync + Send + Copy + 'static,
    TaggedReceiver: From<Receiver<ArcArrayD<T4>>>,
{
    let (res_broad, res_rec) = unbounded();
    let compute = move || {
        let a = inp.recv()?;
        let res = fun(a)?;
        res_broad.send(res.into_shared())?;
        Ok(())
    };
    Ok((Box::new(compute) as _, vec![res_rec]))
}

/// Setup function that maps two inputs to a single output with bidirectional broadcasting.
pub fn setup_binary_zipped<T1, T2, T3>(
    a: Receiver<ArcArrayD<T1>>,
    b: Receiver<ArcArrayD<T2>>,
    fun: impl Fn(&T1, &T2) -> T3 + Send + Sync + 'static + Copy,
) -> anyhow::Result<(Box<dyn crate::Executable>, Vec<crate::TaggedReceiver>)>
where
    T1: Sync + Send + Clone + 'static,
    T2: Sync + Send + Clone + 'static,
    T3: Sync + Send + Clone + 'static,
    TaggedReceiver: From<Receiver<ArcArrayD<T3>>>,
{
    setup_binary((a, b), move |a, b| {
        zip_map_with_co_broadcast(a, b, move |a, b| fun(a, b))
    })
}

/// Setup function that maps three inputs to a single output with bidirectional broadcasting.
pub fn setup_tertiary_zipped<T1, T2, T3, T4>(
    a: Receiver<ArcArrayD<T1>>,
    b: Receiver<ArcArrayD<T2>>,
    c: Receiver<ArcArrayD<T3>>,
    fun: impl Fn(&T1, &T2, &T3) -> T4 + Send + Sync + 'static + Copy,
) -> anyhow::Result<(Box<dyn crate::Executable>, Vec<crate::TaggedReceiver>)>
where
    T1: Sync + Send + Copy + 'static,
    T2: Sync + Send + Copy + 'static,
    T3: Sync + Send + Copy + 'static,
    T4: Sync + Send + Copy + 'static,
    TaggedReceiver: From<Receiver<ArcArrayD<T4>>>,
{
    setup_tertiary(a, b, c, move |a, b, c| {
        zip_map_with_co_broadcast3(a, b, c, move |a, b, c| fun(a, b, c))
    })
}

/// Setup function that maps two inputs to a single output with bidirectional broadcasting.
///
/// The provided vector must have at least one element.
pub fn setup_variadic_zipped_reduce<T1>(
    recs: Vec<Receiver<ArcArrayD<T1>>>,
    fun: impl Fn(&T1, &T1) -> T1 + Send + Sync + 'static + Copy,
) -> anyhow::Result<(Box<dyn crate::Executable>, Vec<crate::TaggedReceiver>)>
where
    T1: Sync + Send + Copy + 'static,
    TaggedReceiver: From<Receiver<ArcArrayD<T1>>>,
{
    let (res_broad, res_rec) = unbounded();
    let compute = move || {
        // TODO: Receive all values at once, broadcast `a` to it, and reuse its buffer
        let mut a = recs[0].recv()?.into_owned();
        for rec in recs[1..].iter() {
            let b = rec.recv()?;
            a = zip_map_with_co_broadcast(a.view(), b.view(), move |a, b| fun(a, b))?;
        }
        res_broad.send(a.into_shared())?;
        Ok(())
    };
    Ok((Box::new(compute) as _, vec![res_rec]))
}

fn setup_binary<T1, T2, T4>(
    inp: (Receiver<ArcArrayD<T1>>, Receiver<ArcArrayD<T2>>),
    fun: impl Fn(ArrayViewD<T1>, ArrayViewD<T2>) -> Result<ArrayD<T4>> + Send + Sync + 'static + Copy,
) -> anyhow::Result<(Box<dyn crate::Executable>, Vec<crate::TaggedReceiver>)>
where
    T1: Sync + Send + Clone + 'static,
    T2: Sync + Send + Clone + 'static,
    T4: Sync + Send + Clone + 'static,
    TaggedReceiver: From<Receiver<ArcArrayD<T4>>>,
{
    let (res_broad, res_rec) = unbounded();
    let compute = move || {
        let (a, b) = (inp.0.recv()?, inp.1.recv()?);
        let res = fun(a.view(), b.view())?;
        res_broad.send(res.into_shared())?;
        Ok(())
    };
    Ok((Box::new(compute) as _, vec![res_rec]))
}

fn setup_tertiary<T1, T2, T3, T4>(
    a: Receiver<ArcArrayD<T1>>,
    b: Receiver<ArcArrayD<T2>>,
    c: Receiver<ArcArrayD<T3>>,
    fun: impl Fn(ArrayViewD<T1>, ArrayViewD<T2>, ArrayViewD<T3>) -> Result<ArrayD<T4>>
    + Send
    + Sync
    + 'static
    + Copy,
) -> anyhow::Result<(Box<dyn crate::Executable>, Vec<crate::TaggedReceiver>)>
where
    T1: Sync + Send + Copy + 'static,
    T2: Sync + Send + Copy + 'static,
    T3: Sync + Send + Copy + 'static,
    T4: Sync + Send + Copy + 'static,
    TaggedReceiver: From<Receiver<ArcArrayD<T4>>>,
{
    let (res_broad, res_rec) = unbounded();
    let compute = move || {
        let (a, b, c) = (a.recv()?, b.recv()?, c.recv()?);
        let res = fun(a.view(), b.view(), c.view())?;
        res_broad.send(res.into_shared())?;
        Ok(())
    };
    Ok((Box::new(compute) as _, vec![res_rec]))
}

impl<F> Executable for F
where
    F: Fn() -> Result<()> + Send + Sync,
{
    fn execute(&self) -> Result<()> {
        self()
    }
}
