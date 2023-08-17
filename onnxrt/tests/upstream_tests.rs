use std::collections::HashMap;
use std::fs::read;

use anyhow::{anyhow, bail, Result};
use approx::{abs_diff_eq, relative_eq, ulps_eq, AbsDiffEq, RelativeEq, UlpsEq};
use num::Float;
use num_traits::AsPrimitive;
use onnx_protos::onnx_proto3::TensorProto;
use protobuf::Message;
use upstream_test_data::{all_node_tests, get_node_test, TestCase};

use ::onnxrt::{get_opset, ArcArrayD, Session, Value};

struct TestResult {
    expected: HashMap<String, Value>,
    observed: HashMap<String, Value>,
}

fn run_test_case(tc: &TestCase) -> Result<TestResult> {
    let session = Session::from_file(&tc.model, &get_opset())?;
    let inputs = tc
        .inputs
        .iter()
        .map(|p| TensorProto::parse_from_bytes(read(p).unwrap().as_slice()).unwrap())
        .map(|tp| (tp.name.clone(), (&tp).try_into().unwrap()))
        .collect();
    let expected: HashMap<_, _> = tc
        .expected_outputs
        .iter()
        .map(|p| TensorProto::parse_from_bytes(read(p).unwrap().as_slice()).unwrap())
        .map(|tp| (tp.name.clone(), (&tp).try_into().unwrap()))
        .collect();
    let observed = session.run(&inputs)?;
    Ok(TestResult { expected, observed })
}

macro_rules! unequal_bail {
    ($k:expr, $exp:expr, $obs:expr) => {
        bail!(
            "Unequal outputs for `{}`:\n    expected: {:?}\n    observed: {:?}",
            $k,
            $exp,
            $obs,
        )
    };
}

macro_rules! unequal_bail_numeric {
    ($k:expr, $exp:expr, $obs:expr) => {
        if $exp.shape() != $obs.shape() {
            bail!(
                "Unequal shapes: exp: {:?}; obs: {:?}",
                $exp.shape(),
                $obs.shape()
            );
        } else {
            bail!(
                "Unequal outputs for `{}`:\n    expected: {:?}\n    observed: {:?}\n diff: {:?}",
                $k,
                $exp,
                $obs,
                $exp.to_owned() - $obs,
            )
        }
    };
}

fn check_float_almost_equal<T>(exp: &ArcArrayD<T>, obs: &ArcArrayD<T>) -> Result<()>
where
    T: Float + RelativeEq + UlpsEq + Clone + std::fmt::Debug + 'static + Copy,
    f64: AsPrimitive<<T as AbsDiffEq>::Epsilon>,
    <T as AbsDiffEq>::Epsilon: Clone + Copy,
{
    if exp.shape() != obs.shape() {
        bail!("Expected shape {:?} found {:?}", exp.shape(), obs.shape());
    }

    if abs_diff_eq!(exp, obs) {
        return Ok(());
    }

    if ulps_eq!(exp, obs, max_ulps = 5) {
        return Ok(());
    }
    for (exp_el, obs_el) in exp.iter().zip(obs.iter()) {
        let rel = 1e-5f64.as_();
        if relative_eq!(exp_el, obs_el, epsilon = rel) {
            continue;
        }
        if exp_el.is_nan() && obs_el.is_nan() {
            continue;
        }
        if exp_el.is_infinite()
            && obs_el.is_infinite()
            && exp_el.is_sign_positive() == obs_el.is_sign_positive()
        {
            continue;
        }
        bail!("Tensors are not equal")
    }
    Ok(())
}

impl TestResult {
    fn compare(&self) -> Result<String> {
        if self.expected.len() != self.observed.len() {
            bail!("Unequal number of outputs.");
        }

        let mut keys: Vec<_> = self.expected.keys().collect();
        keys.sort();
        for k in keys.iter() {
            let v_exp = self.expected.get(k.as_str()).unwrap();
            let v_obs = self
                .observed
                .get(k.as_str())
                .ok_or_else(|| anyhow!("Key {} is not in the observed output.", k))?;
            match (v_exp, v_obs) {
                (Value::Bool(exp), Value::Bool(obs)) => {
                    if exp != obs {
                        unequal_bail!(k, exp, obs)
                    }
                }
                (Value::I8(exp), Value::I8(obs)) => {
                    if exp != obs {
                        unequal_bail_numeric!(k, exp, obs)
                    }
                }
                (Value::I16(exp), Value::I16(obs)) => {
                    if exp != obs {
                        unequal_bail_numeric!(k, exp, obs)
                    }
                }
                (Value::I32(exp), Value::I32(obs)) => {
                    if exp != obs {
                        unequal_bail_numeric!(k, exp, obs)
                    }
                }
                (Value::I64(exp), Value::I64(obs)) => {
                    if exp != obs {
                        unequal_bail_numeric!(k, exp, obs)
                    }
                }
                (Value::U8(exp), Value::U8(obs)) => {
                    if exp != obs {
                        unequal_bail_numeric!(k, exp, obs)
                    }
                }
                (Value::U16(exp), Value::U16(obs)) => {
                    if exp != obs {
                        unequal_bail_numeric!(k, exp, obs)
                    }
                }
                (Value::U32(exp), Value::U32(obs)) => {
                    if exp != obs {
                        unequal_bail_numeric!(k, exp, obs)
                    }
                }
                (Value::U64(exp), Value::U64(obs)) => {
                    if exp != obs {
                        unequal_bail_numeric!(k, exp, obs)
                    }
                }
                (Value::String(exp), Value::String(obs)) => {
                    if exp != obs {
                        unequal_bail!(k, exp, obs)
                    }
                }
                (Value::F32(exp), Value::F32(obs)) => {
                    if check_float_almost_equal(exp, obs).is_err() {
                        unequal_bail_numeric!(k, exp, obs)
                    }
                }
                (Value::F64(exp), Value::F64(obs)) => {
                    if check_float_almost_equal(exp, obs).is_err() {
                        unequal_bail_numeric!(k, exp, obs)
                    }
                }
                _ => bail!(
                    "Types don't match:\n expected: {:?}\n observed: {:?}",
                    v_exp,
                    v_obs
                ),
            }
        }
        Ok("PASS".to_string())
    }
}

#[test]
fn run_all() {
    let mut passed = 0;
    let mut failed = 0;
    println!();
    for tc in &all_node_tests().unwrap() {
        match run_test_case(tc).and_then(|tr| tr.compare()) {
            Ok(_res) => {
                passed += 1;
                // println!("{}: {}", tc.name, res);
            }
            Err(msg) => {
                failed += 1;
                println!("{}: {}", tc.name, msg);
            }
        }
    }

    println!(
        "PASSED: {}, FAILED: {}, PERCENTAGE: {}%",
        passed,
        failed,
        (passed as f64) / ((passed + failed) as f64) * 100.
    );
}

#[test]
fn run_one_test() {
    let tc = get_node_test("test_if").unwrap();
    run_test_case(&tc).and_then(|tr| tr.compare()).unwrap();
}
