use std::collections::HashMap;
use std::fs::read;

use anyhow::{anyhow, bail, Result};
use approx::relative_eq;
use onnx_protos::onnx_proto3::TensorProto;
use protobuf::Message;
use upstream_test_data::{all_node_tests, TestCase};

use ::onnxer::{get_opset, Session, Value};

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

impl TestResult {
    fn compare(&self) -> Result<String> {
        if self.expected.len() != self.observed.len() {
            bail!("Unequal number of outputs.");
        }
        for (k, v_exp) in self.expected.iter() {
            let v_obs = self
                .observed
                .get(k)
                .ok_or_else(|| anyhow!("Key {} is not in the observed output.", k))?;
            macro_rules! unequal_bail {
                ($k:expr, $exp:expr, $obs:expr) => {
                    bail!(
                        "Unequal outputs for `{}`:\n    expected: {:?}\n    observed: {:?}",
                        $k,
                        $exp,
                        $obs
                    )
                };
            }
            match (v_exp, v_obs) {
                (Value::Bool(exp), Value::Bool(obs)) => {
                    if exp != obs {
                        unequal_bail!(k, exp, obs)
                    }
                }
                (Value::I32(exp), Value::I32(obs)) => {
                    if exp != obs {
                        unequal_bail!(k, exp, obs)
                    }
                }
                (Value::I64(exp), Value::I64(obs)) => {
                    if exp != obs {
                        unequal_bail!(k, exp, obs)
                    }
                }
                (Value::String(exp), Value::String(obs)) => {
                    if exp != obs {
                        unequal_bail!(k, exp, obs)
                    }
                }
                (Value::F32(exp), Value::F32(obs)) => {
                    if exp.shape() != obs.shape() {
                        bail!("Expected shape {:?} found {:?}", exp.shape(), obs.shape());
                    }
                    for (exp_el, obs_el) in exp.iter().zip(obs.iter()) {
                        if relative_eq!(exp_el, obs_el, epsilon = f32::EPSILON) {
                            continue;
                        }
                        if exp_el.is_nan() && obs_el.is_nan() {
                            continue;
                        }
                        if exp_el.is_infinite() && obs_el.is_infinite() {
                            if exp_el.is_sign_positive() == obs_el.is_sign_positive() {
                                continue;
                            }
                        }
                        unequal_bail!(k, exp, obs)
                    }
                }
                (Value::F64(exp), Value::F64(obs)) => {
                    if exp.shape() != obs.shape() {
                        bail!("Expected shape {:?} found {:?}", exp.shape(), obs.shape());
                    }
                    for (exp_el, obs_el) in exp.iter().zip(obs.iter()) {
                        if relative_eq!(exp_el, obs_el, epsilon = f64::EPSILON) {
                            continue;
                        }
                        if exp_el.is_nan() && obs_el.is_nan() {
                            continue;
                        }
                        if exp_el.is_infinite() && obs_el.is_infinite() {
                            if exp_el.is_sign_positive() == obs_el.is_sign_positive() {
                                continue;
                            }
                        }
                        unequal_bail!(k, exp, obs)
                    }
                }
                _ => panic!(),
            }
        }
        Ok("PASS".to_string())
    }
}

#[test]
fn run_all() {
    let mut passed = 0;
    let mut failed = 0;
    println!("");
    for tc in &all_node_tests().unwrap() {
        print!("{}: ", tc.name);
        match run_test_case(tc).and_then(|tr| tr.compare()) {
            Ok(res) => {
                passed += 1;
                println!("{}", res)
            }
            Err(msg) => {
                failed += 1;
                println!("{}", msg)
            }
        }
    }

    println!(
        "PASSED: {}, FAILED: {}, PERCENTAGE: {}%",
        passed,
        failed,
        (passed as f64) / (failed as f64) * 100.
    );
}
