use std::fs::read_dir;
use std::path::{Path, PathBuf};

use anyhow::Result;

fn node_tests_folder() -> PathBuf {
    dbg!(env!("CARGO_MANIFEST_DIR"));
    concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../upstream_test_data/onnx/onnx/backend/test/data/node"
    )
    .into()
}

pub struct TestCase {
    pub name: String,
    pub model: PathBuf,
    pub inputs: Vec<PathBuf>,
    pub expected_outputs: Vec<PathBuf>,
}

pub fn all_node_tests() -> Result<Vec<TestCase>> {
    let mut test_cases = vec![];
    for dir in read_dir(node_tests_folder()).unwrap() {
        let dir = dir.unwrap();
        test_cases.push(TestCase::from_path(dir.path().as_path())?);
    }
    test_cases.sort_by_key(|item| item.name.clone());
    Ok(test_cases)
}

pub fn get_node_test(name: &str) -> Result<TestCase> {
    TestCase::from_path(node_tests_folder().join(name).as_path())
}

fn get_paths_protobuf_files(folder: &Path, name_predicate: &str) -> Result<Vec<PathBuf>> {
    let mut path_data_pb = vec![];
    for dir in read_dir(folder.join("test_data_set_0"))? {
        let data_path = dir?.path();
        if data_path
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .contains(name_predicate)
        {
            path_data_pb.push(data_path)
        }
    }
    Ok(path_data_pb)
}

impl TestCase {
    fn from_path(p: &Path) -> Result<Self> {
        let name = p.file_name().unwrap().to_str().unwrap().to_string();
        Ok(Self {
            name,
            model: p.join("model.onnx"),
            inputs: get_paths_protobuf_files(p, "input")?,
            expected_outputs: get_paths_protobuf_files(p, "output")?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all() {
        let tests = all_node_tests().unwrap();
        assert!(tests.len() > 0);
        for test in &tests {
            assert!(test.model.exists());
            assert!(test.expected_outputs.len() > 0);
        }
    }

    #[test]
    fn test_get_by_name() {
        get_node_test("test_abs").unwrap();
    }
}
