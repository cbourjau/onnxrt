#! /bin/bash

ENV_DIR="$(dirname $0)/.venv"
unset CONDA_PREFIX
python -m venv ${ENV_DIR}

source ${ENV_DIR}/bin/activate

pip install -r requirements.txt

# Install onnxruntime_rs into the current venv
cd .. && maturin develop -r && cd test_python

pytest test.py -k pipeline -s
