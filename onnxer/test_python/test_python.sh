#! /bin/bash

set -eux
ENV_DIR="$(dirname $0)/.venv"
# unset CONDA_PREFIX
# python -m venv ${ENV_DIR}
echo ${ENV_DIR}
source ${ENV_DIR}/bin/activate

# pip install -r requirements.txt

# # Install onnxruntime_rs into the current venv
# pushd ..
# maturin develop -r
# popd

python -m pytest test.py -k pipeline -s
