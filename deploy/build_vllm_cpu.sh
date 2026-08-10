#!/bin/bash
set -euo pipefail
exec > >(tee -a /home/ubuntu/vllm-cpu-build.log) 2>&1
echo "=== vLLM CPU build start $(date -Is) ==="
sudo apt-get update -y
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y gcc-13 g++-13 libnuma-dev python3-dev git
source /home/ubuntu/neoserve-venv/bin/activate
pip install --upgrade pip
pip uninstall -y vllm || true
pip install "cmake>=3.26" wheel packaging ninja "setuptools-scm>=8" numpy
rm -rf /home/ubuntu/vllm-src
git clone --depth 1 https://github.com/vllm-project/vllm.git /home/ubuntu/vllm-src
cd /home/ubuntu/vllm-src
if [[ -f requirements/cpu.txt ]]; then
  pip install -v -r requirements/cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
elif [[ -f requirements/cpu-build.txt ]]; then
  pip install -v -r requirements/cpu-build.txt --extra-index-url https://download.pytorch.org/whl/cpu
else
  pip install -v torch --index-url https://download.pytorch.org/whl/cpu
fi
export VLLM_TARGET_DEVICE=cpu
export CC=gcc-13 CXX=g++-13
python setup.py bdist_wheel
pip install dist/*.whl
python - <<'PY'
import os
os.environ["VLLM_TARGET_DEVICE"] = "cpu"
from vllm.platforms import current_platform
print("platform=", current_platform)
print("device_type=", getattr(current_platform, "device_type", None))
PY
echo "=== vLLM CPU build done $(date -Is) ==="
