#!/bin/bash
#
# Installer for the INR-Interpolation environment (DW0122).
#
# Creates the `inr_interp` conda environment from environment.yml, then compiles
# the two source-only dependencies (pyseistr and tiny-cuda-nn) that cannot be
# resolved by `conda env create` alone.
#
# Run: ./install_env.sh
#
set -eo pipefail

ENV_NAME=inr_interp

echo "Creating '$ENV_NAME' environment from environment.yml ..."

# 1. Base environment: python, CUDA 11.8 toolkit (nvcc), torch, numpy, matplotlib, tqdm, ninja
conda env create -f environment.yml

# Activate it (handle common conda install locations)
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null \
  || source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
  || source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
echo "Activated environment: $(which python)"

# 2. pyseistr (used by the 3D script) — build with --no-build-isolation so its
#    setup.py can see the NumPy already installed in the environment.
echo 'Installing pyseistr ...'
pip install --no-build-isolation pyseistr

# 3. tiny-cuda-nn — compiled from source against the environment's nvcc 11.8 and
#    torch (cu118). TCNN_CUDA_ARCHITECTURES targets RTX 8000 (75), A100 (80),
#    and Ada (86/89). --no-build-isolation is required because tiny-cuda-nn's
#    setup.py imports pkg_resources, removed in setuptools >= 81.
echo 'Building tiny-cuda-nn (this takes ~10-15 minutes) ...'
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export TCNN_CUDA_ARCHITECTURES="75;80;86;89"
export MAX_JOBS=4
pip install --no-build-isolation \
  "git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"

# 4. Verify torch + CUDA + tiny-cuda-nn
echo 'Checking the installation ...'
python -c 'import torch, tinycudann; \
print("torch", torch.__version__, "| CUDA", torch.version.cuda); \
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only"); \
print("tiny-cuda-nn imported OK")'

echo "Done! Activate the environment with: conda activate $ENV_NAME"
