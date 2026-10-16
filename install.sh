#!/usr/bin/env bash
set -euo pipefail

# Run inside a Python 3.11 or 3.12 virtual/conda environment.
# CPU is the default and does not require a GPU or CUDA installation.
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
python -c 'import sys; assert (3, 11) <= sys.version_info[:2] < (3, 13), "Use Python 3.11 or 3.12"'
backend="${1:-cpu}"
case "$backend" in
    cpu|cu128) ;;
    *) echo "Usage: bash install.sh [cpu|cu128]" >&2; exit 2 ;;
esac
python -m pip install 'torch==2.8.0' --index-url "https://download.pytorch.org/whl/${backend}"
python -m pip install --only-binary=:all: \
    'torch-scatter==2.1.2' 'torch-sparse==0.6.18' 'torch-cluster==1.6.3' \
    -f "https://data.pyg.org/whl/torch-2.8.0+${backend}.html"
python -m pip install .
echo 'Installed SPARK. Run: spark demo'
