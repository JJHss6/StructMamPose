# Selective-scan CUDA extension

This directory contains the CUDA sources used by StructMamPose. The S and B
training models use `selective_scan_cuda_core`; the bundled extension also builds
`selective_scan_cuda_oflex` for the retained scan wrapper.

Install CUDA-enabled PyTorch and a matching CUDA toolkit with `nvcc`, then run
from the repository root:

```bash
python -m pip install -r requirements.txt
(cd kernels/selective_scan && python -m pip install --no-build-isolation .)
python -c "import selective_scan_cuda_core; import selective_scan_cuda_oflex"
```

The build selects the compute capability of the visible NVIDIA GPU. Build on the
target GPU system, with CUDA available. The original CUDA numerical kernels and
compiler optimization flags are retained. Source copyright and attribution
notices are kept in the corresponding files.
