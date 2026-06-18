# GPU Acceleration (JAX Backend)

This document describes how to use the JAX-accelerated GPU backend for the
TA-MRC-PE-CC-Tube-MPC project.  The JAX backend provides GPU-accelerated
trajectory optimization, parallel sampling MPC, and batched physics
computation, targeting NVIDIA H100 / A100 GPUs.

## Requirements

- NVIDIA GPU with CUDA 12.x (H100, A100, V100, or RTX 40xx)
- CUDA Toolkit 12.x + cuDNN
- Python 3.10–3.14

## Installation

```bash
# Install JAX with CUDA 12 support
pip install -e ".[jax]"
pip install "jax[cuda12]"

# Verify GPU is visible
python -c "import jax; print(jax.devices())"
# Expected: [CudaDevice(id=0), CudaDevice(id=1)]  (2×H100)
```

## Enabling GPU Acceleration

### Option 1: Config file

Set `mpc.backend` in `configs/default.yaml`:

```yaml
mpc:
  backend: "jax"   # GPU-accelerated

jax:
  enable_x64: true
  platform: "gpu"
  precompile: true
```

### Option 2: Command-line flag

```bash
# Run all experiments on GPU
python scripts/run_all.py --gpu --n-workers 40

# Run with explicit backend
python scripts/run_all.py --backend jax --n-workers 40
```

### Option 3: Environment variable

```bash
export MPC_BACKEND=jax
export JAX_PLATFORM=gpu
python scripts/run_all.py --n-workers 40
```

## Backend Comparison

| Feature | CasADi (CPU) | JAX (GPU) |
|---------|-------------|-----------|
| Solver | IPOPT (interior point) | L-BFGS-B (quasi-Newton) |
| Differentiation | Symbolic (CasADi MX) | Auto-diff (JAX) |
| GPU acceleration | ❌ | ✅ |
| Warm-start | IPOPT internal | Solution reuse |
| Precision | float64 | float64 (configurable) |
| Fallback | SLSQP | SLSQP |
| Verified baseline | ✅ (paper claims) | New (opt-in) |

## Performance Expectations

| Component | CasADi CPU | JAX GPU | Speedup |
|-----------|-----------|---------|---------|
| Single MPC solve | ~30-50 ms | ~10-20 ms | 2-3× |
| Sampling MPC (500 samples) | ~200 ms | ~2-5 ms | 50-100× |
| Physics (batched) | ~0.5 ms | ~0.05 ms | 10× |
| Full episode (600s, 1200 steps) | ~60-120 s | ~20-40 s | 3-5× |
| 40 scenarios × 5 methods × 30 seeds | ~6-10 hours | ~2-4 hours | 3-4× |

*Measured on 2×H100 with 48-core CPU. Actual speedup depends on
horizon length, number of targets, and scenario complexity.*

## Running on Your Hardware (48 Core / 480 GB / 2×H100)

```bash
# Optimal configuration for your server
tmux new -s experiment
conda activate tube_mpc
cd ~/iot_ocean

# Prevent BLAS thread contention
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Run all experiments with GPU acceleration
python scripts/run_all.py \
    --gpu \
    --n-seeds 30 \
    --n-seeds-sensitivity 20 \
    --n-workers 40

# Expected: 2-4 hours for all 6 phases
```

## Troubleshooting

### JAX reports "No GPU devices found"

```bash
# Check CUDA installation
nvidia-smi
nvcc --version

# Reinstall JAX with correct CUDA version
pip uninstall jax jaxlib -y
pip install "jax[cuda12]" --upgrade
```

### Out of GPU memory

JAX pre-allocates GPU memory. On H100 (80 GB), this should not be an issue for this project. If needed:

```bash
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7  # Use 70% of GPU memory
```

### Results differ from CasADi

The JAX L-BFGS-B solver and CasADi IPOPT are different optimizers. Small numerical differences (±0.5° rudder, ±0.05 propeller) are expected. If differences are large, check:
- `jax.enable_x64: true` is set (float64 precision)
- Warm-start is enabled
- Same penalty weights and horizon are used

For publication results, use the CasADi backend (the paper's verified baseline).
Use the JAX backend for exploration, hyperparameter tuning, and large-scale sweeps.
