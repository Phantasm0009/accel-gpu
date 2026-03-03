# accel-gpu Roadmap

## Implemented (v0.3.0+)

### Basic Math
- `add`, `sub`, `mul`, `div` — element-wise (array or scalar)
- `pow(exponent)` — element-wise power
- `sqrt`, `abs`, `neg`, `exp`, `log` — unary ops

### Reductions
- `sum`, `max`, `min`, `mean`
- `variance()`, `std()` — variance and standard deviation
- `argmax()`, `argmin()` — return index of max/min

### Activations
- `relu`, `sigmoid`, `tanh`
- `gelu()`, `leakyRelu(alpha)`

### Comparison
- `equal(other)`, `greater(other)`, `less(other)` — element-wise, returns 0/1 mask
- `clamp(min, max)`

### Slicing & Indexing
- `slice(start, end)` — returns new GPUArray
- `get(index)`, `set(index, value)`
- `concat(other)`, `split(numSections)`

### Shape
- `flatten()`, `squeeze()`, `unsqueeze(dim)`

### Utility
- `gpu.zeros(shape)`, `gpu.ones(shape)`, `gpu.full(shape, value)`
- `gpu.arange(start, stop, step?)`, `gpu.linspace(start, stop, num)`
- `gpu.random(shape)`, `gpu.randn(shape)` — uniform and normal random

### Other
- `clone()` — deep copy
- `reshape`, `dot`, `matmul`, `transpose`
- `softmax`, `layerNorm`, `attentionScores`
- `norm(ord?)`, `outer(other)` — L1/L2 norm, outer product
- `mse(target)`, `crossEntropy(target)` — loss functions

### Memory
- `dispose()`, `isDisposed`
- `toArraySync()` — CPU backend only

### Shape
- `broadcast(targetShape)` — replicate along dims of size 1

### Axis-specific Reductions
- `sum(axis?)`, `mean(axis?)`, `max(axis?)` — reduce along axis

### Matrix Ops
- `inv()`, `det()`, `solve(b)`, `qr()`, `svd()` — WebGPU iterative paths for `inv/qr/svd` + CPU fallback

### ML
- `maxPool2d`, `avgPool2d`, `conv2d(kernel, stride?, padding?)` — WebGPU kernels + CPU/WebGL fallback
- `batchNorm`, `normalize(axis?)`

### FFT & Signal
- `fft()`, `ifft()`, `fftMagnitude()`, `spectrogram()` — WebGPU kernels for FFT/IFFT/magnitude + CPU fallback

### DX
- `enableProfiling()`, `recordOp()`, `getProfilingResults()`

### Backend & Performance
- WebAssembly CPU backend path via `init({ forceCPU: true, preferWasmCPU: true, wasmModule })` *(experimental)*
- Web Worker CPU execution via `init({ forceCPU: true, worker: true })` *(experimental)*

### Optimization
- Automatic scalar-chain fusion for `add/sub/mul/div` (affine fusion before materialization)

### Training
- Numerical gradient computation utilities for training: `gradients(...)`
- SGD update utility: `sgdStep(...)`

### Memory Management
- `FinalizationRegistry` best-effort cleanup for leaked arrays
- Scoped API: `gpu.scoped(fn)` deterministic disposal on scope exit

---

## Planned

- No active roadmap items at the moment.
