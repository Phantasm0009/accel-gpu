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
- `inv()`, `det()`, `solve(b)`, `qr()`, `svd()` — CPU implementation

### ML
- `maxPool2d`, `avgPool2d`, `conv2d(kernel, stride?, padding?)`
- `batchNorm`, `normalize(axis?)`

### FFT & Signal
- `fft()`, `ifft()`, `fftMagnitude()`, `spectrogram()` — complex support, butterfly FFT

### DX
- `enableProfiling()`, `recordOp()`, `getProfilingResults()`

---

## Planned

### Future
- GPU-accelerated FFT, conv2d, pooling
- GPU matrix ops (inv, qr, svd)
