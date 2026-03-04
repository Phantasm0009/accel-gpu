# accel-gpu Roadmap

## Implemented (through v1.0.0)

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
- Zero-copy Apache Arrow-like data import via `fromArrow(...)` and `gpu.fromArrow(...)`
- Raw binary ingestion via `fromBuffer(...)` and `gpu.fromBuffer(...)` for `ArrayBuffer`/`SharedArrayBuffer`

### Optimization
- Automatic scalar-chain fusion for `add/sub/mul/div` (affine fusion before materialization)

### Training
- Numerical gradient computation utilities for training: `gradients(...)`
- SGD update utility: `sgdStep(...)`

### Demos & Adoption
- Local audio transcriber/visualizer demo (`example/audio`) using FFT/spectrogram
- Browser-native vector search demo (`example/vector-search`) for RAG-style cosine similarity
- Interactive playground embedded directly in landing docs

### Memory Management
- `FinalizationRegistry` best-effort cleanup for leaked arrays
- Scoped API: `gpu.scoped(fn)` deterministic disposal on scope exit
- Tidy API: `gpu.tidy(fn)` alias for scoped memory cleanup

### Packaging & Bundle Optimization
- `sideEffects: false` for bundler tree-shaking
- Subpath exports for targeted imports: `accel-gpu/math`, `accel-gpu/linalg`, `accel-gpu/ml`, `accel-gpu/signal`, `accel-gpu/data`

### Quality & CI
- Cross-browser Playwright checks on Chromium, Firefox, and WebKit

### Documentation
- Dedicated VitePress docs site with quick start and API pages
- Embedded live playground and example iframes directly in docs pages

---

## Post-1.0 Hardening (Implemented)

### 1) Backend Consistency Test Matrix
- Added cross-backend numeric consistency tests for `matmul`, `inv`, `fft`, and `conv2d`
- Added tolerance gates with default epsilon and op-specific overrides
- Added drift reporting (max absolute / relative error per op) in browser consistency test output

### 2) Memory Safety Stress Validation
- Added long-run stress loops for `gpu.tidy(...)` and `gpu.scoped(...)` repeated allocations
- Added nested-scope disposal and survivor assertions
- Added backend tolerance/debug guide notes including memory management guidance

### 3) Error Message Standardization
- Standardized shape/rank/power-of-two/even-length validation messages via shared helpers
- Audited `matmul`, `solve`, `conv2d`, `fft`/`ifft`/`spectrogram`, and transpose validation paths
- Added canonical error assertion tests covering core validation paths

### 4) Docs & Demo UX Consistency
- Kept docs-first themed navigation as canonical entrypoint (`/`)
- Added backend selection/performance/debug guidance in docs
- Added numeric tolerance guidance for cross-backend comparisons

### 5) Packaging/Bundle Verification
- Added bundle-size smoke checks for `math`, `linalg`, `ml`, `signal`, `data`, and main entrypoints
- Wired bundle verification into CI to catch size regressions

## Status Notes

- Safety layer (`tidy`/`scoped` + `FinalizationRegistry`): implemented
- Subpath exports and tree-shaking setup: implemented
- Interactive docs site (VitePress + live embeds): implemented
- Cross-backend consistency runner with epsilon metrics and drift reporting: implemented
- Error-message standardization audit and validation tests: implemented
