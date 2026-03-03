# accel-gpu

<p align="center">
  <img src="icon.png" alt="accel-gpu" width="64" height="64">
</p>

**NumPy for the browser GPU — zero shaders, zero dependencies.**

A lightweight WebGPU wrapper for data processing and math. No WGSL required. Automatic fallback to WebGL2 or CPU. Perfect for local-first AI, data dashboards, and heavy array computations.

### Why accel-gpu?

- **Shader-free API** — No WGSL or GLSL. Write NumPy-like JavaScript; kernels are built-in.
- **Zero dependencies** — ~160KB minified, lightweight and self-contained.
- **Universal fallback** — WebGPU → WebGL2 → CPU. Runs in Safari, Firefox, Node, and headless.
- **Shape inference** — Matmul and ML ops automatically infer dimensions.
- **Performance** — WebGPU delivers 2–3× speedups over WebGL for compute; ~20× faster than CPU on large matmul (Chrome, M3 MacBook).
- **Accelerated ops** — `conv2d`, `maxPool2d`, `avgPool2d`, `fft`, `ifft`, and `fftMagnitude` run on WebGPU when available.
- **Automatic scalar fusion** — chained scalar `add/sub/mul/div` are fused before execution.
- **Arrow interop** — import Apache Arrow-like vectors/columns via `fromArrow(...)` and `gpu.fromArrow(...)`.

Compared to TensorFlow.js or GPU.js, accel-gpu offers a simpler API focused on core array operations without the overhead of a full ML framework.

[![npm](https://img.shields.io/npm/v/accel-gpu)](https://www.npmjs.com/package/accel-gpu)
[![Bundlephobia](https://img.shields.io/bundlephobia/minzip/accel-gpu)](https://bundlephobia.com/package/accel-gpu)
[![Tests](https://github.com/Phantasm0009/accel-gpu/actions/workflows/test.yml/badge.svg)](https://github.com/Phantasm0009/accel-gpu/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![TypeScript](https://img.shields.io/badge/TypeScript-Ready-blue)](https://www.typescriptlang.org/)

## Install

```bash
npm install accel-gpu
```

**TypeScript:** Definitions are included; no `@types` package needed.

## Quick Start

```javascript
import { init, matmul, softmax } from "accel-gpu";

const gpu = await init();

// Create GPU-backed arrays (WebGPU, WebGL2, or CPU)
const a = gpu.array([1, 2, 3, 4]);
const b = gpu.array([5, 6, 7, 8]);

// Method chaining
await a.add(b);
const total = await a.sum();
console.log(total); // 26

// Shape inference — no need to pass M, N, K
const A = gpu.array(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]);
const B = gpu.array(new Float32Array([1, 2, 3, 4, 5, 6]), [3, 2]);
const C = await matmul(gpu, A, B);

// Softmax with shape inference
const logits = gpu.array([1, 2, 3, 4]);
const probs = await softmax(gpu, logits);
console.log(await probs.toArray());
```

## Demos

- **[Demo](https://phantasm0009.github.io/accel-gpu/example/)** — Basic usage
- **[Image Processing](https://phantasm0009.github.io/accel-gpu/example/image/)** — Brightness, contrast, invert
- **[Heatmap](https://phantasm0009.github.io/accel-gpu/example/heatmap/)** — GPU-computed 2D data visualization
- **[Neural Network](https://phantasm0009.github.io/accel-gpu/example/nn/)** — Feedforward inference (MNIST-style)
- **[N-Body](https://phantasm0009.github.io/accel-gpu/example/nbody/)** — Gravitational particle simulation
- **[Local Audio Transcriber](https://phantasm0009.github.io/accel-gpu/example/audio/)** — In-browser spectrogram visualizer + local token preview
- **[Vector Search (RAG)](https://phantasm0009.github.io/accel-gpu/example/vector-search/)** — Browser-native cosine search over large vector sets
- **[Benchmarks](https://phantasm0009.github.io/accel-gpu/benchmark/)** — WebGPU vs WebGL vs CPU performance
- **[Playground](https://phantasm0009.github.io/accel-gpu/playground/)** — Interactive code editor

Run `npm run build` first, then `npx serve .` — visit `/`, `/example/`, `/example/image/`, etc.

## API Reference

### Initialize

```js
const gpu = await init();
const gpu = await init({ forceCPU: true });   // Force CPU for testing
const gpu = await init({ forceWebGL: true }); // Force WebGL2
const gpu = await init({ forceCPU: true, worker: true }); // CPU ops in Web Worker (experimental)
const gpu = await init({ forceCPU: true, preferWasmCPU: true, wasmModule }); // WASM CPU path (experimental)
```

Runtime flags:

```js
console.log(gpu.backendType);    // 'webgpu' | 'webgl' | 'cpu'
console.log(gpu.workerEnabled);  // true when worker CPU runner is active
console.log(gpu.cpuEngine);      // 'js' | 'wasm' when CPU backend is used
```

Scoped lifecycle:

```js
await gpu.scoped(async (ctx) => {
  const tmp = ctx.array([1, 2, 3]);
  await tmp.mul(2);
}); // arrays created in scope are disposed on exit
```

Arrow interoperability:

```js
// Apache Arrow-like vector/column
const arr = gpu.fromArrow(arrowColumn, { shape: [rows, cols] });
// or function form
const arr2 = fromArrow(gpu, arrowVector);
```

### Create Arrays

```js
const arr = gpu.array([1, 2, 3]);
const arr2 = gpu.array(new Float32Array([1, 2, 3]), [3]); // with shape
const mat = gpu.array(data, [2, 3]); // 2×3 matrix
const z = gpu.zeros([2, 3]);        // all zeros
const o = gpu.ones([2, 3]);         // all ones
const r = gpu.arange(0, 10, 2);     // [0, 2, 4, 6, 8]
const l = gpu.linspace(0, 1, 100);  // 100 values from 0 to 1
const rand = gpu.random([2, 3]);    // uniform [0, 1)
const norm = gpu.randn([2, 3]);     // standard normal
```

### Math Operations (chainable)

| Method                   | Description           |
| ------------------------ | --------------------- |
| `a.add(b)` or `a.add(5)` | Element-wise add      |
| `a.sub(b)` or `a.sub(5)` | Element-wise subtract |
| `a.mul(b)` or `a.mul(2)` | Element-wise multiply |
| `a.div(b)` or `a.div(2)` | Element-wise divide   |
| `a.pow(2)`               | Element-wise power    |
| `a.sqrt()`, `a.abs()`, `a.neg()` | Unary ops     |
| `a.exp()`, `a.log()`     | Element-wise exp/log  |
| `a.sum()`, `a.max()`, `a.min()`, `a.mean()` | Reductions (pass `axis` for axis-specific) |
| `a.variance()`, `a.std()`, `a.argmax()`, `a.argmin()` | Stats |
| `a.dot(b)`               | Dot product → scalar  |
| `a.reshape(2, 3)`        | Reshape (same length) |
| `a.relu()`, `a.sigmoid()`, `a.tanh()`, `a.gelu()`, `a.leakyRelu(α)` | Activations |
| `a.clamp(min, max)`      | Clamp values          |
| `a.equal(b)`, `a.greater(b)`, `a.less(b)` | Comparison (returns 0/1 mask) |
| `a.slice(start, end)`, `a.get(i)`, `a.set(i, v)` | Slicing |
| `a.concat(b)`, `a.split(n)` | Concat/split |
| `a.flatten()`, `a.squeeze()`, `a.unsqueeze(dim)` | Shape |
| `a.broadcast(shape)`     | NumPy-style broadcast |
| `a.norm(ord?)`, `a.outer(b)` | Norm, outer product |
| `a.mse(target)`, `a.crossEntropy(target)` | Loss functions |
| `a.normalize(axis?)`     | L2 normalize along axis |
| `a.dispose()`, `a.isDisposed` | Memory management |
| `a.toArraySync()`        | Sync read (CPU only)  |

### Linear Algebra

| Function                          | Description                       |
| --------------------------------- | --------------------------------- |
| `matmul(gpu, A, B)`               | Matrix multiply (shape inference) |
| `dot(gpu, a, b)`                  | Vector dot product                |
| `transpose(gpu, a, rows?, cols?)` | Transpose matrix                  |
| `inv(gpu, a)`, `det(gpu, a)`      | Inverse, determinant              |
| `solve(gpu, A, b)`                | Solve Ax = b                      |
| `qr(gpu, a)`                      | QR decomposition                  |
| `svd(gpu, a)`                     | Singular value decomposition      |

`inv`, `qr`, and `svd` use iterative WebGPU paths when running on WebGPU backend, with CPU fallback.

### ML Primitives

| Function                                           | Description                 |
| -------------------------------------------------- | --------------------------- |
| `softmax(gpu, input, rows?, cols?)`                | Softmax over last dimension |
| `layerNorm(gpu, input, gamma, beta, rows?, cols?)` | Layer normalization         |
| `batchNorm(gpu, input, gamma, beta, rows?, cols?)` | Batch normalization         |
| `attentionScores(gpu, Q, K, seq?, dim?)`           | Q @ K^T / sqrt(dim)         |
| `maxPool2d(gpu, input, kernelSize, stride?, padding?)` | Max pooling 2D      |
| `avgPool2d(gpu, input, kernelSize, stride?, padding?)` | Avg pooling 2D      |
| `conv2d(gpu, input, kernel, stride?, padding?)`   | 2D convolution              |

### FFT & Signal

| Function | Description |
| -------- | ----------- |
| `fft(gpu, input)` | Forward FFT (power-of-2 length) |
| `ifft(gpu, input)` | Inverse FFT |
| `fftMagnitude(gpu, complex)` | Magnitude spectrum |
| `spectrogram(gpu, input, frameLength, hopLength?, window?)` | STFT + magnitude |

`fft`, `ifft`, and `fftMagnitude` use WebGPU kernels when running on WebGPU backend, with CPU/WebGL fallback.

### Training Helpers

| Function | Description |
| -------- | ----------- |
| `gradients(gpu, params, lossFn, epsilon?)` | Numerical gradients (central differences) |
| `sgdStep(params, grads, learningRate)` | SGD update: `param -= lr * grad` |

### Profiling

```js
const gpu = await init({ profiling: true });
gpu.enableProfiling(true);
// ... run ops ...
gpu.recordOp("matmul", 12.5); // manual timing
const results = gpu.getProfilingResults();
```


### Canvas Integration

```js
const img = gpu.fromImageData(imageData);
const canvas = await gpu.toCanvas(arr, width, height);
```

### Read Back

```js
const data = await arr.toArray(); // Float32Array
```

## Fallback Chain

1. **WebGPU** — Chrome 113+, Edge 113+ (best performance)
2. **WebGL2** — Safari, Firefox, older Chrome (GPU-accelerated)
3. **CPU** — Node, headless, or when no GPU available

## Troubleshooting

- `GET /.well-known/appspecific/com.chrome.devtools.json` returning `404` in local server logs is a Chrome DevTools probe and is harmless.
- `304` responses for files like `dist/index.js` and source maps are normal cache revalidation, not runtime failures.

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, architecture, and guidelines. Quick start: clone, `npm install`, `npm test`, then open a PR. We adhere to the [Contributor Covenant](CODE_OF_CONDUCT.md) code of conduct.

## License

MIT