# accel-gpu

**NumPy for the browser GPU — zero shaders, zero dependencies.**

A lightweight WebGPU wrapper for data processing and math. No WGSL required. Automatic WebGL2 and CPU fallback. Perfect for local-first AI, data dashboards, and heavy array computations.

[![npm](https://img.shields.io/npm/v/accel-gpu)](https://www.npmjs.com/package/accel-gpu)

## Install

```bash
npm install accel-gpu
```

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
- **[Benchmarks](https://phantasm0009.github.io/accel-gpu/benchmark/)** — WebGPU vs WebGL vs CPU
- **[Playground](https://phantasm0009.github.io/accel-gpu/playground/)** — Interactive code editor

Run locally: `npx serve .` then open `/`, `/example/`, `/benchmark/`, `/playground/`.

## API Reference

### Initialize

```js
const gpu = await init();
const gpu = await init({ forceCPU: true });   // Force CPU for testing
const gpu = await init({ forceWebGL: true }); // Force WebGL2
```

### Create Arrays

```js
const arr = gpu.array([1, 2, 3]);
const arr2 = gpu.array(new Float32Array([1, 2, 3]), [3]); // with shape
const mat = gpu.array(data, [2, 3]); // 2×3 matrix
```

### Math Operations (chainable)

| Method | Description |
|--------|-------------|
| `a.add(b)` or `a.add(5)` | Element-wise add |
| `a.mul(b)` or `a.mul(2)` | Element-wise multiply |
| `a.sum()` | Reduce sum → scalar |
| `a.max()` | Reduce max → scalar |
| `a.dot(b)` | Dot product → scalar |
| `a.reshape(2, 3)` | Reshape (same length) |

### Linear Algebra

| Function | Description |
|----------|-------------|
| `matmul(gpu, A, B)` | Matrix multiply (shape inference) |
| `matmul(gpu, A, B, M, N, K)` | Explicit dimensions |
| `dot(gpu, a, b)` | Vector dot product |
| `transpose(gpu, a, rows?, cols?)` | Transpose matrix |

### ML Primitives

| Function | Description |
|----------|-------------|
| `softmax(gpu, input, rows?, cols?)` | Softmax over last dimension |
| `layerNorm(gpu, input, gamma, beta, rows?, cols?)` | Layer normalization |
| `attentionScores(gpu, Q, K, seq?, dim?)` | Q @ K^T / sqrt(dim) |

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

## License

MIT
