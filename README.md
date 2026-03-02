# accel-gpu

<p align="center">
  <img src="icon.png" alt="accel-gpu" width="64" height="64">
</p>

**NumPy for the browser GPU — zero shaders, zero dependencies.**

A lightweight WebGPU wrapper for data processing and math. No WGSL required. Automatic fallback to WebGL2 or CPU. Perfect for local-first AI, data dashboards, and heavy array computations.

### Why accel-gpu?

- **Shader-free API** — No WGSL or GLSL. Write NumPy-like JavaScript; kernels are built-in.
- **Zero dependencies** — ~65KB minified, lightweight and self-contained.
- **Universal fallback** — WebGPU → WebGL2 → CPU. Runs in Safari, Firefox, Node, and headless.
- **Shape inference** — Matmul and ML ops automatically infer dimensions.
- **Performance** — WebGPU delivers 2–3× speedups over WebGL for compute; ~20× faster than CPU on large matmul (Chrome, M3 MacBook).

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
- **[Benchmarks](https://phantasm0009.github.io/accel-gpu/benchmark/)** — WebGPU vs WebGL vs CPU performance comparison
- **[Playground](https://phantasm0009.github.io/accel-gpu/playground/)** — Interactive code editor

Run `npm run build` first, then `npx serve .` — visit `/`, `/example/`, `/benchmark/`, or `/playground/`.

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


| Method                   | Description           |
| ------------------------ | --------------------- |
| `a.add(b)` or `a.add(5)` | Element-wise add      |
| `a.mul(b)` or `a.mul(2)` | Element-wise multiply |
| `a.sum()`                | Reduce sum → scalar   |
| `a.max()`                | Reduce max → scalar   |
| `a.dot(b)`               | Dot product → scalar  |
| `a.reshape(2, 3)`        | Reshape (same length) |


### Linear Algebra


| Function                          | Description                       |
| --------------------------------- | --------------------------------- |
| `matmul(gpu, A, B)`               | Matrix multiply (shape inference) |
| `matmul(gpu, A, B, M, N, K)`      | Explicit dimensions               |
| `dot(gpu, a, b)`                  | Vector dot product                |
| `transpose(gpu, a, rows?, cols?)` | Transpose matrix                  |


### ML Primitives


| Function                                           | Description                 |
| -------------------------------------------------- | --------------------------- |
| `softmax(gpu, input, rows?, cols?)`                | Softmax over last dimension |
| `layerNorm(gpu, input, gamma, beta, rows?, cols?)` | Layer normalization         |
| `attentionScores(gpu, Q, K, seq?, dim?)`           | Q @ K^T / sqrt(dim)         |


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

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, architecture, and guidelines. Quick start: clone, `npm install`, `npm test`, then open a PR. We adhere to the [Contributor Covenant](CODE_OF_CONDUCT.md) code of conduct.

## License

MIT