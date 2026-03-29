# accel-gpu

<p align="center">
  <img src="icon.png" alt="accel-gpu" width="64" height="64">
</p>

**NumPy for the browser GPU — zero shaders, zero dependencies. test**

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

## Documentation

- **Docs site (VitePress):** https://phantasm0009.github.io/accel-gpu/
- **Quick Start:** https://phantasm0009.github.io/accel-gpu/guide/quickstart
- **API Reference:** https://phantasm0009.github.io/accel-gpu/api

The full API reference, shape expectations, and runnable embedded playground/examples have moved to the docs site.

### Tree-shakable imports

```js
import { matmul, transpose } from "accel-gpu/linalg";
import { softmax } from "accel-gpu/ml";
import { fft } from "accel-gpu/signal";
import { fromArrow, fromBuffer } from "accel-gpu/data";
```

## Fallback Chain

1. **WebGPU** — Chrome 113+, Edge 113+ (best performance)
2. **WebGL2** — Safari, Firefox, older Chrome (GPU-accelerated)
3. **CPU** — Node, headless, or when no GPU available

## Troubleshooting

- `GET /.well-known/appspecific/com.chrome.devtools.json` returning `404` in local server logs is a Chrome DevTools probe and is harmless.
- `304` responses for files like `dist/index.js` and source maps are normal cache revalidation, not runtime failures.

## Cross-Browser Validation

- Playwright browser tests run across Chromium, Firefox, and WebKit in CI to validate fallback behavior.

## Backend Tolerance & Debugging

- Results are validated across backend selections using numeric tolerances (typically `1e-4` to `2e-4`).
- For debugging or deterministic comparisons, use `init({ forceCPU: true })` and compare against `init()` output.
- Full guidance: https://phantasm0009.github.io/accel-gpu/guide/backend-tolerance

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, architecture, and guidelines. Quick start: clone, `npm install`, `npm test`, then open a PR. We adhere to the [Contributor Covenant](CODE_OF_CONDUCT.md) code of conduct.

## License

MIT
