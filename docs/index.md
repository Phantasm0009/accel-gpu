---
layout: home

hero:
  name: "accel-gpu"
  text: "NumPy for the browser GPU"
  tagline: "WebGPU-first math with automatic WebGL2/CPU fallback."
  actions:
    - theme: brand
      text: Quick Start
      link: /guide/quickstart
    - theme: alt
      text: API Reference
      link: /api
    - theme: alt
      text: Open Playground
      link: https://phantasm0009.github.io/accel-gpu/playground/

features:
  - title: Zero-shader API
    details: NumPy-style operations without writing WGSL.
  - title: Smart fallback chain
    details: WebGPU → WebGL2 → CPU for broad browser compatibility.
  - title: Tree-shakeable modules
    details: Import only what you need via math/linalg/ml/signal/data subpaths.
  - title: Data interop
    details: Arrow-like ingestion and raw ArrayBuffer/SharedArrayBuffer support.
  - title: Memory safety
    details: Use gpu.scoped/gpu.tidy with FinalizationRegistry fallback cleanup.
  - title: Browser-tested
    details: Cross-browser Playwright coverage on Chromium, Firefox, and WebKit.
---

## Try It Live

<iframe src="https://phantasm0009.github.io/accel-gpu/playground/" width="100%" height="560"></iframe>

## Examples

- [Demo](https://phantasm0009.github.io/accel-gpu/example/)
- [Image Processing](https://phantasm0009.github.io/accel-gpu/example/image/)
- [Heatmap](https://phantasm0009.github.io/accel-gpu/example/heatmap/)
- [Neural Network](https://phantasm0009.github.io/accel-gpu/example/nn/)
- [N-Body](https://phantasm0009.github.io/accel-gpu/example/nbody/)
- [Local Audio Transcriber](https://phantasm0009.github.io/accel-gpu/example/audio/)
- [Vector Search (RAG)](https://phantasm0009.github.io/accel-gpu/example/vector-search/)
