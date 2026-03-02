# Changelog

All notable changes to accel-gpu will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.2.5] - 2025-03-01

### Added

- **JSDoc** — Full documentation for public API (`init`, `InitOptions`, `AccelContext`, `GPUArray`, and all ops)
- **ESLint + Prettier** — Lint and format scripts; consistent code style
- **Vitest** — Test suite replacing `scripts/test.mjs`; `npm test` runs Vitest
- **Dependabot** — Weekly dependency updates for npm and GitHub Actions
- **Bundlephobia badge** — Package size badge in README
- **Package icon** — `icon.png` for docs and README

### Changed

- ESLint flat config (no `--ext`); Prettier formats all `src/**/*.ts`
- README includes icon and Bundlephobia badge; favicon in docs

## [0.2.0] - 2025-03-01

### Added

- **WebGL2 fallback** — Full WebGL2 backend when WebGPU unavailable (Safari, Firefox, older Chrome)
- **Shape inference** — `matmul(gpu, A, B)` infers M, N, K from array shapes
- **Method chaining** — `a.add(b).mul(2)` returns `this` for chaining (await each step)
- **reshape()** — Reshape arrays with shape metadata
- **CPU fallback** — Automatic fallback when WebGPU/WebGL unavailable (Node, headless)
- **Buffer pooling** — Reuse GPUBuffers for better performance
- **fromImageData() / toCanvas()** — Canvas integration for image processing
- **layerNorm** — Layer normalization kernel for transformers
- **attentionScores** — Q @ K^T / sqrt(dim) for attention
- **Clear error messages** — Descriptive errors with shape info
- **Benchmark page** — Compare WebGPU vs WebGL vs CPU performance
- **Playground** — Interactive code editor
- **forceCPU** init option — Force CPU backend for testing
- **forceWebGL** init option — Force WebGL2 backend for testing

### Changed

- Package renamed from `@accel/gpu` to `accel-gpu`
- `init()` now uses WebGPU → WebGL2 → CPU fallback chain
- `matmul`, `softmax`, `transpose` support shape inference from array metadata

## [0.1.0] - Initial Release

- Core API: `init()`, `gpu.array()`, `toArray()`
- Math ops: `add`, `mul`, `sum`, `max`
- Linear algebra: `matmul`, `dot`, `transpose`
- ML: `softmax`
- WebGPU backend
