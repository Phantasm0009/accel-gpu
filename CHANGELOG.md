# Changelog

All notable changes to accel-gpu will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [1.0.2] - 2026-03-04

### Fixed

- **WebKit browser consistency failure** — hardened WebGL `matmul` shader indexing math to use integer coordinate paths, eliminating intermittent `maxAbs: 154` drift in Playwright WebKit runs

### Changed

- **Backend consistency reliability** — `matmul` cross-backend checks now pass consistently across Chromium, Firefox, and WebKit in CI

## [1.0.1] - 2026-03-04

### Added

- **Backend consistency tests** — CPU/auto/WASM checks for `matmul`, `inv`, `fft`, and `conv2d`
- **Memory stress tests** — repeated and nested `tidy`/`scoped` lifecycle validation
- **Error canonicalization tests** — standardized validation message assertions
- **Bundle-size guard** — `test:bundle-size` script with CI enforcement
- **Backend tolerance docs** — new guide for cross-backend drift expectations and debugging

### Fixed

- **WebGL matmul indexing bug** — output coordinate mapping now uses output texture dimensions, fixing large numeric drift vs CPU

### Changed

- **Error message standardization** — unified shape/rank/power-of-two/even-length validation text across linear, matrix, conv, and FFT ops
- **Browser consistency checks** — strict matmul consistency restored after WebGL fix

## [1.0.0] - 2026-03-04

### Added

- **TypeScript shape safety (opt-in)** — generic `GPUArray<Shape>` plus shape-aware overloads for `matmul`, `dot`, and `transpose`
- **Docs theming** — custom VitePress theme styling and cleaner home/guide/reference presentation

### Changed

- **GitHub Pages docs routing fix** — `build:site` now includes VitePress build output so `/guide/quickstart` resolves correctly
- **Docs UX refresh** — improved navigation, sidebar grouping, search, footer, and page structure
- **`tidy()` semantics hardened** — preserves returned arrays, disposes intermediates, and supports nested tidy scope retention

## [0.2.10] - 2026-03-03

### Added

- **Raw buffer interop** — `fromBuffer(...)` and `gpu.fromBuffer(...)` for direct `ArrayBuffer`/`SharedArrayBuffer` Float32 ingestion
- **Docs site scaffold** — VitePress docs under `docs/` with quick start + API pages and embedded live iframes

### Changed

- **README slimmed down** — long-form API reference moved to docs site entry points

## [0.2.8] - 2026-03-03

### Added

- **WebGPU kernels** — `conv2d`, `maxPool2d`, `avgPool2d` now dispatch on WebGPU backend
- **WebGPU FFT kernels** — `fft()`, `ifft()`, and `fftMagnitude()` now use GPU compute on WebGPU backend
- **GPU matrix ops** — `inv()`, `qr()`, and `svd()` now use iterative WebGPU paths on WebGPU backend
- **CPU worker path (experimental)** — `init({ forceCPU: true, worker: true })`
- **CPU WASM path (experimental)** — `init({ forceCPU: true, preferWasmCPU: true, wasmModule })`
- **Runtime backend flags** — `gpu.workerEnabled`, `gpu.cpuEngine`
- **Scoped lifecycle API** — `gpu.scoped(fn)` for deterministic disposal of temporary arrays
- **Training helpers** — `gradients(...)` (numerical gradients) and `sgdStep(...)`
- **Arrow interop** — Apache Arrow-like vector/column import via `fromArrow(...)` and `gpu.fromArrow(...)`
- **Killer app demos** — `example/audio` (local audio spectrogram) and `example/vector-search` (browser-native vector search)
- **Interactive docs playground** — embedded playground on landing page
- **Tidy memory API** — `gpu.tidy(fn)` alias for scoped cleanup
- **Subpath exports** — `accel-gpu/math`, `accel-gpu/linalg`, `accel-gpu/ml`, `accel-gpu/signal`, `accel-gpu/data`
- **Cross-browser CI** — Playwright checks for Chromium, Firefox, and WebKit

### Changed

- `conv2d`, pooling, and FFT ops keep existing CPU/WebGL behavior as fallback while using WebGPU when available
- Updated docs and site examples for worker/WASM init options and accelerated op coverage
- Added troubleshooting guidance: `/.well-known/appspecific/com.chrome.devtools.json` 404 and `304` asset responses in local dev server logs are expected and harmless
- Added automatic scalar-chain fusion optimization for `add/sub/mul/div` before buffer materialization
- Added `FinalizationRegistry` best-effort cleanup for leaked arrays

## [0.2.7] - 2025-03-02

### Changed

- Re-publish with docs and site updates

## [0.2.6] - 2025-03-01

### Added

- **Reductions** — `variance()`, `std()`, `argmax()`, `argmin()`
- **Axis-specific reductions** — `sum(axis?)`, `mean(axis?)`, `max(axis?)`
- **Activations** — `gelu()`, `leakyRelu(alpha)`
- **Comparison** — `equal()`, `greater()`, `less()`, `clamp(min, max)`
- **Slicing** — `slice()`, `get()`, `set()`, `concat()`, `split()`
- **Shape** — `flatten()`, `squeeze()`, `unsqueeze()`, `broadcast()`
- **Memory** — `dispose()`, `isDisposed`, `toArraySync()` (CPU only)
- **Matrix ops** — `inv()`, `det()`, `solve()`, `qr()`, `svd()` (CPU)
- **ML** — `maxPool2d`, `avgPool2d`, `conv2d`, `batchNorm`, `normalize()`
- **FFT & signal** — `fft()`, `ifft()`, `fftMagnitude()`, `spectrogram()`
- **Other** — `norm()`, `outer()`, `mse()`, `crossEntropy()`
- **Profiling** — `enableProfiling()`, `recordOp()`, `getProfilingResults()`, `init({ profiling: true })`

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
