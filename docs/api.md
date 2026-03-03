# API Reference

## Context

- `init(options?)`
- `gpu.array(data, shape?)`
- `gpu.zeros(shape)` / `gpu.ones(shape)` / `gpu.full(shape, value)`
- `gpu.arange(start, stop, step?)` / `gpu.linspace(start, stop, num)`
- `gpu.random(shape)` / `gpu.randn(shape)`
- `gpu.fromArrow(column, options?)`
- `gpu.fromBuffer(buffer, options?)`
- `gpu.scoped(fn)` / `gpu.tidy(fn)`

## Math Ops

- `add`, `sub`, `mul`, `div`, `sum`, `max`, `min`, `mean`

## Linear Algebra

- `matmul`, `dot`, `transpose`, `inv`, `det`, `solve`, `qr`, `svd`

## ML & Convolution

- `softmax`, `layerNorm`, `batchNorm`, `attentionScores`
- `conv2d`, `maxPool2d`, `avgPool2d`

## Signal Processing

- `fft`, `ifft`, `fftMagnitude`, `spectrogram`

## Training

- `gradients`, `sgdStep`

## Shape Notes

- `matmul(gpu, A, B)` infers dimensions from array shape metadata.
- `layerNorm(gpu, input, gamma, beta, rows?, cols?)` supports explicit row/column shape override.

## Live API Playground

<iframe src="https://phantasm0009.github.io/accel-gpu/playground/" width="100%" height="560" style="border:1px solid #ddd;border-radius:8px;"></iframe>
