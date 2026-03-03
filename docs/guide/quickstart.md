# Quick Start

Install, initialize, run a few ops, and choose the right import path.

## Install

```bash
npm install accel-gpu
```

## Initialize

```ts
import { init } from "accel-gpu";

const gpu = await init();
console.log(gpu.backendType); // "webgpu" | "webgl" | "cpu"
```

You can force a backend for testing:

```ts
await init({ forceCPU: true });
await init({ forceWebGL: true });
```

## Basic Ops

```ts
const a = gpu.array([1, 2, 3, 4]);
const b = gpu.array([5, 6, 7, 8]);
await a.add(b);
await a.mul(2);
console.log(await a.sum());
```

## Memory Safety

```ts
await gpu.tidy(async (ctx) => {
  const tmp = ctx.array([1, 2, 3]);
  await tmp.mul(2);
});
```

## Data Ingestion

```ts
import { fromArrow, fromBuffer } from "accel-gpu/data";

const a = fromArrow(gpu, arrowColumn, { shape: [rows, cols] });
const b = fromBuffer(gpu, sharedArrayBuffer, { shape: [rows, cols] });
```

## Next Steps

- Read the [API Reference](/api)
- Try the [playground](https://phantasm0009.github.io/accel-gpu/playground/)
- Explore [examples](https://phantasm0009.github.io/accel-gpu/example/)
