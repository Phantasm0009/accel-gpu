/**
 * accel-gpu - NumPy for the browser GPU
 * Zero shaders, zero dependencies.
 */
/// <reference types="@webgpu/types" />

// Polyfill GPUBufferUsage for Node (CPU backend) - values match WebGPU spec
if (typeof (globalThis as any).GPUBufferUsage === "undefined") {
  (globalThis as any).GPUBufferUsage = {
    MAP_READ: 1,
    COPY_SRC: 2,
    COPY_DST: 4,
    STORAGE: 8,
    UNIFORM: 16,
  };
}

import { createBackend } from "./backend/fallback";
import { GPUArray } from "./array";
import type { AccelContext } from "./types";
import type { WebGPUBackend } from "./backend/webgpu";
import type { WebGLBackend } from "./backend/webgl-backend";
import type { CPUBackend } from "./backend/cpu-backend";

export { GPUArray } from "./array";
export type { AccelContext, ProfilingEntry } from "./types";

/**
 * Options for initializing the Accel GPU context.
 */
export interface InitOptions {
  /** Prefer CPU backend (e.g. for testing or headless environments) */
  forceCPU?: boolean;
  /** Prefer WebGL2 backend (e.g. when WebGPU is unavailable) */
  forceWebGL?: boolean;
  /** Run in a Web Worker (reserved for future use) */
  worker?: boolean;
  /** Enable profiling from the start */
  profiling?: boolean;
}

/**
 * Initialize the Accel GPU context.
 *
 * Automatically selects the best available backend in order: WebGPU → WebGL2 → CPU.
 * Use `forceCPU` or `forceWebGL` to override for testing or compatibility.
 *
 * @param options - Optional initialization options
 * @returns Promise resolving to the Accel context with `array`, `fromImageData`, `toCanvas`, and backend info
 * @example
 * ```ts
 * const gpu = await init();
 * const arr = gpu.array([1, 2, 3]);
 * ```
 */
export async function init(options?: InitOptions): Promise<AccelContext> {
  let backend: WebGPUBackend | WebGLBackend | CPUBackend;
  let runner:
    | import("./backend/kernel-runner").KernelRunner
    | import("./backend/webgl-runner").WebGLRunner
    | import("./backend/cpu-runner").CPURunner;
  let backendType: "webgpu" | "webgl" | "cpu";

  if (options?.forceCPU) {
    const cpu = await import("./backend/cpu-backend");
    const cpuRunner = await import("./backend/cpu-runner");
    backend = cpu.createCPUBackend();
    runner = new cpuRunner.CPURunner();
    backendType = "cpu";
  } else if (options?.forceWebGL) {
    const webgl = await import("./backend/webgl-backend");
    const webglRunner = await import("./backend/webgl-runner");
    backend = webgl.createWebGLBackend();
    runner = new webglRunner.WebGLRunner(backend);
    backendType = "webgl";
  } else {
    const result = await createBackend();
    backend = result.backend;
    runner = result.runner;
    backendType = result.backendType;
  }

  const profilingEntries: import("./types").ProfilingEntry[] = [];
  let profilingEnabled = options?.profiling ?? false;

  const ctx: AccelContext = {
    backend,
    runner,
    backendType,
    enableProfiling(enable: boolean) {
      profilingEnabled = enable;
    },
    recordOp(op: string, durationMs: number) {
      if (profilingEnabled) {
        profilingEntries.push({ op, durationMs, timestamp: performance.now() });
      }
    },
    getProfilingResults() {
      return [...profilingEntries];
    },
    array(data: Float32Array | number[], shape?: number[]) {
      const arr = data instanceof Float32Array ? data : new Float32Array(data);
      const G = (globalThis as any).GPUBufferUsage;
      const usage = G.STORAGE | G.COPY_SRC | G.COPY_DST;
      const buffer = backend.createBufferFromData(arr.buffer as ArrayBuffer, usage);
      return new GPUArray(backend, runner, buffer, arr.length, shape ?? [arr.length]);
    },
    zeros(shape: number[]) {
      const size = shape.reduce((a, b) => a * b, 1);
      return ctx.array(new Float32Array(size).fill(0), shape);
    },
    ones(shape: number[]) {
      const size = shape.reduce((a, b) => a * b, 1);
      return ctx.array(new Float32Array(size).fill(1), shape);
    },
    full(shape: number[], value: number) {
      const size = shape.reduce((a, b) => a * b, 1);
      return ctx.array(new Float32Array(size).fill(value), shape);
    },
    arange(start: number, stop: number, step = 1) {
      const len = Math.max(0, Math.ceil((stop - start) / step));
      const arr = new Float32Array(len);
      for (let i = 0; i < len; i++) arr[i] = start + i * step;
      return ctx.array(arr);
    },
    linspace(start: number, stop: number, num: number) {
      const arr = new Float32Array(num);
      if (num === 1) arr[0] = start;
      else for (let i = 0; i < num; i++) arr[i] = start + (stop - start) * (i / (num - 1));
      return ctx.array(arr);
    },
    random(shape: number[]) {
      const size = shape.reduce((a, b) => a * b, 1);
      const arr = new Float32Array(size);
      for (let i = 0; i < size; i++) arr[i] = Math.random();
      return ctx.array(arr, shape);
    },
    randn(shape: number[]) {
      const size = shape.reduce((a, b) => a * b, 1);
      const arr = new Float32Array(size);
      for (let i = 0; i < size; i++) {
        const u1 = Math.random();
        const u2 = Math.random();
        arr[i] = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      }
      return ctx.array(arr, shape);
    },
    fromImageData(imageData: ImageData): GPUArray {
      const { width, height, data } = imageData;
      const floats = new Float32Array(width * height * 4);
      for (let i = 0; i < data.length; i++) floats[i] = data[i] / 255;
      return ctx.array(floats, [height, width, 4]);
    },
    async toCanvas(arr: GPUArray, width: number, height: number): Promise<HTMLCanvasElement> {
      const canvas = document.createElement("canvas");
      canvas.width = width;
      canvas.height = height;
      const ctx2d = canvas.getContext("2d")!;
      const imageData = ctx2d.createImageData(width, height);
      const data = await arr.toArray();
      for (let i = 0; i < data.length; i++) {
        imageData.data[i] = Math.max(0, Math.min(255, Math.round(data[i] * 255)));
      }
      ctx2d.putImageData(imageData, 0, 0);
      return canvas;
    },
  };

  return ctx;
}

/** Math ops: add, sub, mul, div, sum, max, min, mean. Mutate first arg in-place. */
export { add, sub, mul, div, sum, max, min, mean } from "./ops/math";
/** Linear algebra: matmul, dot, transpose, inv, det, solve, qr, svd. */
export { matmul, dot, transpose } from "./ops/linear";
export { inv, det, solve, qr, svd } from "./ops/matrix";
/** ML primitives: softmax, layerNorm, attentionScores, batchNorm. */
export { softmax, layerNorm, attentionScores, batchNorm } from "./ops/ml";
/** Convolution and pooling: maxPool2d, avgPool2d, conv2d. */
export { maxPool2d, avgPool2d, conv2d } from "./ops/conv";
/** FFT and spectrogram: fft, ifft, fftMagnitude, spectrogram. */
export { fft, ifft, fftMagnitude, spectrogram } from "./ops/fft";
