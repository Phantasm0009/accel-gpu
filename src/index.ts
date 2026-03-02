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
export type { AccelContext } from "./types";

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

  const ctx: AccelContext = {
    backend,
    runner,
    backendType,
    array(data: Float32Array | number[], shape?: number[]) {
      const arr = data instanceof Float32Array ? data : new Float32Array(data);
      const G = (globalThis as any).GPUBufferUsage;
      const usage = G.STORAGE | G.COPY_SRC | G.COPY_DST;
      const buffer = backend.createBufferFromData(arr.buffer as ArrayBuffer, usage);
      return new GPUArray(backend, runner, buffer, arr.length, shape ?? [arr.length]);
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

/** Math ops: add, mul, sum, max. Mutate first arg in-place. */
export { add, mul, sum, max } from "./ops/math";
/** Linear algebra: matmul, dot, transpose. */
export { matmul, dot, transpose } from "./ops/linear";
/** ML primitives: softmax, layerNorm, attentionScores. */
export { softmax, layerNorm, attentionScores } from "./ops/ml";
