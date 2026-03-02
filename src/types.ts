/**
 * Core types for Accel GPU
 */

import type { WebGPUBackend } from "./backend/webgpu";
import type { WebGLBackend } from "./backend/webgl-backend";
import type { CPUBackend } from "./backend/cpu-backend";
import type { KernelRunner } from "./backend/kernel-runner";
import type { WebGLRunner } from "./backend/webgl-runner";
import type { CPURunner } from "./backend/cpu-runner";
import type { GPUArray } from "./array";

/**
 * Context returned by {@link init}. Provides array creation and canvas helpers.
 */
export interface AccelContext {
  /** @internal */
  backend: WebGPUBackend | WebGLBackend | CPUBackend;
  /** @internal */
  runner: KernelRunner | WebGLRunner | CPURunner;
  /** Backend in use: 'webgpu' | 'webgl' | 'cpu' */
  backendType: "webgpu" | "webgl" | "cpu";
  /** Create a GPU-backed array from data. Optionally specify shape (e.g. [2, 3] for 2×3). */
  array(data: Float32Array | number[], shape?: number[]): GPUArray;
  /** Create a GPU-backed array from ImageData (RGBA, normalized to [0,1]). */
  fromImageData(imageData: ImageData): GPUArray;
  /** Render a GPU array to an HTMLCanvasElement (width×height). */
  toCanvas(arr: GPUArray, width: number, height: number): Promise<HTMLCanvasElement>;
}
