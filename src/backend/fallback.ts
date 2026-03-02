/**
 * Fallback logic: WebGPU -> WebGL -> CPU
 */

import { createWebGPUBackend } from "./webgpu";
import { createWebGLBackend } from "./webgl-backend";
import { createCPUBackend } from "./cpu-backend";
import { KernelRunner } from "./kernel-runner";
import { WebGLRunner } from "./webgl-runner";
import { CPURunner } from "./cpu-runner";
import type { WebGPUBackend } from "./webgpu";
import type { WebGLBackend } from "./webgl-backend";
import type { CPUBackend } from "./cpu-backend";

export type Backend = WebGPUBackend | WebGLBackend | CPUBackend;
export type Runner = KernelRunner | WebGLRunner | CPURunner;

export async function createBackend(): Promise<{
  backend: Backend;
  runner: Runner;
  backendType: "webgpu" | "webgl" | "cpu";
}> {
  try {
    if (typeof navigator !== "undefined" && navigator.gpu) {
      const backend = await createWebGPUBackend();
      const runner = new KernelRunner(backend);
      return { backend, runner, backendType: "webgpu" };
    }
  } catch {
    // WebGPU failed, fall through to WebGL
  }

  try {
    if (typeof document !== "undefined") {
      const backend = createWebGLBackend();
      const runner = new WebGLRunner(backend);
      return { backend, runner, backendType: "webgl" };
    }
  } catch {
    // WebGL2 failed, fall through to CPU
  }

  const backend = createCPUBackend();
  const runner = new CPURunner();
  return { backend, runner, backendType: "cpu" };
}
