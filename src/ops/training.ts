/**
 * Training helpers - numerical gradients and simple SGD updates.
 *
 * This is backend-agnostic and works across WebGPU/WebGL/CPU contexts.
 */

import type { GPUArray } from "../array";
import type { AccelContext } from "../types";

/** Compute numerical gradients with central differences. */
export async function gradients(
  ctx: AccelContext,
  params: GPUArray[],
  lossFn: () => Promise<number>,
  epsilon = 1e-3
): Promise<GPUArray[]> {
  const grads: GPUArray[] = [];

  for (const param of params) {
    const grad = new Float32Array(param.length);
    const base = await param.toArray();

    for (let i = 0; i < param.length; i++) {
      const original = base[i];

      await param.set(i, original + epsilon);
      const lossPlus = await lossFn();

      await param.set(i, original - epsilon);
      const lossMinus = await lossFn();

      grad[i] = (lossPlus - lossMinus) / (2 * epsilon);
      await param.set(i, original);
    }

    grads.push(ctx.array(grad, param.shape));
  }

  return grads;
}

/** Apply SGD update: param -= learningRate * grad. */
export async function sgdStep(
  params: GPUArray[],
  grads: GPUArray[],
  learningRate: number
): Promise<void> {
  if (params.length !== grads.length) {
    throw new Error(`sgdStep: params/grads length mismatch (${params.length} vs ${grads.length})`);
  }

  for (let p = 0; p < params.length; p++) {
    const param = params[p];
    const grad = grads[p];
    if (param.length !== grad.length) {
      throw new Error(`sgdStep: param and grad length mismatch at index ${p}`);
    }

    const pData = await param.toArray();
    const gData = await grad.toArray();
    for (let i = 0; i < param.length; i++) {
      await param.set(i, pData[i] - learningRate * gData[i]);
    }
  }
}
