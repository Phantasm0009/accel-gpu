/**
 * ML primitives - softmax, layerNorm, attention (inference-only)
 */

import type { GPUArray } from "../array";
import type { AccelContext } from "../types";

/**
 * Softmax over the last dimension.
 * Input shape: [rows, cols] - softmax applied per row.
 */
export async function softmax(
  ctx: AccelContext,
  input: GPUArray,
  rows?: number,
  cols?: number
): Promise<GPUArray> {
  let r: number, c: number;
  if (rows !== undefined && cols !== undefined) {
    r = rows;
    c = cols;
  } else if (input.shape.length === 2) {
    [r, c] = input.shape;
  } else if (input.shape.length === 1) {
    r = 1;
    c = input.shape[0];
  } else {
    throw new Error("softmax: provide rows and cols, or use array with shape.");
  }

  if (input.length !== r * c) {
    throw new Error(`softmax: shape mismatch — expected ${r * c} elements, got ${input.length}.`);
  }

  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
  const outBuffer = ctx.backend.createBuffer(r * c * 4, usage);

  await (
    ctx.runner as { softmax(i: unknown, o: unknown, r: number, c: number): Promise<void> }
  ).softmax(input.getBuffer(), outBuffer, r, c);

  const resultShape = r === 1 ? [c] : [r, c];
  return new (await import("../array")).GPUArray(
    ctx.backend,
    ctx.runner,
    outBuffer,
    r * c,
    resultShape
  );
}

/**
 * Layer normalization: (x - mean) / sqrt(var + eps) * gamma + beta
 */
export async function layerNorm(
  ctx: AccelContext,
  input: GPUArray,
  gamma: GPUArray,
  beta: GPUArray,
  rows?: number,
  cols?: number
): Promise<GPUArray> {
  let r: number, c: number;
  if (rows !== undefined && cols !== undefined) {
    r = rows;
    c = cols;
  } else if (input.shape.length === 2) {
    [r, c] = input.shape;
  } else {
    throw new Error("layerNorm: provide rows and cols, or use 2D array with shape.");
  }

  if (input.length !== r * c || gamma.length !== c || beta.length !== c) {
    throw new Error(
      `layerNorm: shape mismatch — input ${r}×${c}, gamma and beta must have length ${c}.`
    );
  }

  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
  const outBuffer = ctx.backend.createBuffer(r * c * 4, usage);

  const runner = ctx.runner as {
    layerNorm(i: unknown, g: unknown, b: unknown, o: unknown, r: number, c: number): Promise<void>;
  };
  await runner.layerNorm(input.getBuffer(), gamma.getBuffer(), beta.getBuffer(), outBuffer, r, c);

  return new (await import("../array")).GPUArray(ctx.backend, ctx.runner, outBuffer, r * c, [r, c]);
}

/**
 * Attention scores: Q @ K^T / sqrt(dim). Output shape [seq, seq].
 */
export async function attentionScores(
  ctx: AccelContext,
  Q: GPUArray,
  K: GPUArray,
  seq?: number,
  dim?: number
): Promise<GPUArray> {
  let s: number, d: number;
  if (seq !== undefined && dim !== undefined) {
    s = seq;
    d = dim;
  } else if (Q.shape.length === 2 && K.shape.length === 2) {
    [s, d] = Q.shape;
    if (K.shape[0] !== s || K.shape[1] !== d) {
      throw new Error(
        `attentionScores: Q is ${s}×${d}, K must match (got ${K.shape[0]}×${K.shape[1]}).`
      );
    }
  } else {
    throw new Error("attentionScores: provide seq and dim, or use 2D arrays with shape.");
  }

  if (Q.length !== s * d || K.length !== s * d) {
    throw new Error(`attentionScores: shape mismatch — Q and K must have ${s * d} elements.`);
  }

  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
  const outBuffer = ctx.backend.createBuffer(s * s * 4, usage);

  const runner = ctx.runner as {
    attentionScores(Q: unknown, K: unknown, o: unknown, seq: number, dim: number): Promise<void>;
  };
  await runner.attentionScores(Q.getBuffer(), K.getBuffer(), outBuffer, s, d);

  return new (await import("../array")).GPUArray(ctx.backend, ctx.runner, outBuffer, s * s, [s, s]);
}
