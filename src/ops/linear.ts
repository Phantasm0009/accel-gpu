/**
 * Linear algebra operations - matmul, dot, transpose
 */

import type { GPUArray } from "../array";
import type { AccelContext } from "../types";
import { errMatmulShapes } from "../errors";

/** Infer M, N, K from shapes. A is M×K, B is K×N, result is M×N. */
function inferMatmulShapes(a: GPUArray, b: GPUArray): { M: number; N: number; K: number } {
  const aShape = a.shape;
  const bShape = b.shape;

  if (aShape.length === 1 && bShape.length === 1) {
    if (aShape[0] !== bShape[0]) {
      errMatmulShapes(
        "matmul",
        `[${aShape[0]}]`,
        `[${bShape[0]}]`,
        "vectors must have same length for dot product."
      );
    }
    return { M: 1, N: 1, K: aShape[0] };
  }

  if (aShape.length === 2 && bShape.length === 2) {
    const [M, K] = aShape;
    const [K2, N] = bShape;
    if (K !== K2) {
      errMatmulShapes(
        "matmul",
        `${M}×${K}`,
        `${K2}×${N}`,
        `inner dimensions must match (${K} vs ${K2}).`
      );
    }
    return { M, N, K };
  }

  if (aShape.length === 2 && bShape.length === 1) {
    const [M, K] = aShape;
    const [N] = bShape;
    if (K !== N) {
      errMatmulShapes(
        "matmul",
        `${M}×${K}`,
        `[${N}]`,
        `matrix columns (${K}) must match vector length (${N}).`
      );
    }
    return { M, N: 1, K };
  }

  if (aShape.length === 1 && bShape.length === 2) {
    const [K] = aShape;
    const [K2, N] = bShape;
    if (K !== K2) {
      errMatmulShapes(
        "matmul",
        `[${K}]`,
        `${K2}×${N}`,
        `vector length (${K}) must match matrix rows (${K2}).`
      );
    }
    return { M: 1, N, K };
  }

  if (aShape.length === 1 && bShape.length === 1) {
    return { M: 1, N: 1, K: aShape[0] };
  }

  errMatmulShapes(
    "matmul",
    `[${aShape.join(", ")}]`,
    `[${bShape.join(", ")}]`,
    "unsupported shape combination."
  );
}

/**
 * Matrix multiply.
 * @param ctx - Accel context
 * @param a - Left matrix (M×K)
 * @param b - Right matrix (K×N)
 * @param M - Optional explicit rows of A
 * @param N - Optional explicit cols of B
 * @param K - Optional explicit inner dimension
 * @returns Promise resolving to the result matrix (M×N)
 */
export async function matmul(
  ctx: AccelContext,
  a: GPUArray,
  b: GPUArray,
  M?: number,
  N?: number,
  K?: number
): Promise<GPUArray> {
  let dims: { M: number; N: number; K: number };
  if (M !== undefined && N !== undefined && K !== undefined) {
    dims = { M, N, K };
    if (a.length !== M * K || b.length !== K * N) {
      throw new Error(
        `matmul: shape mismatch — A should be ${M}×${K} (${M * K} elements), got ${a.length}. B should be ${K}×${N} (${K * N} elements), got ${b.length}.`
      );
    }
  } else {
    dims = inferMatmulShapes(a, b);
  }

  const { M: m, N: n, K: k } = dims;

  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
  const outBuffer = ctx.backend.createBuffer(m * n * 4, usage);

  await (
    ctx.runner as {
      matmul(a: unknown, b: unknown, o: unknown, M: number, N: number, K: number): Promise<void>;
    }
  ).matmul(a.getBuffer(), b.getBuffer(), outBuffer, m, n, k);

  const resultShape = m === 1 && n === 1 ? [1] : n === 1 ? [m] : [m, n];
  return new (await import("../array")).GPUArray(
    ctx.backend,
    ctx.runner,
    outBuffer,
    m * n,
    resultShape
  );
}

export async function dot(ctx: AccelContext, a: GPUArray, b: GPUArray): Promise<number> {
  return a.dot(b);
}

/**
 * Transpose a matrix.
 * @param ctx - Accel context
 * @param a - Input matrix (rows×cols)
 * @param rows - Optional explicit rows
 * @param cols - Optional explicit cols
 * @returns Promise resolving to transposed matrix (cols×rows)
 */
export async function transpose(
  ctx: AccelContext,
  a: GPUArray,
  rows?: number,
  cols?: number
): Promise<GPUArray> {
  let r: number, c: number;
  if (rows !== undefined && cols !== undefined) {
    r = rows;
    c = cols;
  } else if (a.shape.length === 2) {
    [r, c] = a.shape;
  } else {
    throw new Error("transpose: provide rows and cols, or use a 2D array with shape.");
  }

  if (a.length !== r * c) {
    throw new Error(`transpose: shape mismatch — expected ${r * c} elements, got ${a.length}.`);
  }

  const data = await a.toArray();
  const out = new Float32Array(r * c);
  for (let i = 0; i < r; i++) {
    for (let j = 0; j < c; j++) {
      out[j * r + i] = data[i * c + j];
    }
  }
  return ctx.array(out, [c, r]);
}
