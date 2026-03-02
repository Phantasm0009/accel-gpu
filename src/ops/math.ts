/**
 * Math operations - add, mul, sum, max
 */

import type { GPUArray } from "../array";
import type { AccelContext } from "../types";

/** Element-wise add: a + b. Returns a new array (delegates to a.add(b)). */
export function add(ctx: AccelContext, a: GPUArray, b: GPUArray): Promise<GPUArray> {
  return a.add(b);
}

/** Element-wise multiply: a * b. Returns a new array (delegates to a.mul(b)). */
export function mul(ctx: AccelContext, a: GPUArray, b: GPUArray): Promise<GPUArray> {
  return a.mul(b);
}

export function sum(ctx: AccelContext, a: GPUArray): Promise<number> {
  return a.sum();
}

/** Reduce max over all elements of a. */
export function max(ctx: AccelContext, a: GPUArray): Promise<number> {
  return a.max();
}
