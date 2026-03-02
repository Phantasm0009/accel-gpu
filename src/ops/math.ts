/**
 * Math operations - add, mul, sum, max
 */

import type { GPUArray } from "../array";
import type { AccelContext } from "../types";

/** Element-wise add: a + b. Returns a new array (delegates to a.add(b)). */
export function add(ctx: AccelContext, a: GPUArray, b: GPUArray): Promise<GPUArray> {
  return a.add(b);
}

/** Element-wise multiply: a * b. Mutates a in-place. */
export function mul(ctx: AccelContext, a: GPUArray, b: GPUArray): Promise<GPUArray> {
  return a.mul(b);
}

/** Element-wise subtract: a - b. Mutates a in-place. */
export function sub(ctx: AccelContext, a: GPUArray, b: GPUArray): Promise<GPUArray> {
  return a.sub(b);
}

/** Element-wise divide: a / b. Mutates a in-place. */
export function div(ctx: AccelContext, a: GPUArray, b: GPUArray): Promise<GPUArray> {
  return a.div(b);
}

/** Reduce sum. Pass axis for axis-specific reduction. */
export function sum(ctx: AccelContext, a: GPUArray, axis?: number): Promise<number | GPUArray> {
  return a.sum(axis);
}

/** Reduce max over all elements of a. Pass axis for axis-specific reduction. */
export function max(ctx: AccelContext, a: GPUArray, axis?: number): Promise<number | GPUArray> {
  return a.max(axis);
}

/** Reduce min over all elements. */
export function min(ctx: AccelContext, a: GPUArray): Promise<number> {
  return a.min();
}

/** Mean (average). Pass axis for axis-specific reduction. */
export function mean(ctx: AccelContext, a: GPUArray, axis?: number): Promise<number | GPUArray> {
  return a.mean(axis);
}
