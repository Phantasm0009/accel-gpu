/**
 * Apache Arrow interoperability.
 *
 * Supports Arrow-like vectors from apache-arrow JS (duck-typed) and provides
 * zero-copy import when underlying values are Float32Array.
 */

import type { AccelContext, ArrowImportOptions } from "../types";
import type { GPUArray } from "../array";

type Typed =
  | Float32Array
  | Float64Array
  | Int8Array
  | Uint8Array
  | Int16Array
  | Uint16Array
  | Int32Array
  | Uint32Array;

function isTypedArray(v: unknown): v is Typed {
  return ArrayBuffer.isView(v) && !(v instanceof DataView);
}

function extractArrowValues(column: any): Typed {
  if (isTypedArray(column)) return column;
  if (column && isTypedArray(column.values)) return column.values;

  if (column && Array.isArray(column.data) && column.data.length > 0) {
    const first = column.data[0];
    if (first && isTypedArray(first.values)) return first.values;
  }

  if (column && typeof column.toArray === "function") {
    const arr = column.toArray();
    if (isTypedArray(arr)) return arr;
    if (Array.isArray(arr)) return Float32Array.from(arr);
  }

  throw new Error("fromArrow: unsupported Arrow column/vector shape");
}

export function fromArrow(
  ctx: AccelContext,
  column: unknown,
  options?: ArrowImportOptions
): GPUArray {
  const values = extractArrowValues(column as any);

  if (values instanceof Float32Array) {
    const zeroCopyView = new Float32Array(values.buffer, values.byteOffset, values.length);
    return ctx.array(zeroCopyView, options?.shape ?? [zeroCopyView.length]);
  }

  const converted = new Float32Array(values.length);
  for (let i = 0; i < values.length; i++) converted[i] = values[i] as number;
  return ctx.array(converted, options?.shape ?? [converted.length]);
}
