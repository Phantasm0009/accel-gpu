/**
 * Clear, descriptive error messages. Stack traces are included automatically when thrown.
 */

export function errLengthMismatch(op: string, a: number, b: number): never {
  throw new Error(`${op}: length mismatch — ${a} vs ${b}. Arrays must have the same length.`);
}

export function errShapeMismatch(op: string, expected: string, actual: string): never {
  throw new Error(`${op}: shape mismatch — expected ${expected}, got ${actual}.`);
}

export function errMatmulShapes(op: string, aShape: string, bShape: string, reason: string): never {
  throw new Error(`${op}: A is ${aShape}, B is ${bShape} — ${reason}`);
}

export function errInvalidShape(op: string, shape: number[], expectedElements: number): never {
  throw new Error(
    `${op}: shape [${shape.join(", ")}] has ${shape.reduce((a, b) => a * b, 1)} elements, expected ${expectedElements}.`
  );
}
