/**
 * Matrix operations - inv, det, solve, qr, svd
 * WebGPU iterative paths for inv/qr/svd with CPU fallback implementations
 */

import type { GPUArray } from "../array";
import type { AccelContext } from "../types";
import { matmul, transpose } from "./linear";

function copyTo2D(arr: Float32Array, rows: number, cols: number): number[][] {
  const m: number[][] = [];
  for (let i = 0; i < rows; i++) {
    m[i] = [];
    for (let j = 0; j < cols; j++) m[i][j] = arr[i * cols + j];
  }
  return m;
}

function copyFrom2D(m: number[][]): Float32Array {
  const r = m.length;
  const c = m[0].length;
  const out = new Float32Array(r * c);
  for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) out[i * c + j] = m[i][j];
  return out;
}

function luDecompose(a: number[][]): { L: number[][]; U: number[][]; perm: number[] } {
  const n = a.length;
  const L = Array.from({ length: n }, () => Array(n).fill(0));
  const U = a.map((row) => [...row]);
  const perm = [...Array(n).keys()];

  for (let k = 0; k < n; k++) {
    let max = 0;
    let pivot = k;
    for (let i = k; i < n; i++) {
      if (Math.abs(U[i][k]) > max) {
        max = Math.abs(U[i][k]);
        pivot = i;
      }
    }
    if (max < 1e-10) throw new Error("inv: matrix is singular");
    [U[k], U[pivot]] = [U[pivot], U[k]];
    [perm[k], perm[pivot]] = [perm[pivot], perm[k]];
    for (let i = 0; i < k; i++) [L[k][i], L[pivot][i]] = [L[pivot][i], L[k][i]];

    L[k][k] = 1;
    for (let i = k + 1; i < n; i++) {
      L[i][k] = U[i][k] / U[k][k];
      for (let j = k; j < n; j++) U[i][j] -= L[i][k] * U[k][j];
    }
  }
  return { L, U, perm };
}

function detFromLU(U: number[][]): number {
  let d = 1;
  for (let i = 0; i < U.length; i++) d *= U[i][i];
  return d;
}

function solveLower(L: number[][], b: number[]): number[] {
  const n = L.length;
  const x = [...b];
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < i; j++) x[i] -= L[i][j] * x[j];
    x[i] /= L[i][i];
  }
  return x;
}

function solveUpper(U: number[][], b: number[]): number[] {
  const n = U.length;
  const x = [...b];
  for (let i = n - 1; i >= 0; i--) {
    for (let j = i + 1; j < n; j++) x[i] -= U[i][j] * x[j];
    x[i] /= U[i][i];
  }
  return x;
}

function invertFromLU(L: number[][], U: number[][], perm: number[]): number[][] {
  const n = L.length;
  const inv: number[][] = [];
  for (let j = 0; j < n; j++) {
    const b = perm.map((_, i) => (perm[i] === j ? 1 : 0));
    const y = solveLower(L, b);
    inv.push(solveUpper(U, y));
  }
  return inv.map((col, j) => col.map((_, i) => inv[i][j]));
}

function identityData(n: number): Float32Array {
  const out = new Float32Array(n * n);
  for (let i = 0; i < n; i++) out[i * n + i] = 1;
  return out;
}

function matrixNorm1(data: Float32Array, rows: number, cols: number): number {
  let max = 0;
  for (let c = 0; c < cols; c++) {
    let sum = 0;
    for (let r = 0; r < rows; r++) sum += Math.abs(data[r * cols + c]);
    if (sum > max) max = sum;
  }
  return max;
}

function matrixNormInf(data: Float32Array, rows: number, cols: number): number {
  let max = 0;
  for (let r = 0; r < rows; r++) {
    let sum = 0;
    for (let c = 0; c < cols; c++) sum += Math.abs(data[r * cols + c]);
    if (sum > max) max = sum;
  }
  return max;
}

async function invWebGPU(ctx: AccelContext, a: GPUArray, n: number): Promise<GPUArray> {
  await a.materialize();
  const aData = await a.toArray();
  const norm1 = matrixNorm1(aData, n, n);
  const normInf = matrixNormInf(aData, n, n);
  const scale = 1 / Math.max(1e-8, norm1 * normInf);

  const at = await transpose(ctx, a, n, n);
  let X = await at.clone();
  await X.mul(scale);
  at.dispose();

  const I = ctx.array(identityData(n), [n, n]);

  for (let iter = 0; iter < 8; iter++) {
    const AX = await matmul(ctx, a, X, n, n, n);
    const M = await I.clone();
    await M.mul(2);
    await M.sub(AX);
    AX.dispose();

    const Xnext = await matmul(ctx, X, M, n, n, n);
    M.dispose();
    X.dispose();
    X = Xnext;
  }

  I.dispose();
  return X;
}

async function qrWebGPU(
  ctx: AccelContext,
  a: GPUArray,
  m: number,
  n: number
): Promise<{ Q: GPUArray; R: GPUArray }> {
  await a.materialize();
  let Q = await a.clone();
  const I = ctx.array(identityData(n), [n, n]);

  for (let iter = 0; iter < 6; iter++) {
    const Qt = await transpose(ctx, Q, m, n);
    const QtQ = await matmul(ctx, Qt, Q, n, n, m);
    Qt.dispose();

    const T = await I.clone();
    await T.mul(3);
    await T.sub(QtQ);
    QtQ.dispose();
    await T.mul(0.5);

    const Qnext = await matmul(ctx, Q, T, m, n, n);
    T.dispose();
    Q.dispose();
    Q = Qnext;
  }

  const Qt = await transpose(ctx, Q, m, n);
  const R = await matmul(ctx, Qt, a, n, n, m);
  Qt.dispose();
  I.dispose();
  return { Q, R };
}

async function svdWebGPU(
  ctx: AccelContext,
  a: GPUArray,
  m: number,
  n: number
): Promise<{ U: GPUArray; S: GPUArray; V: GPUArray }> {
  await a.materialize();
  const k = Math.min(m, n);
  const At = await transpose(ctx, a, m, n);
  let B = await matmul(ctx, At, a, n, n, m);
  At.dispose();

  const Ucols: Float32Array[] = [];
  const Vcols: Float32Array[] = [];
  const S = new Float32Array(k);

  for (let comp = 0; comp < k; comp++) {
    let v = ctx.random([n]);
    for (let it = 0; it < 16; it++) {
      const Bv = await matmul(ctx, B, v, n, 1, n);
      const norm = Math.max(1e-8, await Bv.norm(2));
      await Bv.div(norm);
      v.dispose();
      v = Bv;
    }

    const Bv = await matmul(ctx, B, v, n, 1, n);
    const lambda = await v.dot(Bv);
    Bv.dispose();
    const sigma = Math.sqrt(Math.max(0, lambda));
    S[comp] = sigma;

    const Av = await matmul(ctx, a, v, m, 1, n);
    if (sigma > 1e-8) await Av.div(sigma);
    const uData = await Av.toArray();
    const vData = await v.toArray();
    Ucols.push(uData);
    Vcols.push(vData);

    const vvT = await v.outer(v);
    await vvT.mul(lambda);
    await B.sub(vvT);
    vvT.dispose();
    Av.dispose();
    v.dispose();
  }

  const Uarr = new Float32Array(m * k);
  const Varr = new Float32Array(n * k);
  for (let i = 0; i < m; i++) for (let j = 0; j < k; j++) Uarr[i * k + j] = Ucols[j][i] ?? 0;
  for (let i = 0; i < n; i++) for (let j = 0; j < k; j++) Varr[i * k + j] = Vcols[j][i] ?? 0;

  B.dispose();
  return {
    U: ctx.array(Uarr, [m, k]),
    S: ctx.array(S, [k]),
    V: ctx.array(Varr, [n, k]),
  };
}

/**
 * Matrix inverse. Input must be square matrix.
 */
export async function inv(
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
    throw new Error("inv: provide rows and cols, or use 2D array");
  }
  if (r !== c) throw new Error(`inv: matrix must be square, got ${r}×${c}`);

  if (ctx.backendType === "webgpu") {
    return invWebGPU(ctx, a, r);
  }

  const data = await a.toArray();
  const m = copyTo2D(data, r, c);
  const { L, U, perm } = luDecompose(m);
  const invM = invertFromLU(L, U, perm);
  const out = copyFrom2D(invM);
  const G = (globalThis as any).GPUBufferUsage;
  const usage = G.STORAGE | G.COPY_SRC | G.COPY_DST;
  const buffer = ctx.backend.createBufferFromData(out.buffer as ArrayBuffer, usage);
  return new (await import("../array")).GPUArray(ctx.backend, ctx.runner, buffer, r * r, [r, r]);
}

/**
 * Determinant of a square matrix.
 */
export async function det(
  ctx: AccelContext,
  a: GPUArray,
  rows?: number,
  cols?: number
): Promise<number> {
  let r: number, c: number;
  if (rows !== undefined && cols !== undefined) {
    r = rows;
    c = cols;
  } else if (a.shape.length === 2) {
    [r, c] = a.shape;
  } else {
    throw new Error("det: provide rows and cols, or use 2D array");
  }
  if (r !== c) throw new Error(`det: matrix must be square, got ${r}×${c}`);
  const data = await a.toArray();
  const m = copyTo2D(data, r, c);
  const { U } = luDecompose(m);
  return detFromLU(U);
}

/**
 * Solve Ax = b. Returns x.
 */
export async function solve(
  ctx: AccelContext,
  A: GPUArray,
  b: GPUArray,
  rows?: number
): Promise<GPUArray> {
  let n: number;
  if (rows !== undefined) {
    n = rows;
  } else if (A.shape.length === 2) {
    n = A.shape[0];
  } else {
    throw new Error("solve: provide rows or use 2D A");
  }
  if (b.length !== n) throw new Error(`solve: b must have length ${n}`);
  const aData = await A.toArray();
  const bData = await b.toArray();
  const m = copyTo2D(aData, n, n);
  const { L, U, perm } = luDecompose(m);
  const pb = perm.map((i) => bData[i]);
  const y = solveLower(L, pb);
  const x = solveUpper(U, y);
  const G = (globalThis as any).GPUBufferUsage;
  const usage = G.STORAGE | G.COPY_SRC | G.COPY_DST;
  const buffer = ctx.backend.createBufferFromData(
    new Float32Array(x).buffer as ArrayBuffer,
    usage
  );
  return new (await import("../array")).GPUArray(ctx.backend, ctx.runner, buffer, n, [n]);
}

/**
 * QR decomposition: A = Q * R. Returns { Q, R }.
 */
export async function qr(
  ctx: AccelContext,
  a: GPUArray,
  rows?: number,
  cols?: number
): Promise<{ Q: GPUArray; R: GPUArray }> {
  let m: number, n: number;
  if (rows !== undefined && cols !== undefined) {
    m = rows;
    n = cols;
  } else if (a.shape.length === 2) {
    [m, n] = a.shape;
  } else {
    throw new Error("qr: provide rows and cols, or use 2D array");
  }
  if (ctx.backendType === "webgpu") {
    return qrWebGPU(ctx, a, m, n);
  }

  const data = await a.toArray();
  const A = copyTo2D(data, m, n);
  const Q: number[][] = Array.from({ length: m }, (_, i) =>
    Array.from({ length: m }, (_, j) => (i === j ? 1 : 0))
  );
  const R = A.map((row) => [...row]);

  for (let k = 0; k < Math.min(m, n); k++) {
    let norm = 0;
    for (let i = k; i < m; i++) norm += R[i][k] * R[i][k];
    norm = Math.sqrt(norm);
    if (norm < 1e-10) continue;
    const u = new Array(m).fill(0);
    u[k] = R[k][k] + (R[k][k] >= 0 ? 1 : -1) * norm;
    for (let i = k + 1; i < m; i++) u[i] = R[i][k];
    let uNorm = 0;
    for (let i = k; i < m; i++) uNorm += u[i] * u[i];
    uNorm = Math.sqrt(uNorm);
    for (let i = k; i < m; i++) u[i] /= uNorm;

    for (let j = k; j < n; j++) {
      let dot = 0;
      for (let i = k; i < m; i++) dot += u[i] * R[i][j];
      for (let i = k; i < m; i++) R[i][j] -= 2 * u[i] * dot;
    }
    for (let j = 0; j < m; j++) {
      let dot = 0;
      for (let i = k; i < m; i++) dot += u[i] * Q[i][j];
      for (let i = k; i < m; i++) Q[i][j] -= 2 * u[i] * dot;
    }
  }

  const Qarr = copyFrom2D(Q);
  const Rarr = copyFrom2D(R);
  const G = (globalThis as any).GPUBufferUsage;
  const usage = G.STORAGE | G.COPY_SRC | G.COPY_DST;
  const Qbuf = ctx.backend.createBufferFromData(Qarr.buffer as ArrayBuffer, usage);
  const Rbuf = ctx.backend.createBufferFromData(Rarr.buffer as ArrayBuffer, usage);
  const GPUArray = (await import("../array")).GPUArray;
  return {
    Q: new GPUArray(ctx.backend, ctx.runner, Qbuf, m * m, [m, m]),
    R: new GPUArray(ctx.backend, ctx.runner, Rbuf, m * n, [m, n]),
  };
}

/**
 * SVD: A = U * S * V^T. Returns { U, S, V }.
 * S is a 1D array of singular values. Uses power iteration for small matrices.
 */
export async function svd(
  ctx: AccelContext,
  a: GPUArray,
  rows?: number,
  cols?: number
): Promise<{ U: GPUArray; S: GPUArray; V: GPUArray }> {
  let m: number, n: number;
  if (rows !== undefined && cols !== undefined) {
    m = rows;
    n = cols;
  } else if (a.shape.length === 2) {
    [m, n] = a.shape;
  } else {
    throw new Error("svd: provide rows and cols, or use 2D array");
  }
  if (ctx.backendType === "webgpu") {
    return svdWebGPU(ctx, a, m, n);
  }

  const data = await a.toArray();
  const A = copyTo2D(data, m, n);

  const k = Math.min(m, n);
  const S = new Float32Array(k);
  const U = Array.from({ length: m }, () => new Array(k).fill(0));
  const V = Array.from({ length: n }, () => new Array(k).fill(0));

  let remainder = A.map((row) => [...row]);

  for (let i = 0; i < k; i++) {
    let v = new Array(n).fill(0);
    v[Math.min(i, n - 1)] = 1;
    for (let iter = 0; iter < 50; iter++) {
      const u = new Array(m).fill(0);
      for (let r = 0; r < m; r++) {
        for (let c = 0; c < n; c++) u[r] += remainder[r][c] * v[c];
      }
      let unorm = Math.sqrt(u.reduce((s, x) => s + x * x, 0)) || 1;
      for (let r = 0; r < m; r++) u[r] /= unorm;

      v = new Array(n).fill(0);
      for (let c = 0; c < n; c++) {
        for (let r = 0; r < m; r++) v[c] += remainder[r][c] * u[r];
      }
      let vnorm = Math.sqrt(v.reduce((s, x) => s + x * x, 0)) || 1;
      for (let c = 0; c < n; c++) v[c] /= vnorm;
    }
    let s = 0;
    for (let r = 0; r < m; r++) {
      for (let c = 0; c < n; c++) s += remainder[r][c] * v[c] * (r < m ? 1 : 0);
    }
    const u = new Array(m).fill(0);
    for (let r = 0; r < m; r++) {
      for (let c = 0; c < n; c++) u[r] += remainder[r][c] * v[c];
    }
    const unorm = Math.sqrt(u.reduce((s, x) => s + x * x, 0)) || 1;
    s = unorm;
    S[i] = s;
    for (let r = 0; r < m; r++) U[r][i] = u[r] / (s || 1);
    for (let c = 0; c < n; c++) V[c][i] = v[c];
    for (let r = 0; r < m; r++) {
      for (let c = 0; c < n; c++) remainder[r][c] -= (u[r] / (s || 1)) * v[c] * s;
    }
  }

  const G = (globalThis as any).GPUBufferUsage;
  const usage = G.STORAGE | G.COPY_SRC | G.COPY_DST;
  const Uarr = new Float32Array(m * k);
  const Varr = new Float32Array(n * k);
  for (let i = 0; i < m; i++) for (let j = 0; j < k; j++) Uarr[i * k + j] = U[i][j];
  for (let i = 0; i < n; i++) for (let j = 0; j < k; j++) Varr[i * k + j] = V[i][j];
  const Ubuf = ctx.backend.createBufferFromData(Uarr.buffer as ArrayBuffer, usage);
  const Vbuf = ctx.backend.createBufferFromData(Varr.buffer as ArrayBuffer, usage);
  const Sbuf = ctx.backend.createBufferFromData(S.buffer as ArrayBuffer, usage);
  const GPUArray = (await import("../array")).GPUArray;
  return {
    U: new GPUArray(ctx.backend, ctx.runner, Ubuf, m * k, [m, k]),
    S: new GPUArray(ctx.backend, ctx.runner, Sbuf, k, [k]),
    V: new GPUArray(ctx.backend, ctx.runner, Vbuf, n * k, [n, k]),
  };
}
