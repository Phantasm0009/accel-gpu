/**
 * Convolution and pooling - maxPool2d, avgPool2d, conv2d
 */

import type { GPUArray } from "../array";
import type { AccelContext } from "../types";
import type { KernelRunner } from "../backend/kernel-runner";

function get4DShape(
  input: GPUArray,
  h?: number,
  w?: number,
  c?: number
): { n: number; h: number; w: number; c: number } {
  if (h !== undefined && w !== undefined && c !== undefined) {
    return { n: 1, h, w, c };
  }
  const s = input.shape;
  if (s.length === 4) return { n: s[0], h: s[1], w: s[2], c: s[3] };
  if (s.length === 3) return { n: 1, h: s[0], w: s[1], c: s[2] };
  throw new Error("conv/pool: need 3D [H,W,C] or 4D [N,H,W,C] shape");
}

/**
 * Max pooling 2D. Input [N,H,W,C] or [H,W,C].
 */
export async function maxPool2d(
  ctx: AccelContext,
  input: GPUArray,
  kernelSize: number,
  stride?: number,
  padding = 0,
  h?: number,
  w?: number,
  c?: number
): Promise<GPUArray> {
  const { n, h: H, w: W, c: C } = get4DShape(input, h, w, c);
  const s = stride ?? kernelSize;
  const pH = Math.floor((H + 2 * padding - kernelSize) / s) + 1;
  const pW = Math.floor((W + 2 * padding - kernelSize) / s) + 1;

  if (ctx.backendType === "webgpu") {
    await input.materialize();
    const G = (globalThis as any).GPUBufferUsage;
    const usage = G.STORAGE | G.COPY_SRC | G.COPY_DST;
    const out = ctx.backend.createBuffer(n * pH * pW * C * 4, usage);
    await (ctx.runner as KernelRunner).maxPool2d(
      input.getBuffer() as GPUBuffer,
      out as GPUBuffer,
      n,
      H,
      W,
      C,
      kernelSize,
      s,
      padding,
      pH,
      pW
    );
    return new (await import("../array")).GPUArray(
      ctx.backend,
      ctx.runner,
      out,
      n * pH * pW * C,
      n === 1 ? [pH, pW, C] : [n, pH, pW, C]
    );
  }

  const data = await input.toArray();

  const out = new Float32Array(n * pH * pW * C);
  for (let ni = 0; ni < n; ni++) {
    for (let yi = 0; yi < pH; yi++) {
      for (let xi = 0; xi < pW; xi++) {
        for (let ci = 0; ci < C; ci++) {
          let max = -Infinity;
          for (let ky = 0; ky < kernelSize; ky++) {
            for (let kx = 0; kx < kernelSize; kx++) {
              const y = yi * s + ky - padding;
              const x = xi * s + kx - padding;
              if (y >= 0 && y < H && x >= 0 && x < W) {
                const v = data[ni * H * W * C + y * W * C + x * C + ci];
                if (v > max) max = v;
              }
            }
          }
          out[ni * pH * pW * C + yi * pW * C + xi * C + ci] = max === -Infinity ? 0 : max;
        }
      }
    }
  }
  const G = (globalThis as any).GPUBufferUsage;
  const buffer = ctx.backend.createBufferFromData(out.buffer as ArrayBuffer, G.STORAGE | G.COPY_SRC | G.COPY_DST);
  return new (await import("../array")).GPUArray(
    ctx.backend,
    ctx.runner,
    buffer,
    out.length,
    n === 1 ? [pH, pW, C] : [n, pH, pW, C]
  );
}

/**
 * Average pooling 2D.
 */
export async function avgPool2d(
  ctx: AccelContext,
  input: GPUArray,
  kernelSize: number,
  stride?: number,
  padding = 0,
  h?: number,
  w?: number,
  c?: number
): Promise<GPUArray> {
  const { n, h: H, w: W, c: C } = get4DShape(input, h, w, c);
  const s = stride ?? kernelSize;
  const pH = Math.floor((H + 2 * padding - kernelSize) / s) + 1;
  const pW = Math.floor((W + 2 * padding - kernelSize) / s) + 1;

  if (ctx.backendType === "webgpu") {
    await input.materialize();
    const G = (globalThis as any).GPUBufferUsage;
    const usage = G.STORAGE | G.COPY_SRC | G.COPY_DST;
    const out = ctx.backend.createBuffer(n * pH * pW * C * 4, usage);
    await (ctx.runner as KernelRunner).avgPool2d(
      input.getBuffer() as GPUBuffer,
      out as GPUBuffer,
      n,
      H,
      W,
      C,
      kernelSize,
      s,
      padding,
      pH,
      pW
    );
    return new (await import("../array")).GPUArray(
      ctx.backend,
      ctx.runner,
      out,
      n * pH * pW * C,
      n === 1 ? [pH, pW, C] : [n, pH, pW, C]
    );
  }

  const data = await input.toArray();

  const out = new Float32Array(n * pH * pW * C);
  const k2 = kernelSize * kernelSize;
  for (let ni = 0; ni < n; ni++) {
    for (let yi = 0; yi < pH; yi++) {
      for (let xi = 0; xi < pW; xi++) {
        for (let ci = 0; ci < C; ci++) {
          let sum = 0;
          let count = 0;
          for (let ky = 0; ky < kernelSize; ky++) {
            for (let kx = 0; kx < kernelSize; kx++) {
              const y = yi * s + ky - padding;
              const x = xi * s + kx - padding;
              if (y >= 0 && y < H && x >= 0 && x < W) {
                sum += data[ni * H * W * C + y * W * C + x * C + ci];
                count++;
              }
            }
          }
          out[ni * pH * pW * C + yi * pW * C + xi * C + ci] = count > 0 ? sum / count : 0;
        }
      }
    }
  }
  const G = (globalThis as any).GPUBufferUsage;
  const buffer = ctx.backend.createBufferFromData(out.buffer as ArrayBuffer, G.STORAGE | G.COPY_SRC | G.COPY_DST);
  return new (await import("../array")).GPUArray(
    ctx.backend,
    ctx.runner,
    buffer,
    out.length,
    n === 1 ? [pH, pW, C] : [n, pH, pW, C]
  );
}

/**
 * 2D convolution. Input [N,H,W,C_in], kernel [kH,kW,C_in,C_out].
 */
export async function conv2d(
  ctx: AccelContext,
  input: GPUArray,
  kernel: GPUArray,
  stride = 1,
  padding = 0,
  h?: number,
  w?: number,
  cIn?: number
): Promise<GPUArray> {
  const { n, h: H, w: W, c: C_in } = get4DShape(input, h, w, cIn);
  const kShape = kernel.shape;
  if (kShape.length !== 4) throw new Error("conv2d: kernel must be 4D [kH,kW,C_in,C_out]");
  const [kH, kW, kCIn, C_out] = kShape;
  if (kCIn !== C_in) throw new Error(`conv2d: kernel C_in ${kCIn} != input C ${C_in}`);

  const outH = Math.floor((H + 2 * padding - kH) / stride) + 1;
  const outW = Math.floor((W + 2 * padding - kW) / stride) + 1;

  if (ctx.backendType === "webgpu") {
    await input.materialize();
    await kernel.materialize();
    const G = (globalThis as any).GPUBufferUsage;
    const usage = G.STORAGE | G.COPY_SRC | G.COPY_DST;
    const out = ctx.backend.createBuffer(n * outH * outW * C_out * 4, usage);
    await (ctx.runner as KernelRunner).conv2d(
      input.getBuffer() as GPUBuffer,
      kernel.getBuffer() as GPUBuffer,
      out as GPUBuffer,
      n,
      H,
      W,
      C_in,
      kH,
      kW,
      C_out,
      outH,
      outW,
      stride,
      padding
    );
    return new (await import("../array")).GPUArray(
      ctx.backend,
      ctx.runner,
      out,
      n * outH * outW * C_out,
      n === 1 ? [outH, outW, C_out] : [n, outH, outW, C_out]
    );
  }

  const inData = await input.toArray();
  const kData = await kernel.toArray();

  const out = new Float32Array(n * outH * outW * C_out);
  for (let ni = 0; ni < n; ni++) {
    for (let yi = 0; yi < outH; yi++) {
      for (let xi = 0; xi < outW; xi++) {
        for (let co = 0; co < C_out; co++) {
          let sum = 0;
          for (let ky = 0; ky < kH; ky++) {
            for (let kx = 0; kx < kW; kx++) {
              const y = yi * stride + ky - padding;
              const x = xi * stride + kx - padding;
              if (y >= 0 && y < H && x >= 0 && x < W) {
                for (let ci = 0; ci < C_in; ci++) {
                  sum +=
                    inData[ni * H * W * C_in + y * W * C_in + x * C_in + ci] *
                    kData[ky * kW * C_in * C_out + kx * C_in * C_out + ci * C_out + co];
                }
              }
            }
          }
          out[ni * outH * outW * C_out + yi * outW * C_out + xi * C_out + co] = sum;
        }
      }
    }
  }
  const G = (globalThis as any).GPUBufferUsage;
  const buffer = ctx.backend.createBufferFromData(out.buffer as ArrayBuffer, G.STORAGE | G.COPY_SRC | G.COPY_DST);
  return new (await import("../array")).GPUArray(
    ctx.backend,
    ctx.runner,
    buffer,
    out.length,
    n === 1 ? [outH, outW, C_out] : [n, outH, outW, C_out]
  );
}
