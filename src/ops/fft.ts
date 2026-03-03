/**
 * FFT and spectrogram - complex-number support and butterfly operations
 * Complex numbers stored as interleaved [real0, imag0, real1, imag1, ...]
 */

import type { GPUArray } from "../array";
import type { AccelContext } from "../types";
import type { KernelRunner } from "../backend/kernel-runner";

/** In-place Cooley-Tukey FFT. Input: interleaved real/imag, length must be power of 2. */
function fft1dCPU(data: Float32Array, inverse = false): void {
  const n = data.length / 2;
  if ((n & (n - 1)) !== 0) throw new Error(`fft: length must be power of 2, got ${n}`);

  // Bit-reversal permutation
  const bits = Math.log2(n);
  for (let i = 0; i < n; i++) {
    let rev = 0;
    for (let k = 0; k < bits; k++) {
      rev = (rev << 1) | ((i >> k) & 1);
    }
    if (i < rev) {
      const tr = data[2 * i];
      const ti = data[2 * i + 1];
      data[2 * i] = data[2 * rev];
      data[2 * i + 1] = data[2 * rev + 1];
      data[2 * rev] = tr;
      data[2 * rev + 1] = ti;
    }
  }

  const sign = inverse ? 1 : -1;
  for (let len = 2; len <= n; len *= 2) {
    const angle = (sign * 2 * Math.PI) / len;
    const wlenR = Math.cos(angle);
    const wlenI = Math.sin(angle);
    for (let i = 0; i < n; i += len) {
      let wR = 1;
      let wI = 0;
      for (let j = 0; j < len / 2; j++) {
        const u = 2 * (i + j);
        const v = 2 * (i + j + len / 2);
        const tR = wR * data[v] - wI * data[v + 1];
        const tI = wR * data[v + 1] + wI * data[v];
        data[v] = data[u] - tR;
        data[v + 1] = data[u + 1] - tI;
        data[u] += tR;
        data[u + 1] += tI;
        const nwR = wR * wlenR - wI * wlenI;
        const nwI = wR * wlenI + wI * wlenR;
        wR = nwR;
        wI = nwI;
      }
    }
  }

  if (inverse) {
    for (let i = 0; i < data.length; i++) data[i] /= n;
  }
}

/**
 * FFT of a real-valued signal. Input: Float32Array of length N (power of 2).
 * Returns complex result as interleaved [real0, imag0, ...] in a new GPUArray.
 */
export async function fft(
  ctx: AccelContext,
  input: GPUArray,
  inverse = false
): Promise<GPUArray> {
  const n = input.length;
  if ((n & (n - 1)) !== 0) {
    throw new Error(`fft: length must be power of 2, got ${n}`);
  }
  if (ctx.backendType === "webgpu") {
    await input.materialize();
    const G = (globalThis as any).GPUBufferUsage;
    const usage = G.STORAGE | G.COPY_SRC | G.COPY_DST;
    const out = ctx.backend.createBuffer(2 * n * 4, usage);
    await (ctx.runner as KernelRunner).fftReal(
      input.getBuffer() as GPUBuffer,
      out as GPUBuffer,
      n,
      inverse
    );
    return new (await import("../array")).GPUArray(ctx.backend, ctx.runner, out, 2 * n, [n, 2]);
  }

  const data = await input.toArray();
  const complex = new Float32Array(2 * n);
  for (let i = 0; i < n; i++) {
    complex[2 * i] = data[i];
    complex[2 * i + 1] = 0;
  }
  fft1dCPU(complex, inverse);
  const G = (globalThis as any).GPUBufferUsage;
  const usage = G.STORAGE | G.COPY_SRC | G.COPY_DST;
  const buffer = ctx.backend.createBufferFromData(complex.buffer as ArrayBuffer, usage);
  return new (await import("../array")).GPUArray(
    ctx.backend,
    ctx.runner,
    buffer,
    2 * n,
    [n, 2]
  );
}

/**
 * Inverse FFT of complex data (interleaved real/imag).
 * Input: GPUArray of length 2*N (N complex numbers).
 */
export async function ifft(ctx: AccelContext, input: GPUArray): Promise<GPUArray> {
  if (input.length % 2 !== 0) {
    throw new Error(`ifft: input length must be even (2*N complex), got ${input.length}`);
  }
  if (ctx.backendType === "webgpu") {
    await input.materialize();
    const G = (globalThis as any).GPUBufferUsage;
    const usage = G.STORAGE | G.COPY_SRC | G.COPY_DST;
    const out = ctx.backend.createBuffer(input.length * 4, usage);
    await (ctx.runner as KernelRunner).ifftComplex(
      input.getBuffer() as GPUBuffer,
      out as GPUBuffer,
      input.length / 2
    );
    return new (await import("../array")).GPUArray(
      ctx.backend,
      ctx.runner,
      out,
      input.length,
      [input.length / 2, 2]
    );
  }

  const data = await input.toArray();
  const complex = new Float32Array(data);
  fft1dCPU(complex, true);
  const G = (globalThis as any).GPUBufferUsage;
  const usage = G.STORAGE | G.COPY_SRC | G.COPY_DST;
  const buffer = ctx.backend.createBufferFromData(complex.buffer as ArrayBuffer, usage);
  return new (await import("../array")).GPUArray(
    ctx.backend,
    ctx.runner,
    buffer,
    complex.length,
    [complex.length / 2, 2]
  );
}

/**
 * Magnitude spectrum from complex FFT output.
 * Input: interleaved [real, imag, ...] from fft().
 * Output: real-valued magnitudes.
 */
export async function fftMagnitude(ctx: AccelContext, complex: GPUArray): Promise<GPUArray> {
  if (ctx.backendType === "webgpu") {
    await complex.materialize();
    const n = complex.length / 2;
    const G = (globalThis as any).GPUBufferUsage;
    const usage = G.STORAGE | G.COPY_SRC | G.COPY_DST;
    const out = ctx.backend.createBuffer(n * 4, usage);
    await (ctx.runner as KernelRunner).fftMagnitude(
      complex.getBuffer() as GPUBuffer,
      out as GPUBuffer,
      n
    );
    return new (await import("../array")).GPUArray(ctx.backend, ctx.runner, out, n, [n]);
  }

  const data = await complex.toArray();
  const n = data.length / 2;
  const mag = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const r = data[2 * i];
    const im = data[2 * i + 1];
    mag[i] = Math.sqrt(r * r + im * im);
  }
  const G = (globalThis as any).GPUBufferUsage;
  const usage = G.STORAGE | G.COPY_SRC | G.COPY_DST;
  const buffer = ctx.backend.createBufferFromData(mag.buffer as ArrayBuffer, usage);
  return new (await import("../array")).GPUArray(ctx.backend, ctx.runner, buffer, n, [n]);
}

/**
 * Short-time Fourier transform (STFT) + magnitude = spectrogram.
 * @param input - 1D signal array
 * @param frameLength - FFT size (power of 2)
 * @param hopLength - hop between frames (default frameLength/2)
 * @param window - 'hann' | 'hamming' | 'rect'
 * @returns Spectrogram as GPUArray [numFrames, frameLength/2+1] (real magnitudes)
 */
export async function spectrogram(
  ctx: AccelContext,
  input: GPUArray,
  frameLength: number,
  hopLength?: number,
  window: "hann" | "hamming" | "rect" = "hann"
): Promise<GPUArray> {
  const n = input.length;
  const hop = hopLength ?? Math.floor(frameLength / 2);
  if ((frameLength & (frameLength - 1)) !== 0) {
    throw new Error(`spectrogram: frameLength must be power of 2, got ${frameLength}`);
  }

  const numFrames = Math.floor((n - frameLength) / hop) + 1;
  const fftBins = Math.floor(frameLength / 2) + 1;

  const data = await input.toArray();

  // Build window
  const win = new Float32Array(frameLength);
  for (let i = 0; i < frameLength; i++) {
    if (window === "hann") {
      win[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (frameLength - 1)));
    } else if (window === "hamming") {
      win[i] = 0.54 - 0.46 * Math.cos((2 * Math.PI * i) / (frameLength - 1));
    } else {
      win[i] = 1;
    }
  }

  const spec = new Float32Array(numFrames * fftBins);
  const frame = new Float32Array(2 * frameLength);

  for (let f = 0; f < numFrames; f++) {
    const start = f * hop;
    for (let i = 0; i < frameLength; i++) {
      frame[2 * i] = (data[start + i] ?? 0) * win[i];
      frame[2 * i + 1] = 0;
    }
    fft1dCPU(frame, false);
    for (let k = 0; k < fftBins; k++) {
      const r = frame[2 * k];
      const im = frame[2 * k + 1];
      spec[f * fftBins + k] = Math.sqrt(r * r + im * im);
    }
  }

  const G = (globalThis as any).GPUBufferUsage;
  const usage = G.STORAGE | G.COPY_SRC | G.COPY_DST;
  const buffer = ctx.backend.createBufferFromData(spec.buffer as ArrayBuffer, usage);
  return new (await import("../array")).GPUArray(
    ctx.backend,
    ctx.runner,
    buffer,
    spec.length,
    [numFrames, fftBins]
  );
}
