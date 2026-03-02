import { describe, it, expect, beforeAll } from "vitest";
import {
  init,
  matmul,
  softmax,
  transpose,
  layerNorm,
  inv,
  det,
  solve,
  fft,
  fftMagnitude,
  spectrogram,
  maxPool2d,
  batchNorm,
} from "../../dist/index.js";

describe("accel-gpu API", () => {
  let gpu: Awaited<ReturnType<typeof init>>;

  beforeAll(async () => {
    gpu = await init({ forceCPU: true });
  });

  it("add", async () => {
    const a = gpu.array(new Float32Array([1, 2, 3, 4]));
    const b = gpu.array(new Float32Array([5, 6, 7, 8]));
    await a.add(b);
    const result = await a.toArray();
    expect(result[0]).toBe(6);
    expect(result[3]).toBe(12);
  });

  it("mul", async () => {
    const m = gpu.array(new Float32Array([2, 4, 6, 8]));
    await m.mul(2);
    const result = await m.toArray();
    expect(result[0]).toBe(4);
    expect(result[3]).toBe(16);
  });

  it("sum", async () => {
    const sum = await gpu.array(new Float32Array([1, 2, 3, 4])).sum();
    expect(sum).toBe(10);
  });

  it("max", async () => {
    const maxVal = await gpu.array(new Float32Array([1, 5, 3, 2])).max();
    expect(maxVal).toBe(5);
  });

  it("dot", async () => {
    const v1 = gpu.array(new Float32Array([1, 2, 3]));
    const v2 = gpu.array(new Float32Array([4, 5, 6]));
    const dotVal = await v1.dot(v2);
    expect(dotVal).toBe(32);
  });

  it("matmul", async () => {
    const A = gpu.array(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]);
    const B = gpu.array(new Float32Array([1, 2, 3, 4, 5, 6]), [3, 2]);
    const C = await matmul(gpu, A, B);
    const result = await C.toArray();
    const expected = [22, 28, 49, 64];
    expected.forEach((exp, i) => expect(result[i]).toBeCloseTo(exp, 5));
  });

  it("transpose", async () => {
    const T = gpu.array(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]);
    const Tt = await transpose(gpu, T, 2, 3);
    const result = await Tt.toArray();
    const expected = [1, 4, 2, 5, 3, 6];
    expected.forEach((exp, i) => expect(result[i]).toBeCloseTo(exp, 5));
  });

  it("softmax", async () => {
    const logits = gpu.array(new Float32Array([1, 2, 3, 4]));
    const probs = await softmax(gpu, logits);
    const result = await probs.toArray();
    const sum = result.reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(1, 5);
  });

  it("sub", async () => {
    const a = gpu.array(new Float32Array([10, 20, 30]));
    await a.sub(5);
    const result = await a.toArray();
    expect(result[0]).toBe(5);
    expect(result[2]).toBe(25);
  });

  it("div", async () => {
    const a = gpu.array(new Float32Array([4, 8, 12]));
    await a.div(2);
    const result = await a.toArray();
    expect(result[0]).toBe(2);
    expect(result[2]).toBe(6);
  });

  it("sqrt", async () => {
    const a = gpu.array(new Float32Array([4, 9, 16]));
    await a.sqrt();
    const result = await a.toArray();
    expect(result[0]).toBe(2);
    expect(result[2]).toBe(4);
  });

  it("relu", async () => {
    const a = gpu.array(new Float32Array([-1, 2, -3, 4]));
    await a.relu();
    const result = await a.toArray();
    expect(result[0]).toBe(0);
    expect(result[1]).toBe(2);
    expect(result[3]).toBe(4);
  });

  it("min", async () => {
    const minVal = await gpu.array(new Float32Array([5, 2, 8, 1])).min();
    expect(minVal).toBe(1);
  });

  it("mean", async () => {
    const avg = await gpu.array(new Float32Array([2, 4, 6])).mean();
    expect(avg).toBe(4);
  });

  it("clone", async () => {
    const a = gpu.array(new Float32Array([1, 2, 3]));
    const b = await a.clone();
    await a.add(10);
    const aData = await a.toArray();
    const bData = await b.toArray();
    expect(aData[0]).toBe(11);
    expect(bData[0]).toBe(1);
  });

  it("zeros, ones, arange, linspace", async () => {
    const z = gpu.zeros([3]);
    expect((await z.toArray())[0]).toBe(0);
    const o = gpu.ones([2, 2]);
    expect((await o.toArray())[0]).toBe(1);
    const r = gpu.arange(0, 5, 1);
    const rData = await r.toArray();
    expect(rData[0]).toBe(0);
    expect(rData[4]).toBe(4);
    const l = gpu.linspace(0, 1, 3);
    const lData = await l.toArray();
    expect(lData[0]).toBe(0);
    expect(lData[2]).toBeCloseTo(1, 5);
  });

  it("layerNorm", async () => {
    const lnInput = gpu.array(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]);
    const gamma = gpu.array(new Float32Array([1, 1, 1]));
    const beta = gpu.array(new Float32Array([0, 0, 0]));
    const lnOut = await layerNorm(gpu, lnInput, gamma, beta, 2, 3);
    const result = await lnOut.toArray();
    expect(result.length).toBe(6);
    const row0Sum = result.slice(0, 3).reduce((a, b) => a + b, 0);
    const row1Sum = result.slice(3, 6).reduce((a, b) => a + b, 0);
    expect(row0Sum).toBeCloseTo(0, 5);
    expect(row1Sum).toBeCloseTo(0, 5);
  });

  it("variance, std, argmax, argmin", async () => {
    const a = gpu.array(new Float32Array([1, 2, 3, 4, 5]));
    expect(await a.variance()).toBeCloseTo(2, 4);
    expect(await a.std()).toBeCloseTo(Math.sqrt(2), 4);
    expect(await a.argmax()).toBe(4);
    expect(await a.argmin()).toBe(0);
  });

  it("equal, greater, less", async () => {
    const a = gpu.array(new Float32Array([1, 2, 3]));
    const b = gpu.array(new Float32Array([1, 0, 3]));
    const eq = await a.equal(b);
    expect(await eq.toArray()).toEqual(new Float32Array([1, 0, 1]));
    const gt = await a.greater(b);
    expect(await gt.toArray()).toEqual(new Float32Array([0, 1, 0]));
    const lt = await a.less(b);
    expect(await lt.toArray()).toEqual(new Float32Array([0, 0, 0]));
  });

  it("clamp, gelu, leakyRelu", async () => {
    const a = gpu.array(new Float32Array([-1, 0, 1, 2, 3]));
    await a.clamp(0, 2);
    const clamped = await a.toArray();
    expect(clamped[0]).toBe(0);
    expect(clamped[4]).toBe(2);
    const b = gpu.array(new Float32Array([0, 1]));
    await b.gelu();
    const geluOut = await b.toArray();
    expect(geluOut[0]).toBeCloseTo(0, 4);
    const c = gpu.array(new Float32Array([-2, 2]));
    await c.leakyRelu(0.01);
    const lr = await c.toArray();
    expect(lr[0]).toBeCloseTo(-0.02, 4);
    expect(lr[1]).toBe(2);
  });

  it("slice, get, set, concat, split", async () => {
    const a = gpu.array(new Float32Array([1, 2, 3, 4, 5]));
    const s = await a.slice(1, 4);
    expect(await s.toArray()).toEqual(new Float32Array([2, 3, 4]));
    expect(await a.get(2)).toBe(3);
    await a.set(2, 99);
    expect(await a.get(2)).toBe(99);
    const b = gpu.array(new Float32Array([10, 20]));
    const cat = await a.concat(b);
    expect((await cat.toArray()).length).toBe(7);
    const parts = await cat.split(1);
    expect(parts.length).toBe(1);
    expect(parts[0].length).toBe(7);
  });

  it("flatten, squeeze, unsqueeze", async () => {
    const a = gpu.array(new Float32Array([1, 2, 3, 4]), [2, 2]);
    a.flatten();
    expect(a.shape).toEqual([4]);
    const b = gpu.array(new Float32Array([1]), [1, 1]);
    b.squeeze();
    expect(b.shape).toEqual([1]);
    a.unsqueeze(0);
    expect(a.shape).toEqual([1, 4]);
  });

  it("dispose, isDisposed, toArraySync", async () => {
    const a = gpu.array(new Float32Array([1, 2, 3]));
    expect(a.isDisposed).toBe(false);
    const sync = a.toArraySync();
    expect(sync[0]).toBe(1);
    a.dispose();
    expect(a.isDisposed).toBe(true);
    await expect(a.toArray()).rejects.toThrow();
  });

  it("broadcast, sum(axis), mean(axis), max(axis)", async () => {
    const a = gpu.array(new Float32Array([1, 2, 3]), [1, 3]);
    const b = await a.broadcast([2, 3]);
    const bData = await b.toArray();
    expect(bData[0]).toBe(1);
    expect(bData[5]).toBe(3);
    const m = gpu.array(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]);
    const s0 = (await m.sum(0)) as import("../../dist/index.js").GPUArray;
    expect(s0.length).toBe(3);
    const s0Data = await s0.toArray();
    expect(s0Data[0]).toBe(5);
    const mean1 = (await m.mean(1)) as import("../../dist/index.js").GPUArray;
    const mean1Data = await mean1.toArray();
    expect(mean1Data[0]).toBeCloseTo(2, 4);
  });

  it("inv, det, solve", async () => {
    const a = gpu.array(new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1]), [3, 3]);
    const aInv = await inv(gpu, a);
    const invData = await aInv.toArray();
    expect(invData[0]).toBeCloseTo(1, 4);
    expect(await det(gpu, a)).toBeCloseTo(1, 4);
    const b = gpu.array(new Float32Array([1, 2, 3]));
    const x = await solve(gpu, a, b);
    const xData = await x.toArray();
    expect(xData[0]).toBeCloseTo(1, 4);
  });

  it("fft, spectrogram", async () => {
    const signal = gpu.array(new Float32Array(8).map((_, i) => Math.sin((2 * Math.PI * i) / 8)));
    const freq = await fft(gpu, signal);
    const mag = await fftMagnitude(gpu, freq);
    const magData = await mag.toArray();
    expect(magData.length).toBe(8);
    const spec = await spectrogram(gpu, signal, 4, 2);
    const specData = await spec.toArray();
    expect(specData.length).toBeGreaterThan(0);
  });

  it("maxPool2d, batchNorm", async () => {
    const img = gpu.array(new Float32Array(16).fill(1), [4, 4, 1]);
    const pooled = await maxPool2d(gpu, img, 2, 2);
    expect(pooled.length).toBe(4);
    const ln = gpu.array(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]);
    const gamma = gpu.array(new Float32Array([1, 1, 1]));
    const beta = gpu.array(new Float32Array([0, 0, 0]));
    const bn = await batchNorm(gpu, ln, gamma, beta);
    expect(bn.length).toBe(6);
  });

  it("norm, outer, mse, crossEntropy", async () => {
    const a = gpu.array(new Float32Array([3, 4]));
    expect(await a.norm(2)).toBe(5);
    expect(await a.norm(1)).toBe(7);
    const u = gpu.array(new Float32Array([1, 2]));
    const v = gpu.array(new Float32Array([3, 4]));
    const outer = await u.outer(v);
    const o = await outer.toArray();
    expect(o[0]).toBe(3);
    expect(o[1]).toBe(4);
    expect(o[2]).toBe(6);
    expect(o[3]).toBe(8);
    const pred = gpu.array(new Float32Array([1, 2, 3]));
    const target = gpu.array(new Float32Array([1, 2, 3]));
    expect(await pred.mse(target)).toBe(0);
    const probs = gpu.array(new Float32Array([0.9, 0.1]));
    const oneHot = gpu.array(new Float32Array([1, 0]));
    const ce = await probs.crossEntropy(oneHot);
    expect(ce).toBeGreaterThan(0);
  });
});
