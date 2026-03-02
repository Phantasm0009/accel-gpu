/**
 * Smoke test - runs in Node with CPU backend
 */
import { init, matmul, softmax, transpose, dot, layerNorm } from "../dist/index.js";

async function run() {
  const gpu = await init({ forceCPU: true });
  console.log("Backend:", gpu.backendType);

  // Add
  const a = gpu.array(new Float32Array([1, 2, 3, 4]));
  const b = gpu.array(new Float32Array([5, 6, 7, 8]));
  await a.add(b);
  const addResult = await a.toArray();
  if (addResult[0] !== 6 || addResult[3] !== 12) throw new Error("Add failed: " + addResult);

  // Mul
  const m = gpu.array(new Float32Array([2, 4, 6, 8]));
  await m.mul(2);
  const mulResult = await m.toArray();
  if (mulResult[0] !== 4 || mulResult[3] !== 16) throw new Error("Mul failed: " + mulResult);

  // Sum
  const sum = await gpu.array(new Float32Array([1, 2, 3, 4])).sum();
  if (sum !== 10) throw new Error("Sum failed: " + sum);

  // Max
  const maxVal = await gpu.array(new Float32Array([1, 5, 3, 2])).max();
  if (maxVal !== 5) throw new Error("Max failed: " + maxVal);

  // Dot
  const v1 = gpu.array(new Float32Array([1, 2, 3]));
  const v2 = gpu.array(new Float32Array([4, 5, 6]));
  const dotVal = await v1.dot(v2);
  if (dotVal !== 32) throw new Error("Dot failed: " + dotVal);

  // Matmul
  const A = gpu.array(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]);
  const B = gpu.array(new Float32Array([1, 2, 3, 4, 5, 6]), [3, 2]);
  const C = await matmul(gpu, A, B);
  const matmulResult = await C.toArray();
  const expected = [22, 28, 49, 64];
  if (matmulResult.some((v, i) => Math.abs(v - expected[i]) > 1e-5))
    throw new Error("Matmul failed: " + matmulResult);

  // Transpose
  const T = gpu.array(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]);
  const Tt = await transpose(gpu, T, 2, 3);
  const transResult = await Tt.toArray();
  const transExpected = [1, 4, 2, 5, 3, 6];
  if (transResult.some((v, i) => Math.abs(v - transExpected[i]) > 1e-5))
    throw new Error("Transpose failed: " + transResult);

  // Softmax
  const logits = gpu.array(new Float32Array([1, 2, 3, 4]));
  const probs = await softmax(gpu, logits);
  const probsArr = await probs.toArray();
  const probsSum = probsArr.reduce((a, b) => a + b, 0);
  if (Math.abs(probsSum - 1) > 1e-5) throw new Error("Softmax sum != 1: " + probsSum);

  // LayerNorm (identity: gamma=1, beta=0)
  const lnInput = gpu.array(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]);
  const gamma = gpu.array(new Float32Array([1, 1, 1]));
  const beta = gpu.array(new Float32Array([0, 0, 0]));
  const lnOut = await layerNorm(gpu, lnInput, gamma, beta, 2, 3);
  const lnArr = await lnOut.toArray();
  if (lnArr.length !== 6) throw new Error("LayerNorm length failed: " + lnArr.length);
  // Layer norm: per-row mean=2, std=1; per-row normalized. Row0: [-1,0,1], Row1: [-1,0,1]
  const lnRow0 = lnArr.slice(0, 3);
  const lnRow1 = lnArr.slice(3, 6);
  if (Math.abs(lnRow0.reduce((a, b) => a + b, 0)) > 1e-5)
    throw new Error("LayerNorm row0 sum != 0: " + lnRow0);
  if (Math.abs(lnRow1.reduce((a, b) => a + b, 0)) > 1e-5)
    throw new Error("LayerNorm row1 sum != 0: " + lnRow1);

  console.log("All tests passed!");
}

run().catch((e) => {
  console.error(e);
  process.exit(1);
});
