/**
 * Quick smoke test - runs in Node with CPU backend
 */
import { init, matmul, softmax } from "../dist/index.js";

async function run() {
  const gpu = await init({ forceCPU: true });
  console.log("Backend:", gpu.backendType);

  // Add
  const a = gpu.array(new Float32Array([1, 2, 3, 4]));
  const b = gpu.array(new Float32Array([5, 6, 7, 8]));
  await a.add(b);
  const addResult = await a.toArray();
  if (addResult[0] !== 6 || addResult[3] !== 12) throw new Error("Add failed: " + addResult);

  // Sum
  const sum = await gpu.array(new Float32Array([1, 2, 3, 4])).sum();
  if (sum !== 10) throw new Error("Sum failed: " + sum);

  // Matmul
  const A = gpu.array(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]);
  const B = gpu.array(new Float32Array([1, 2, 3, 4, 5, 6]), [3, 2]);
  const C = await matmul(gpu, A, B);
  const matmulResult = await C.toArray();
  const expected = [22, 28, 49, 64];
  if (matmulResult.some((v, i) => Math.abs(v - expected[i]) > 1e-5))
    throw new Error("Matmul failed: " + matmulResult);

  // Softmax
  const logits = gpu.array(new Float32Array([1, 2, 3, 4]));
  const probs = await softmax(gpu, logits);
  const probsArr = await probs.toArray();
  const probsSum = probsArr.reduce((a, b) => a + b, 0);
  if (Math.abs(probsSum - 1) > 1e-5) throw new Error("Softmax sum != 1: " + probsSum);

  console.log("All tests passed!");
}

run().catch((e) => {
  console.error(e);
  process.exit(1);
});
