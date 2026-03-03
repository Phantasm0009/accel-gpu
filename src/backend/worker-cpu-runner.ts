/**
 * Worker-backed CPU runner (experimental).
 *
 * Uses a single Web Worker to execute heavier CPU ops off the main thread.
 * Falls back to base CPURunner behavior when Worker APIs are unavailable.
 */

import { CPURunner, type CPUBuffer } from "./cpu-runner";

type WorkerRequest = {
  id: number;
  op: "reduceSum" | "reduceMax" | "reduceMin" | "matmul" | "softmax" | "layerNorm" | "attentionScores";
  payload: Record<string, unknown>;
};

type WorkerResponse = {
  id: number;
  ok: true;
  result: Float32Array;
} | {
  id: number;
  ok: false;
  error: string;
};

const WORKER_SOURCE = `
self.onmessage = (event) => {
  const { id, op, payload } = event.data;
  try {
    let result;
    if (op === "reduceSum") {
      const input = payload.input;
      let sum = 0;
      for (let i = 0; i < input.length; i++) sum += input[i];
      result = new Float32Array([sum]);
    } else if (op === "reduceMax") {
      const input = payload.input;
      let max = -Infinity;
      for (let i = 0; i < input.length; i++) if (input[i] > max) max = input[i];
      result = new Float32Array([max]);
    } else if (op === "reduceMin") {
      const input = payload.input;
      let min = Infinity;
      for (let i = 0; i < input.length; i++) if (input[i] < min) min = input[i];
      result = new Float32Array([min]);
    } else if (op === "matmul") {
      const a = payload.a;
      const b = payload.b;
      const M = payload.M;
      const N = payload.N;
      const K = payload.K;
      const out = new Float32Array(M * N);
      for (let i = 0; i < M; i++) {
        for (let j = 0; j < N; j++) {
          let sum = 0;
          for (let k = 0; k < K; k++) sum += a[i * K + k] * b[k * N + j];
          out[i * N + j] = sum;
        }
      }
      result = out;
    } else if (op === "softmax") {
      const input = payload.input;
      const rows = payload.rows;
      const cols = payload.cols;
      const out = new Float32Array(rows * cols);
      for (let row = 0; row < rows; row++) {
        let maxVal = -Infinity;
        for (let c = 0; c < cols; c++) {
          const v = input[row * cols + c];
          if (v > maxVal) maxVal = v;
        }
        let sumExp = 0;
        for (let c = 0; c < cols; c++) {
          const e = Math.exp(input[row * cols + c] - maxVal);
          out[row * cols + c] = e;
          sumExp += e;
        }
        for (let c = 0; c < cols; c++) {
          out[row * cols + c] /= sumExp;
        }
      }
      result = out;
    } else if (op === "layerNorm") {
      const input = payload.input;
      const gamma = payload.gamma;
      const beta = payload.beta;
      const rows = payload.rows;
      const cols = payload.cols;
      const out = new Float32Array(rows * cols);
      const eps = 1e-5;
      for (let row = 0; row < rows; row++) {
        let sum = 0;
        for (let c = 0; c < cols; c++) sum += input[row * cols + c];
        const mean = sum / cols;
        let varSum = 0;
        for (let c = 0; c < cols; c++) {
          const d = input[row * cols + c] - mean;
          varSum += d * d;
        }
        const variance = Math.sqrt(varSum / cols + eps);
        for (let c = 0; c < cols; c++) {
          const normalized = (input[row * cols + c] - mean) / variance;
          out[row * cols + c] = normalized * gamma[c] + beta[c];
        }
      }
      result = out;
    } else if (op === "attentionScores") {
      const Q = payload.Q;
      const K = payload.K;
      const seq = payload.seq;
      const dim = payload.dim;
      const out = new Float32Array(seq * seq);
      const scale = 1 / Math.sqrt(dim);
      for (let i = 0; i < seq; i++) {
        for (let j = 0; j < seq; j++) {
          let score = 0;
          for (let d = 0; d < dim; d++) score += Q[i * dim + d] * K[j * dim + d];
          out[i * seq + j] = score * scale;
        }
      }
      result = out;
    } else {
      throw new Error("Unsupported op: " + op);
    }

    self.postMessage({ id, ok: true, result });
  } catch (error) {
    self.postMessage({ id, ok: false, error: String(error && error.message ? error.message : error) });
  }
};
`;

export class WorkerCPURunner extends CPURunner {
  readonly isWorkerEnabled: boolean;

  private worker: Worker | null;
  private nextId = 1;
  private pending = new Map<number, { resolve: (result: Float32Array) => void; reject: (error: unknown) => void }>();

  constructor() {
    super();
    this.worker = this.createWorker();
    this.isWorkerEnabled = this.worker !== null;
  }

  private createWorker(): Worker | null {
    if (typeof Worker === "undefined" || typeof Blob === "undefined" || typeof URL === "undefined") {
      return null;
    }

    const blob = new Blob([WORKER_SOURCE], { type: "application/javascript" });
    const workerUrl = URL.createObjectURL(blob);
    const worker = new Worker(workerUrl);
    URL.revokeObjectURL(workerUrl);

    worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
      const msg = event.data;
      const task = this.pending.get(msg.id);
      if (!task) return;
      this.pending.delete(msg.id);
      if (msg.ok) task.resolve(msg.result);
      else task.reject(new Error(msg.error));
    };

    worker.onerror = (event) => {
      const error = new Error(event.message || "Worker execution error");
      for (const [, task] of this.pending) task.reject(error);
      this.pending.clear();
    };

    return worker;
  }

  terminate(): void {
    if (this.worker) {
      this.worker.terminate();
      this.worker = null;
    }
    this.pending.clear();
  }

  private async runInWorker(
    op: WorkerRequest["op"],
    payload: Record<string, unknown>
  ): Promise<Float32Array> {
    if (!this.worker) throw new Error("WorkerCPURunner: worker is not available");
    const id = this.nextId++;
    const request: WorkerRequest = { id, op, payload };
    return new Promise<Float32Array>((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.worker!.postMessage(request);
    });
  }

  async reduceSum(input: CPUBuffer, output: CPUBuffer, length: number): Promise<void> {
    if (!this.worker) return super.reduceSum(input, output, length);
    const result = await this.runInWorker("reduceSum", {
      input: input.data.slice(0, length),
    });
    output.data[0] = result[0];
  }

  async reduceMax(input: CPUBuffer, output: CPUBuffer, length: number): Promise<void> {
    if (!this.worker) return super.reduceMax(input, output, length);
    const result = await this.runInWorker("reduceMax", {
      input: input.data.slice(0, length),
    });
    output.data[0] = result[0];
  }

  async reduceMin(input: CPUBuffer, output: CPUBuffer, length: number): Promise<void> {
    if (!this.worker) return super.reduceMin(input, output, length);
    const result = await this.runInWorker("reduceMin", {
      input: input.data.slice(0, length),
    });
    output.data[0] = result[0];
  }

  async matmul(
    a: CPUBuffer,
    b: CPUBuffer,
    out: CPUBuffer,
    M: number,
    N: number,
    K: number
  ): Promise<void> {
    if (!this.worker) return super.matmul(a, b, out, M, N, K);
    const result = await this.runInWorker("matmul", {
      a: a.data,
      b: b.data,
      M,
      N,
      K,
    });
    out.data.set(result);
  }

  async softmax(input: CPUBuffer, output: CPUBuffer, rows: number, cols: number): Promise<void> {
    if (!this.worker) return super.softmax(input, output, rows, cols);
    const result = await this.runInWorker("softmax", {
      input: input.data,
      rows,
      cols,
    });
    output.data.set(result);
  }

  async layerNorm(
    input: CPUBuffer,
    gamma: CPUBuffer,
    beta: CPUBuffer,
    output: CPUBuffer,
    rows: number,
    cols: number
  ): Promise<void> {
    if (!this.worker) return super.layerNorm(input, gamma, beta, output, rows, cols);
    const result = await this.runInWorker("layerNorm", {
      input: input.data,
      gamma: gamma.data,
      beta: beta.data,
      rows,
      cols,
    });
    output.data.set(result);
  }

  async attentionScores(
    Q: CPUBuffer,
    K: CPUBuffer,
    output: CPUBuffer,
    seq: number,
    dim: number
  ): Promise<void> {
    if (!this.worker) return super.attentionScores(Q, K, output, seq, dim);
    const result = await this.runInWorker("attentionScores", {
      Q: Q.data,
      K: K.data,
      seq,
      dim,
    });
    output.data.set(result);
  }
}
