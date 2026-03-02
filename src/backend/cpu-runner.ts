/**
 * CPU fallback - implements same ops in pure JavaScript
 */

export interface CPUBuffer {
  data: Float32Array;
}

export class CPURunner {
  async add(a: CPUBuffer, b: CPUBuffer, out: CPUBuffer, length: number): Promise<void> {
    for (let i = 0; i < length; i++) out.data[i] = a.data[i] + b.data[i];
  }

  async mul(a: CPUBuffer, b: CPUBuffer, out: CPUBuffer, length: number): Promise<void> {
    for (let i = 0; i < length; i++) out.data[i] = a.data[i] * b.data[i];
  }

  async mulScalar(a: CPUBuffer, scalar: number, out: CPUBuffer, length: number): Promise<void> {
    for (let i = 0; i < length; i++) out.data[i] = a.data[i] * scalar;
  }

  async reduceSum(input: CPUBuffer, output: CPUBuffer, length: number): Promise<void> {
    let sum = 0;
    for (let i = 0; i < length; i++) sum += input.data[i];
    output.data[0] = sum;
  }

  async reduceMax(input: CPUBuffer, output: CPUBuffer, length: number): Promise<void> {
    let max = -Infinity;
    for (let i = 0; i < length; i++) if (input.data[i] > max) max = input.data[i];
    output.data[0] = max;
  }

  async matmul(
    a: CPUBuffer,
    b: CPUBuffer,
    out: CPUBuffer,
    M: number,
    N: number,
    K: number
  ): Promise<void> {
    for (let i = 0; i < M; i++) {
      for (let j = 0; j < N; j++) {
        let sum = 0;
        for (let k = 0; k < K; k++) sum += a.data[i * K + k] * b.data[k * N + j];
        out.data[i * N + j] = sum;
      }
    }
  }

  async softmax(input: CPUBuffer, output: CPUBuffer, rows: number, cols: number): Promise<void> {
    for (let row = 0; row < rows; row++) {
      let maxVal = -Infinity;
      for (let c = 0; c < cols; c++) {
        const v = input.data[row * cols + c];
        if (v > maxVal) maxVal = v;
      }
      let sumExp = 0;
      for (let c = 0; c < cols; c++) {
        const e = Math.exp(input.data[row * cols + c] - maxVal);
        output.data[row * cols + c] = e;
        sumExp += e;
      }
      for (let c = 0; c < cols; c++) {
        output.data[row * cols + c] /= sumExp;
      }
    }
  }

  async layerNorm(
    input: CPUBuffer,
    gamma: CPUBuffer,
    beta: CPUBuffer,
    output: CPUBuffer,
    rows: number,
    cols: number
  ): Promise<void> {
    const eps = 1e-5;
    for (let row = 0; row < rows; row++) {
      let sum = 0;
      for (let c = 0; c < cols; c++) sum += input.data[row * cols + c];
      const mean = sum / cols;
      let varSum = 0;
      for (let c = 0; c < cols; c++) {
        const d = input.data[row * cols + c] - mean;
        varSum += d * d;
      }
      const variance = Math.sqrt(varSum / cols + eps);
      for (let c = 0; c < cols; c++) {
        const normalized = (input.data[row * cols + c] - mean) / variance;
        output.data[row * cols + c] = normalized * gamma.data[c] + beta.data[c];
      }
    }
  }

  async attentionScores(
    Q: CPUBuffer,
    K: CPUBuffer,
    output: CPUBuffer,
    seq: number,
    dim: number
  ): Promise<void> {
    const scale = 1 / Math.sqrt(dim);
    for (let i = 0; i < seq; i++) {
      for (let j = 0; j < seq; j++) {
        let score = 0;
        for (let d = 0; d < dim; d++) score += Q.data[i * dim + d] * K.data[j * dim + d];
        output.data[i * seq + j] = score * scale;
      }
    }
  }
}
