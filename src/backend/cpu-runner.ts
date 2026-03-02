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

  async sub(a: CPUBuffer, b: CPUBuffer, out: CPUBuffer, length: number): Promise<void> {
    for (let i = 0; i < length; i++) out.data[i] = a.data[i] - b.data[i];
  }

  async subScalar(a: CPUBuffer, scalar: number, out: CPUBuffer, length: number): Promise<void> {
    for (let i = 0; i < length; i++) out.data[i] = a.data[i] - scalar;
  }

  async div(a: CPUBuffer, b: CPUBuffer, out: CPUBuffer, length: number): Promise<void> {
    for (let i = 0; i < length; i++) out.data[i] = b.data[i] !== 0 ? a.data[i] / b.data[i] : 0;
  }

  async divScalar(a: CPUBuffer, scalar: number, out: CPUBuffer, length: number): Promise<void> {
    const div = scalar !== 0 ? 1 / scalar : 0;
    for (let i = 0; i < length; i++) out.data[i] = a.data[i] * div;
  }

  async powScalar(a: CPUBuffer, exponent: number, out: CPUBuffer, length: number): Promise<void> {
    for (let i = 0; i < length; i++) out.data[i] = Math.pow(a.data[i], exponent);
  }

  async sqrt(a: CPUBuffer, out: CPUBuffer, length: number): Promise<void> {
    for (let i = 0; i < length; i++) out.data[i] = Math.sqrt(a.data[i]);
  }

  async abs(a: CPUBuffer, out: CPUBuffer, length: number): Promise<void> {
    for (let i = 0; i < length; i++) out.data[i] = Math.abs(a.data[i]);
  }

  async neg(a: CPUBuffer, out: CPUBuffer, length: number): Promise<void> {
    for (let i = 0; i < length; i++) out.data[i] = -a.data[i];
  }

  async exp(a: CPUBuffer, out: CPUBuffer, length: number): Promise<void> {
    for (let i = 0; i < length; i++) out.data[i] = Math.exp(a.data[i]);
  }

  async log(a: CPUBuffer, out: CPUBuffer, length: number): Promise<void> {
    for (let i = 0; i < length; i++) out.data[i] = a.data[i] > 0 ? Math.log(a.data[i]) : -1e38;
  }

  async relu(a: CPUBuffer, out: CPUBuffer, length: number): Promise<void> {
    for (let i = 0; i < length; i++) out.data[i] = Math.max(0, a.data[i]);
  }

  async sigmoid(a: CPUBuffer, out: CPUBuffer, length: number): Promise<void> {
    for (let i = 0; i < length; i++) out.data[i] = 1 / (1 + Math.exp(-a.data[i]));
  }

  async tanh(a: CPUBuffer, out: CPUBuffer, length: number): Promise<void> {
    for (let i = 0; i < length; i++) out.data[i] = Math.tanh(a.data[i]);
  }

  async clamp(
    a: CPUBuffer,
    minVal: number,
    maxVal: number,
    out: CPUBuffer,
    length: number
  ): Promise<void> {
    for (let i = 0; i < length; i++)
      out.data[i] = Math.max(minVal, Math.min(maxVal, a.data[i]));
  }

  async gelu(a: CPUBuffer, out: CPUBuffer, length: number): Promise<void> {
    const C = 0.044715;
    const SQRT_2_OVER_PI = 0.7978845608;
    for (let i = 0; i < length; i++) {
      const x = a.data[i];
      out.data[i] =
        0.5 * x * (1 + Math.tanh(SQRT_2_OVER_PI * (x + C * x * x * x)));
    }
  }

  async leakyRelu(
    a: CPUBuffer,
    alpha: number,
    out: CPUBuffer,
    length: number
  ): Promise<void> {
    for (let i = 0; i < length; i++)
      out.data[i] = a.data[i] >= 0 ? a.data[i] : alpha * a.data[i];
  }

  async equal(a: CPUBuffer, b: CPUBuffer, out: CPUBuffer, length: number): Promise<void> {
    for (let i = 0; i < length; i++) out.data[i] = a.data[i] === b.data[i] ? 1 : 0;
  }

  async greater(a: CPUBuffer, b: CPUBuffer, out: CPUBuffer, length: number): Promise<void> {
    for (let i = 0; i < length; i++) out.data[i] = a.data[i] > b.data[i] ? 1 : 0;
  }

  async less(a: CPUBuffer, b: CPUBuffer, out: CPUBuffer, length: number): Promise<void> {
    for (let i = 0; i < length; i++) out.data[i] = a.data[i] < b.data[i] ? 1 : 0;
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

  async reduceMin(input: CPUBuffer, output: CPUBuffer, length: number): Promise<void> {
    let min = Infinity;
    for (let i = 0; i < length; i++) if (input.data[i] < min) min = input.data[i];
    output.data[0] = min;
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
