/**
 * Backend abstraction - WebGPU, WebGL, and CPU implement this
 */

export type BackendType = "webgpu" | "webgl" | "cpu";

export interface ComputeBuffer {
  /** WebGPU: GPUBuffer. CPU: Float32Array. WebGL: WebGLBuffer or texture. */
  _storage: unknown;
  size: number;
  length: number;
}

export interface IBackend {
  readonly type: BackendType;
  createBuffer(size: number, usage: number): ComputeBuffer;
  createBufferFromData(data: ArrayBuffer, usage: number): ComputeBuffer;
  writeBuffer(buffer: ComputeBuffer, data: ArrayBuffer): Promise<void>;
  readBuffer(buffer: ComputeBuffer, output: ArrayBuffer): Promise<void>;
  destroyBuffer(buffer: ComputeBuffer): void;
  /** For WebGPU: device. For CPU/WebGL: undefined */
  device?: GPUDevice;
  /** For WebGPU: queue. For CPU/WebGL: undefined */
  queue?: GPUQueue;
}

export interface IRunner {
  add(a: ComputeBuffer, b: ComputeBuffer, out: ComputeBuffer, length: number): Promise<void>;
  mul(a: ComputeBuffer, b: ComputeBuffer, out: ComputeBuffer, length: number): Promise<void>;
  mulScalar(a: ComputeBuffer, scalar: number, out: ComputeBuffer, length: number): Promise<void>;
  reduceSum(input: ComputeBuffer, output: ComputeBuffer, length: number): Promise<void>;
  reduceMax(input: ComputeBuffer, output: ComputeBuffer, length: number): Promise<void>;
  matmul(
    a: ComputeBuffer,
    b: ComputeBuffer,
    out: ComputeBuffer,
    M: number,
    N: number,
    K: number
  ): Promise<void>;
  softmax(input: ComputeBuffer, output: ComputeBuffer, rows: number, cols: number): Promise<void>;
  layerNorm(
    input: ComputeBuffer,
    gamma: ComputeBuffer,
    beta: ComputeBuffer,
    output: ComputeBuffer,
    rows: number,
    cols: number
  ): Promise<void>;
  attention(
    Q: ComputeBuffer,
    K: ComputeBuffer,
    V: ComputeBuffer,
    output: ComputeBuffer,
    batch: number,
    heads: number,
    seq: number,
    dim: number
  ): Promise<void>;
}
