/**
 * GPUArray - GPU/CPU-backed array with NumPy-like API
 */

import type { WebGPUBackend } from "./backend/webgpu";
import type { WebGLBackend } from "./backend/webgl-backend";
import type { CPUBackend } from "./backend/cpu-backend";
import type { KernelRunner } from "./backend/kernel-runner";
import type { WebGLRunner } from "./backend/webgl-runner";
import type { CPURunner } from "./backend/cpu-runner";
import { errLengthMismatch } from "./errors";

const WORKGROUP_SIZE = 256;

type Backend = WebGPUBackend | WebGLBackend | CPUBackend;
type Runner = KernelRunner | WebGLRunner | CPURunner;
type Buffer = GPUBuffer | import("./backend/webgl-backend").WebGLBuffer | { data: Float32Array };

function isWebGPU(backend: Backend): backend is WebGPUBackend {
  return "device" in backend && backend.device !== undefined;
}

function isWebGL(backend: Backend): backend is WebGLBackend {
  return "type" in backend && backend.type === "webgl";
}

/**
 * GPU-backed (or CPU-backed) array with a NumPy-like API.
 *
 * Supports element-wise ops (add, mul), reductions (sum, max), dot product,
 * and reshape. Data lives on the selected backend (WebGPU, WebGL2, or CPU).
 */
export class GPUArray {
  private backend: Backend;
  private runner: Runner;
  private buffer: Buffer;
  readonly length: number;
  readonly byteLength: number;
  private _shape: number[];

  /** @internal */
  constructor(backend: Backend, runner: Runner, buffer: Buffer, length: number, shape?: number[]) {
    this.backend = backend;
    this.runner = runner;
    this.buffer = buffer;
    this.length = length;
    this.byteLength = length * 4;
    this._shape = shape ?? [length];
  }

  /** Current shape of the array (e.g. [2, 3] for a 2×3 matrix). */
  get shape(): number[] {
    return [...this._shape];
  }

  /**
   * Reshape the array in-place. Total elements must remain the same.
   * @param dims - New dimensions (e.g. 2, 3 for a 2×3 matrix)
   * @returns this for chaining
   */
  reshape(...dims: number[]): GPUArray {
    const total = dims.reduce((a, b) => a * b, 1);
    if (total !== this.length) {
      throw new Error(
        `reshape: cannot reshape [${this._shape.join(", ")}] (${this.length} elements) to [${dims.join(", ")}] (${total} elements).`
      );
    }
    this._shape = dims;
    return this;
  }

  /**
   * Copy data from the backend to a Float32Array.
   * @returns Promise resolving to the array data
   */
  async toArray(): Promise<Float32Array> {
    this._ensureNotDisposed();
    const result = new Float32Array(this.length);
    if (isWebGPU(this.backend)) {
      await this.backend.readBuffer(this.buffer as GPUBuffer, result.buffer);
    } else if (isWebGL(this.backend)) {
      await this.backend.readBuffer(
        this.buffer as import("./backend/webgl-backend").WebGLBuffer,
        result.buffer
      );
    } else {
      (this.backend as CPUBackend).readBuffer(this.buffer as { data: Float32Array }, result.buffer);
    }
    return result;
  }

  private _disposed = false;

  private _ensureNotDisposed(): void {
    if (this._disposed) throw new Error("GPUArray has been disposed");
  }

  /** Whether this array has been disposed. */
  get isDisposed(): boolean {
    return this._disposed;
  }

  /** Release GPU/CPU resources. No-op if already disposed. */
  dispose(): void {
    if (this._disposed) return;
    if (isWebGPU(this.backend)) {
      this.backend.destroyBuffer(this.buffer as GPUBuffer);
    } else if (isWebGL(this.backend)) {
      (this.backend as WebGLBackend).destroyBuffer(
        this.buffer as import("./backend/webgl-backend").WebGLBuffer
      );
    }
    this._disposed = true;
  }

  private async runUnaryOp(
    op:
      | "sqrt"
      | "abs"
      | "neg"
      | "exp"
      | "log"
      | "relu"
      | "sigmoid"
      | "tanh"
      | "gelu"
      | "pow",
    scalarOrExponent?: number
  ): Promise<GPUArray> {
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    if (isWebGPU(this.backend)) {
      const out = this.backend.createBuffer(this.byteLength, usage);
      const kr = this.runner as KernelRunner;
      if (op === "sqrt") await kr.sqrt(this.buffer as GPUBuffer, out, this.length);
      else if (op === "abs") await kr.abs(this.buffer as GPUBuffer, out, this.length);
      else if (op === "neg") await kr.neg(this.buffer as GPUBuffer, out, this.length);
      else if (op === "exp") await kr.exp(this.buffer as GPUBuffer, out, this.length);
      else if (op === "log") await kr.log(this.buffer as GPUBuffer, out, this.length);
      else if (op === "relu") await kr.relu(this.buffer as GPUBuffer, out, this.length);
      else if (op === "sigmoid") await kr.sigmoid(this.buffer as GPUBuffer, out, this.length);
      else if (op === "tanh") await kr.tanh(this.buffer as GPUBuffer, out, this.length);
      else if (op === "gelu") await kr.gelu(this.buffer as GPUBuffer, out, this.length);
      else if (op === "pow" && scalarOrExponent !== undefined)
        await kr.powScalar(this.buffer as GPUBuffer, scalarOrExponent, out, this.length);
      this.backend.destroyBuffer(this.buffer as GPUBuffer);
      this.buffer = out;
    } else if (isWebGL(this.backend)) {
      const webglBackend = this.backend as WebGLBackend;
      const webglRunner = this.runner as WebGLRunner;
      const out = webglBackend.createBuffer(this.byteLength, 0);
      const buf = this.buffer as import("./backend/webgl-backend").WebGLBuffer;
      if (op === "sqrt") await webglRunner.sqrt(buf, out, this.length);
      else if (op === "abs") await webglRunner.abs(buf, out, this.length);
      else if (op === "neg") await webglRunner.neg(buf, out, this.length);
      else if (op === "exp") await webglRunner.exp(buf, out, this.length);
      else if (op === "log") await webglRunner.log(buf, out, this.length);
      else if (op === "relu") await webglRunner.relu(buf, out, this.length);
      else if (op === "sigmoid") await webglRunner.sigmoid(buf, out, this.length);
      else if (op === "tanh") await webglRunner.tanh(buf, out, this.length);
      else if (op === "gelu") await webglRunner.gelu(buf, out, this.length);
      else if (op === "pow" && scalarOrExponent !== undefined)
        await webglRunner.powScalar(buf, scalarOrExponent, out, this.length);
      webglBackend.destroyBuffer(buf);
      this.buffer = out;
    } else {
      const cpuRunner = this.runner as CPURunner;
      const cpuBackend = this.backend as CPUBackend;
      const out = cpuBackend.createBuffer(this.byteLength);
      const buf = this.buffer as { data: Float32Array };
      if (op === "sqrt") await cpuRunner.sqrt(buf, out, this.length);
      else if (op === "abs") await cpuRunner.abs(buf, out, this.length);
      else if (op === "neg") await cpuRunner.neg(buf, out, this.length);
      else if (op === "exp") await cpuRunner.exp(buf, out, this.length);
      else if (op === "log") await cpuRunner.log(buf, out, this.length);
      else if (op === "relu") await cpuRunner.relu(buf, out, this.length);
      else if (op === "sigmoid") await cpuRunner.sigmoid(buf, out, this.length);
      else if (op === "tanh") await cpuRunner.tanh(buf, out, this.length);
      else if (op === "gelu") await cpuRunner.gelu(buf, out, this.length);
      else if (op === "pow" && scalarOrExponent !== undefined)
        await cpuRunner.powScalar(buf, scalarOrExponent, out, this.length);
      cpuBackend.destroyBuffer(buf);
      this.buffer = out;
    }
    return this;
  }

  private async addWebGL(other: GPUArray | number): Promise<GPUArray> {
    const webglBackend = this.backend as WebGLBackend;
    const webglRunner = this.runner as WebGLRunner;
    const usage = 0;

    if (typeof other === "number") {
      const scalarData = new Float32Array(this.length).fill(other);
      const scalarBuf = webglBackend.createBufferFromData(scalarData.buffer as ArrayBuffer, usage);
      const out = webglBackend.createBuffer(this.byteLength, usage);
      await webglRunner.add(
        this.buffer as import("./backend/webgl-backend").WebGLBuffer,
        scalarBuf,
        out,
        this.length
      );
      webglBackend.destroyBuffer(scalarBuf);
      webglBackend.destroyBuffer(this.buffer as import("./backend/webgl-backend").WebGLBuffer);
      this.buffer = out;
      return this;
    }
    const out = webglBackend.createBuffer(this.byteLength, usage);
    await webglRunner.add(
      this.buffer as import("./backend/webgl-backend").WebGLBuffer,
      other.buffer as import("./backend/webgl-backend").WebGLBuffer,
      out,
      this.length
    );
    webglBackend.destroyBuffer(this.buffer as import("./backend/webgl-backend").WebGLBuffer);
    this.buffer = out;
    return this;
  }

  /**
   * Element-wise add. Mutates this array in-place.
   * @param other - Another GPUArray of same length, or a scalar number
   * @returns Promise resolving to this (for chaining)
   */
  async add(other: GPUArray | number): Promise<GPUArray> {
    this._ensureNotDisposed();
    if (typeof other !== "number" && other.length !== this.length) {
      errLengthMismatch("add", this.length, other.length);
    }

    if (isWebGL(this.backend)) return this.addWebGL(other);

    if (typeof other === "number") {
      const scalarData = new Float32Array(this.length).fill(other);
      if (isWebGPU(this.backend)) {
        const scalarBuf = this.backend.createBufferFromData(
          scalarData.buffer as ArrayBuffer,
          GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        );
        const out = this.backend.createBuffer(
          this.byteLength,
          GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        );
        await (this.runner as KernelRunner).add(
          this.buffer as GPUBuffer,
          scalarBuf,
          out,
          this.length
        );
        this.backend.destroyBuffer(scalarBuf);
        this.backend.destroyBuffer(this.buffer as GPUBuffer);
        this.buffer = out;
      } else {
        const cpuRunner = this.runner as CPURunner;
        const out = (this.backend as CPUBackend).createBuffer(this.byteLength);
        await cpuRunner.add(
          this.buffer as { data: Float32Array },
          { data: scalarData },
          out,
          this.length
        );
        (this.backend as CPUBackend).destroyBuffer(this.buffer as { data: Float32Array });
        this.buffer = out;
      }
      return this;
    }

    if (isWebGPU(this.backend)) {
      const out = this.backend.createBuffer(
        this.byteLength,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
      );
      await (this.runner as KernelRunner).add(
        this.buffer as GPUBuffer,
        other.buffer as GPUBuffer,
        out,
        this.length
      );
      this.backend.destroyBuffer(this.buffer as GPUBuffer);
      this.buffer = out;
    } else {
      const out = (this.backend as CPUBackend).createBuffer(this.byteLength);
      await (this.runner as CPURunner).add(
        this.buffer as { data: Float32Array },
        other.buffer as { data: Float32Array },
        out,
        this.length
      );
      (this.backend as CPUBackend).destroyBuffer(this.buffer as { data: Float32Array });
      this.buffer = out;
    }
    return this;
  }

  /**
   * Element-wise multiply. Mutates this array in-place.
   * @param other - Another GPUArray of same length, or a scalar number
   * @returns Promise resolving to this (for chaining)
   */
  async mul(other: GPUArray | number): Promise<GPUArray> {
    this._ensureNotDisposed();
    if (typeof other !== "number" && other.length !== this.length) {
      errLengthMismatch("mul", this.length, other.length);
    }

    if (isWebGL(this.backend)) {
      const webglBackend = this.backend as WebGLBackend;
      const webglRunner = this.runner as WebGLRunner;
      const usage = 0;
      const out = webglBackend.createBuffer(this.byteLength, usage);
      if (typeof other === "number") {
        await webglRunner.mulScalar(
          this.buffer as import("./backend/webgl-backend").WebGLBuffer,
          other,
          out,
          this.length
        );
      } else {
        await webglRunner.mul(
          this.buffer as import("./backend/webgl-backend").WebGLBuffer,
          other.buffer as import("./backend/webgl-backend").WebGLBuffer,
          out,
          this.length
        );
      }
      webglBackend.destroyBuffer(this.buffer as import("./backend/webgl-backend").WebGLBuffer);
      this.buffer = out;
      return this;
    }

    if (typeof other === "number") {
      if (isWebGPU(this.backend)) {
        const out = this.backend.createBuffer(
          this.byteLength,
          GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        );
        await (this.runner as KernelRunner).mulScalar(
          this.buffer as GPUBuffer,
          other,
          out,
          this.length
        );
        this.backend.destroyBuffer(this.buffer as GPUBuffer);
        this.buffer = out;
      } else {
        const out = (this.backend as CPUBackend).createBuffer(this.byteLength);
        await (this.runner as CPURunner).mulScalar(
          this.buffer as { data: Float32Array },
          other,
          out,
          this.length
        );
        (this.backend as CPUBackend).destroyBuffer(this.buffer as { data: Float32Array });
        this.buffer = out;
      }
      return this;
    }

    if (isWebGPU(this.backend)) {
      const out = this.backend.createBuffer(
        this.byteLength,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
      );
      await (this.runner as KernelRunner).mul(
        this.buffer as GPUBuffer,
        other.buffer as GPUBuffer,
        out,
        this.length
      );
      this.backend.destroyBuffer(this.buffer as GPUBuffer);
      this.buffer = out;
    } else {
      const out = (this.backend as CPUBackend).createBuffer(this.byteLength);
      await (this.runner as CPURunner).mul(
        this.buffer as { data: Float32Array },
        other.buffer as { data: Float32Array },
        out,
        this.length
      );
      (this.backend as CPUBackend).destroyBuffer(this.buffer as { data: Float32Array });
      this.buffer = out;
    }
    return this;
  }

  private async subWebGL(other: GPUArray | number): Promise<GPUArray> {
    const webglBackend = this.backend as WebGLBackend;
    const webglRunner = this.runner as WebGLRunner;
    const usage = 0;
    const buf = this.buffer as import("./backend/webgl-backend").WebGLBuffer;
    const out = webglBackend.createBuffer(this.byteLength, usage);
    if (typeof other === "number") {
      await webglRunner.subScalar(buf, other, out, this.length);
    } else {
      await webglRunner.sub(
        buf,
        other.buffer as import("./backend/webgl-backend").WebGLBuffer,
        out,
        this.length
      );
    }
    webglBackend.destroyBuffer(buf);
    this.buffer = out;
    return this;
  }

  private async divWebGL(other: GPUArray | number): Promise<GPUArray> {
    const webglBackend = this.backend as WebGLBackend;
    const webglRunner = this.runner as WebGLRunner;
    const usage = 0;
    const buf = this.buffer as import("./backend/webgl-backend").WebGLBuffer;
    const out = webglBackend.createBuffer(this.byteLength, usage);
    if (typeof other === "number") {
      await webglRunner.divScalar(buf, other, out, this.length);
    } else {
      await webglRunner.div(
        buf,
        other.buffer as import("./backend/webgl-backend").WebGLBuffer,
        out,
        this.length
      );
    }
    webglBackend.destroyBuffer(buf);
    this.buffer = out;
    return this;
  }

  /** Element-wise subtract. Mutates this in-place. */
  async sub(other: GPUArray | number): Promise<GPUArray> {
    this._ensureNotDisposed();
    if (typeof other !== "number" && other.length !== this.length)
      errLengthMismatch("sub", this.length, other.length);
    if (isWebGL(this.backend)) return this.subWebGL(other);
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    if (typeof other === "number") {
      const scalarData = new Float32Array(this.length).fill(other);
      if (isWebGPU(this.backend)) {
        const scalarBuf = this.backend.createBufferFromData(
          scalarData.buffer as ArrayBuffer,
          usage
        );
        const out = this.backend.createBuffer(this.byteLength, usage);
        await (this.runner as KernelRunner).sub(
          this.buffer as GPUBuffer,
          scalarBuf,
          out,
          this.length
        );
        this.backend.destroyBuffer(scalarBuf);
        this.backend.destroyBuffer(this.buffer as GPUBuffer);
        this.buffer = out;
      } else {
        const out = (this.backend as CPUBackend).createBuffer(this.byteLength);
        await (this.runner as CPURunner).subScalar(
          this.buffer as { data: Float32Array },
          other,
          out,
          this.length
        );
        (this.backend as CPUBackend).destroyBuffer(this.buffer as { data: Float32Array });
        this.buffer = out;
      }
      return this;
    }
    if (isWebGPU(this.backend)) {
      const out = this.backend.createBuffer(this.byteLength, usage);
      await (this.runner as KernelRunner).sub(
        this.buffer as GPUBuffer,
        other.buffer as GPUBuffer,
        out,
        this.length
      );
      this.backend.destroyBuffer(this.buffer as GPUBuffer);
      this.buffer = out;
    } else {
      const out = (this.backend as CPUBackend).createBuffer(this.byteLength);
      await (this.runner as CPURunner).sub(
        this.buffer as { data: Float32Array },
        other.buffer as { data: Float32Array },
        out,
        this.length
      );
      (this.backend as CPUBackend).destroyBuffer(this.buffer as { data: Float32Array });
      this.buffer = out;
    }
    return this;
  }

  /** Element-wise divide. Mutates this in-place. */
  async div(other: GPUArray | number): Promise<GPUArray> {
    this._ensureNotDisposed();
    if (typeof other !== "number" && other.length !== this.length)
      errLengthMismatch("div", this.length, other.length);
    if (isWebGL(this.backend)) return this.divWebGL(other);
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    if (typeof other === "number") {
      if (isWebGPU(this.backend)) {
        const out = this.backend.createBuffer(this.byteLength, usage);
        await (this.runner as KernelRunner).divScalar(
          this.buffer as GPUBuffer,
          other,
          out,
          this.length
        );
        this.backend.destroyBuffer(this.buffer as GPUBuffer);
        this.buffer = out;
      } else {
        const out = (this.backend as CPUBackend).createBuffer(this.byteLength);
        await (this.runner as CPURunner).divScalar(
          this.buffer as { data: Float32Array },
          other,
          out,
          this.length
        );
        (this.backend as CPUBackend).destroyBuffer(this.buffer as { data: Float32Array });
        this.buffer = out;
      }
      return this;
    }
    if (isWebGPU(this.backend)) {
      const out = this.backend.createBuffer(this.byteLength, usage);
      await (this.runner as KernelRunner).div(
        this.buffer as GPUBuffer,
        other.buffer as GPUBuffer,
        out,
        this.length
      );
      this.backend.destroyBuffer(this.buffer as GPUBuffer);
      this.buffer = out;
    } else {
      const out = (this.backend as CPUBackend).createBuffer(this.byteLength);
      await (this.runner as CPURunner).div(
        this.buffer as { data: Float32Array },
        other.buffer as { data: Float32Array },
        out,
        this.length
      );
      (this.backend as CPUBackend).destroyBuffer(this.buffer as { data: Float32Array });
      this.buffer = out;
    }
    return this;
  }

  /** Element-wise power. Mutates this in-place. */
  async pow(exponent: number): Promise<GPUArray> {
    return this.runUnaryOp("pow", exponent);
  }

  /** Element-wise square root. Mutates this in-place. */
  async sqrt(): Promise<GPUArray> {
    return this.runUnaryOp("sqrt");
  }

  /** Element-wise absolute value. Mutates this in-place. */
  async abs(): Promise<GPUArray> {
    return this.runUnaryOp("abs");
  }

  /** Element-wise negation. Mutates this in-place. */
  async neg(): Promise<GPUArray> {
    return this.runUnaryOp("neg");
  }

  /** Element-wise e^x. Mutates this in-place. */
  async exp(): Promise<GPUArray> {
    return this.runUnaryOp("exp");
  }

  /** Element-wise natural log. Mutates this in-place. */
  async log(): Promise<GPUArray> {
    return this.runUnaryOp("log");
  }

  /** ReLU activation: max(0, x). Mutates this in-place. */
  async relu(): Promise<GPUArray> {
    return this.runUnaryOp("relu");
  }

  /** Sigmoid: 1/(1+exp(-x)). Mutates this in-place. */
  async sigmoid(): Promise<GPUArray> {
    return this.runUnaryOp("sigmoid");
  }

  /** Tanh activation. Mutates this in-place. */
  async tanh(): Promise<GPUArray> {
    return this.runUnaryOp("tanh");
  }

  /** GELU activation. Mutates this in-place. */
  async gelu(): Promise<GPUArray> {
    return this.runUnaryOp("gelu");
  }

  /** Leaky ReLU: x > 0 ? x : alpha * x. Mutates this in-place. */
  async leakyRelu(alpha = 0.01): Promise<GPUArray> {
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    if (isWebGPU(this.backend)) {
      const out = this.backend.createBuffer(this.byteLength, usage);
      await (this.runner as KernelRunner).leakyRelu(
        this.buffer as GPUBuffer,
        alpha,
        out,
        this.length
      );
      this.backend.destroyBuffer(this.buffer as GPUBuffer);
      this.buffer = out;
    } else if (isWebGL(this.backend)) {
      const webglBackend = this.backend as WebGLBackend;
      const webglRunner = this.runner as WebGLRunner;
      const out = webglBackend.createBuffer(this.byteLength, 0);
      await webglRunner.leakyRelu(
        this.buffer as import("./backend/webgl-backend").WebGLBuffer,
        alpha,
        out,
        this.length
      );
      webglBackend.destroyBuffer(this.buffer as import("./backend/webgl-backend").WebGLBuffer);
      this.buffer = out;
    } else {
      const cpuRunner = this.runner as CPURunner;
      const cpuBackend = this.backend as CPUBackend;
      const out = cpuBackend.createBuffer(this.byteLength);
      await cpuRunner.leakyRelu(
        this.buffer as { data: Float32Array },
        alpha,
        out,
        this.length
      );
      cpuBackend.destroyBuffer(this.buffer as { data: Float32Array });
      this.buffer = out;
    }
    return this;
  }

  /** Clamp values to [minVal, maxVal]. Mutates this in-place. */
  async clamp(minVal: number, maxVal: number): Promise<GPUArray> {
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    if (isWebGPU(this.backend)) {
      const out = this.backend.createBuffer(this.byteLength, usage);
      await (this.runner as KernelRunner).clamp(
        this.buffer as GPUBuffer,
        minVal,
        maxVal,
        out,
        this.length
      );
      this.backend.destroyBuffer(this.buffer as GPUBuffer);
      this.buffer = out;
    } else if (isWebGL(this.backend)) {
      const webglBackend = this.backend as WebGLBackend;
      const webglRunner = this.runner as WebGLRunner;
      const out = webglBackend.createBuffer(this.byteLength, 0);
      await webglRunner.clamp(
        this.buffer as import("./backend/webgl-backend").WebGLBuffer,
        minVal,
        maxVal,
        out,
        this.length
      );
      webglBackend.destroyBuffer(this.buffer as import("./backend/webgl-backend").WebGLBuffer);
      this.buffer = out;
    } else {
      const cpuRunner = this.runner as CPURunner;
      const cpuBackend = this.backend as CPUBackend;
      const out = cpuBackend.createBuffer(this.byteLength);
      await cpuRunner.clamp(
        this.buffer as { data: Float32Array },
        minVal,
        maxVal,
        out,
        this.length
      );
      cpuBackend.destroyBuffer(this.buffer as { data: Float32Array });
      this.buffer = out;
    }
    return this;
  }

  /** Element-wise equal: returns new GPUArray with 1 where a==b else 0. */
  async equal(other: GPUArray): Promise<GPUArray> {
    if (other.length !== this.length) errLengthMismatch("equal", this.length, other.length);
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    if (isWebGPU(this.backend)) {
      const out = this.backend.createBuffer(this.byteLength, usage);
      await (this.runner as KernelRunner).equal(
        this.buffer as GPUBuffer,
        other.buffer as GPUBuffer,
        out,
        this.length
      );
      return new GPUArray(this.backend, this.runner, out, this.length, this.shape);
    }
    if (isWebGL(this.backend)) {
      const webglBackend = this.backend as WebGLBackend;
      const webglRunner = this.runner as WebGLRunner;
      const out = webglBackend.createBuffer(this.byteLength, 0);
      await webglRunner.equal(
        this.buffer as import("./backend/webgl-backend").WebGLBuffer,
        other.buffer as import("./backend/webgl-backend").WebGLBuffer,
        out,
        this.length
      );
      return new GPUArray(this.backend, this.runner, out, this.length, this.shape);
    }
    const cpuBackend = this.backend as CPUBackend;
    const cpuRunner = this.runner as CPURunner;
    const out = cpuBackend.createBuffer(this.byteLength);
    await cpuRunner.equal(
      this.buffer as { data: Float32Array },
      other.buffer as { data: Float32Array },
      out,
      this.length
    );
    return new GPUArray(this.backend, this.runner, out, this.length, this.shape);
  }

  /** Element-wise greater: returns new GPUArray with 1 where a>b else 0. */
  async greater(other: GPUArray): Promise<GPUArray> {
    if (other.length !== this.length) errLengthMismatch("greater", this.length, other.length);
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    if (isWebGPU(this.backend)) {
      const out = this.backend.createBuffer(this.byteLength, usage);
      await (this.runner as KernelRunner).greater(
        this.buffer as GPUBuffer,
        other.buffer as GPUBuffer,
        out,
        this.length
      );
      return new GPUArray(this.backend, this.runner, out, this.length, this.shape);
    }
    if (isWebGL(this.backend)) {
      const webglBackend = this.backend as WebGLBackend;
      const webglRunner = this.runner as WebGLRunner;
      const out = webglBackend.createBuffer(this.byteLength, 0);
      await webglRunner.greater(
        this.buffer as import("./backend/webgl-backend").WebGLBuffer,
        other.buffer as import("./backend/webgl-backend").WebGLBuffer,
        out,
        this.length
      );
      return new GPUArray(this.backend, this.runner, out, this.length, this.shape);
    }
    const cpuBackend = this.backend as CPUBackend;
    const cpuRunner = this.runner as CPURunner;
    const out = cpuBackend.createBuffer(this.byteLength);
    await cpuRunner.greater(
      this.buffer as { data: Float32Array },
      other.buffer as { data: Float32Array },
      out,
      this.length
    );
    return new GPUArray(this.backend, this.runner, out, this.length, this.shape);
  }

  /** Element-wise less: returns new GPUArray with 1 where a<b else 0. */
  async less(other: GPUArray): Promise<GPUArray> {
    if (other.length !== this.length) errLengthMismatch("less", this.length, other.length);
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    if (isWebGPU(this.backend)) {
      const out = this.backend.createBuffer(this.byteLength, usage);
      await (this.runner as KernelRunner).less(
        this.buffer as GPUBuffer,
        other.buffer as GPUBuffer,
        out,
        this.length
      );
      return new GPUArray(this.backend, this.runner, out, this.length, this.shape);
    }
    if (isWebGL(this.backend)) {
      const webglBackend = this.backend as WebGLBackend;
      const webglRunner = this.runner as WebGLRunner;
      const out = webglBackend.createBuffer(this.byteLength, 0);
      await webglRunner.less(
        this.buffer as import("./backend/webgl-backend").WebGLBuffer,
        other.buffer as import("./backend/webgl-backend").WebGLBuffer,
        out,
        this.length
      );
      return new GPUArray(this.backend, this.runner, out, this.length, this.shape);
    }
    const cpuBackend = this.backend as CPUBackend;
    const cpuRunner = this.runner as CPURunner;
    const out = cpuBackend.createBuffer(this.byteLength);
    await cpuRunner.less(
      this.buffer as { data: Float32Array },
      other.buffer as { data: Float32Array },
      out,
      this.length
    );
    return new GPUArray(this.backend, this.runner, out, this.length, this.shape);
  }

  /**
   * Reduce sum. axis=undefined reduces all. axis=0/1 for 2D reduces along that axis.
   * @returns Promise resolving to scalar or GPUArray
   */
  async sum(axis?: number): Promise<number | GPUArray> {
    if (axis !== undefined) return this.sumAxis(axis);
    this._ensureNotDisposed();
    if (this.length === 0) return 0;
    if (this.length === 1) {
      const result = new Float32Array(1);
      if (isWebGPU(this.backend)) {
        await this.backend.readBuffer(this.buffer as GPUBuffer, result.buffer);
      } else if (isWebGL(this.backend)) {
        await this.backend.readBuffer(
          this.buffer as import("./backend/webgl-backend").WebGLBuffer,
          result.buffer
        );
      } else {
        (this.backend as CPUBackend).readBuffer(
          this.buffer as { data: Float32Array },
          result.buffer
        );
      }
      return result[0];
    }

    if (isWebGL(this.backend)) {
      const webglBackend = this.backend as WebGLBackend;
      const webglRunner = this.runner as WebGLRunner;
      const out = webglBackend.createBuffer(4, 0);
      await webglRunner.reduceSum(
        this.buffer as import("./backend/webgl-backend").WebGLBuffer,
        out,
        this.length
      );
      const result = new Float32Array(1);
      await webglBackend.readBuffer(out, result.buffer);
      webglBackend.destroyBuffer(out);
      return result[0];
    }

    if (isWebGPU(this.backend)) {
      let inputBuffer = this.buffer as GPUBuffer;
      let inputLength = this.length;
      while (true) {
        const outputLength = Math.ceil(inputLength / WORKGROUP_SIZE);
        const outputBuffer = this.backend.createBuffer(
          Math.max(4, outputLength * 4),
          GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        );
        await (this.runner as KernelRunner).reduceSum(inputBuffer, outputBuffer, inputLength);
        if (inputBuffer !== this.buffer) this.backend.destroyBuffer(inputBuffer);
        if (outputLength === 1) {
          const result = new Float32Array(1);
          await this.backend.readBuffer(outputBuffer, result.buffer);
          this.backend.destroyBuffer(outputBuffer);
          return result[0];
        }
        inputBuffer = outputBuffer;
        inputLength = outputLength;
      }
    } else {
      const out = (this.backend as CPUBackend).createBuffer(4);
      await (this.runner as CPURunner).reduceSum(
        this.buffer as { data: Float32Array },
        out,
        this.length
      );
      return out.data[0];
    }
  }

  /**
   * Reduce max. axis=undefined reduces all. axis=0/1 for 2D reduces along that axis.
   */
  async max(axis?: number): Promise<number | GPUArray> {
    if (axis !== undefined) return this.maxAxis(axis);
    if (this.length === 0) return -Infinity;
    if (this.length === 1) {
      const result = new Float32Array(1);
      if (isWebGPU(this.backend)) {
        await this.backend.readBuffer(this.buffer as GPUBuffer, result.buffer);
      } else if (isWebGL(this.backend)) {
        await this.backend.readBuffer(
          this.buffer as import("./backend/webgl-backend").WebGLBuffer,
          result.buffer
        );
      } else {
        (this.backend as CPUBackend).readBuffer(
          this.buffer as { data: Float32Array },
          result.buffer
        );
      }
      return result[0];
    }

    if (isWebGL(this.backend)) {
      const webglBackend = this.backend as WebGLBackend;
      const webglRunner = this.runner as WebGLRunner;
      const out = webglBackend.createBuffer(4, 0);
      await webglRunner.reduceMax(
        this.buffer as import("./backend/webgl-backend").WebGLBuffer,
        out,
        this.length
      );
      const result = new Float32Array(1);
      await webglBackend.readBuffer(out, result.buffer);
      webglBackend.destroyBuffer(out);
      return result[0];
    }

    if (isWebGPU(this.backend)) {
      let inputBuffer = this.buffer as GPUBuffer;
      let inputLength = this.length;
      while (true) {
        const outputLength = Math.ceil(inputLength / WORKGROUP_SIZE);
        const outputBuffer = this.backend.createBuffer(
          Math.max(4, outputLength * 4),
          GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        );
        await (this.runner as KernelRunner).reduceMax(inputBuffer, outputBuffer, inputLength);
        if (inputBuffer !== this.buffer) this.backend.destroyBuffer(inputBuffer);
        if (outputLength === 1) {
          const result = new Float32Array(1);
          await this.backend.readBuffer(outputBuffer, result.buffer);
          this.backend.destroyBuffer(outputBuffer);
          return result[0];
        }
        inputBuffer = outputBuffer;
        inputLength = outputLength;
      }
    } else {
      const out = (this.backend as CPUBackend).createBuffer(4);
      await (this.runner as CPURunner).reduceMax(
        this.buffer as { data: Float32Array },
        out,
        this.length
      );
      return out.data[0];
    }
  }

  /** Mean. axis=undefined reduces all. axis=0/1 for 2D reduces along that axis. */
  async mean(axis?: number): Promise<number | GPUArray> {
    if (axis !== undefined) return this.meanAxis(axis);
    if (this.length === 0) return NaN;
    return ((await this.sum()) as number) / this.length;
  }

  /** Variance of all elements (population variance). */
  async variance(): Promise<number> {
    if (this.length === 0) return NaN;
    const sum = (await this.sum()) as number;
    const copy = await this.clone();
    await copy.mul(this);
    const sumSq = (await copy.sum()) as number;
    const mean = sum / this.length;
    return sumSq / this.length - mean * mean;
  }

  /** Standard deviation of all elements. */
  async std(): Promise<number> {
    const v = await this.variance();
    return Math.sqrt(Math.max(0, v));
  }

  /** Index of maximum element. */
  async argmax(): Promise<number> {
    if (this.length === 0) return -1;
    const data = await this.toArray();
    let max = data[0];
    let idx = 0;
    for (let i = 1; i < this.length; i++) {
      if (data[i] > max) {
        max = data[i];
        idx = i;
      }
    }
    return idx;
  }

  /** Index of minimum element. */
  async argmin(): Promise<number> {
    if (this.length === 0) return -1;
    const data = await this.toArray();
    let min = data[0];
    let idx = 0;
    for (let i = 1; i < this.length; i++) {
      if (data[i] < min) {
        min = data[i];
        idx = i;
      }
    }
    return idx;
  }

  /** Reduce min over all elements. */
  async min(): Promise<number> {
    if (this.length === 0) return Infinity;
    if (this.length === 1) {
      const result = new Float32Array(1);
      if (isWebGPU(this.backend)) {
        await this.backend.readBuffer(this.buffer as GPUBuffer, result.buffer);
      } else if (isWebGL(this.backend)) {
        await this.backend.readBuffer(
          this.buffer as import("./backend/webgl-backend").WebGLBuffer,
          result.buffer
        );
      } else {
        (this.backend as CPUBackend).readBuffer(
          this.buffer as { data: Float32Array },
          result.buffer
        );
      }
      return result[0];
    }
    if (isWebGL(this.backend)) {
      const webglBackend = this.backend as WebGLBackend;
      const webglRunner = this.runner as WebGLRunner;
      const out = webglBackend.createBuffer(4, 0);
      await webglRunner.reduceMin(
        this.buffer as import("./backend/webgl-backend").WebGLBuffer,
        out,
        this.length
      );
      const result = new Float32Array(1);
      await webglBackend.readBuffer(out, result.buffer);
      webglBackend.destroyBuffer(out);
      return result[0];
    }
    if (isWebGPU(this.backend)) {
      let inputBuffer = this.buffer as GPUBuffer;
      let inputLength = this.length;
      while (true) {
        const outputLength = Math.ceil(inputLength / WORKGROUP_SIZE);
        const outputBuffer = this.backend.createBuffer(
          Math.max(4, outputLength * 4),
          GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        );
        await (this.runner as KernelRunner).reduceMin(inputBuffer, outputBuffer, inputLength);
        if (inputBuffer !== this.buffer) this.backend.destroyBuffer(inputBuffer);
        if (outputLength === 1) {
          const result = new Float32Array(1);
          await this.backend.readBuffer(outputBuffer, result.buffer);
          this.backend.destroyBuffer(outputBuffer);
          return result[0];
        }
        inputBuffer = outputBuffer;
        inputLength = outputLength;
      }
    } else {
      const out = (this.backend as CPUBackend).createBuffer(4);
      await (this.runner as CPURunner).reduceMin(
        this.buffer as { data: Float32Array },
        out,
        this.length
      );
      return out.data[0];
    }
  }

  /**
   * Dot product with another vector of the same length.
   * @param other - GPUArray of same length
   * @returns Promise resolving to the dot product scalar
   */
  async dot(other: GPUArray): Promise<number> {
    if (other.length !== this.length) errLengthMismatch("dot", this.length, other.length);

    if (isWebGL(this.backend)) {
      const webglBackend = this.backend as WebGLBackend;
      const webglRunner = this.runner as WebGLRunner;
      const multiplied = webglBackend.createBuffer(this.byteLength, 0);
      await webglRunner.mul(
        this.buffer as import("./backend/webgl-backend").WebGLBuffer,
        other.buffer as import("./backend/webgl-backend").WebGLBuffer,
        multiplied,
        this.length
      );
      const temp = new GPUArray(this.backend, this.runner, multiplied, this.length);
      const result = (await temp.sum()) as number;
      webglBackend.destroyBuffer(multiplied);
      return result;
    }

    if (isWebGPU(this.backend)) {
      const multiplied = this.backend.device!.createBuffer({
        size: this.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
      await (this.runner as KernelRunner).mul(
        this.buffer as GPUBuffer,
        other.buffer as GPUBuffer,
        multiplied,
        this.length
      );
      const temp = new GPUArray(this.backend, this.runner, multiplied, this.length);
      const result = (await temp.sum()) as number;
      this.backend.destroyBuffer(multiplied);
      return result;
    } else {
      let sum = 0;
      const a = (this.buffer as { data: Float32Array }).data;
      const b = (other.buffer as { data: Float32Array }).data;
      for (let i = 0; i < this.length; i++) sum += a[i] * b[i];
      return sum;
    }
  }

  /** Deep copy. Returns new GPUArray with same data. */
  async clone(): Promise<GPUArray> {
    this._ensureNotDisposed();
    const data = await this.toArray();
    const G = (globalThis as any).GPUBufferUsage;
    const usage = G.STORAGE | G.COPY_SRC | G.COPY_DST;
    const newBuffer = this.backend.createBufferFromData(data.buffer as ArrayBuffer, usage);
    return new GPUArray(this.backend, this.runner, newBuffer, this.length, this.shape);
  }

  getBuffer(): GPUBuffer | import("./backend/webgl-backend").WebGLBuffer | { data: Float32Array } {
    this._ensureNotDisposed();
    return this.buffer;
  }

  /**
   * Synchronously copy data to Float32Array. Only works with CPU backend.
   * @throws If backend is not CPU
   */
  toArraySync(): Float32Array {
    this._ensureNotDisposed();
    if (!isWebGPU(this.backend) && !isWebGL(this.backend)) {
      return new Float32Array((this.buffer as { data: Float32Array }).data.slice());
    }
    throw new Error("toArraySync() is only supported on CPU backend. Use toArray() for async read.");
  }

  /** Slice [start, end). Returns new GPUArray. */
  async slice(start: number, end?: number): Promise<GPUArray> {
    this._ensureNotDisposed();
    const e = end ?? this.length;
    const data = await this.toArray();
    const sliced = data.slice(start, e);
    const G = (globalThis as any).GPUBufferUsage;
    const usage = G.STORAGE | G.COPY_SRC | G.COPY_DST;
    const newBuffer = this.backend.createBufferFromData(sliced.buffer as ArrayBuffer, usage);
    return new GPUArray(this.backend, this.runner, newBuffer, sliced.length, [sliced.length]);
  }

  /** Get element at index. */
  async get(index: number): Promise<number> {
    this._ensureNotDisposed();
    const data = await this.toArray();
    return data[index];
  }

  /** Set element at index. Mutates in-place. */
  async set(index: number, value: number): Promise<GPUArray> {
    this._ensureNotDisposed();
    if (isWebGPU(this.backend)) {
      const data = await this.toArray();
      data[index] = value;
      await this.backend.writeBuffer(this.buffer as GPUBuffer, data.buffer as ArrayBuffer);
    } else if (isWebGL(this.backend)) {
      const data = await this.toArray();
      data[index] = value;
      const webglBackend = this.backend as WebGLBackend;
      webglBackend.destroyBuffer(this.buffer as import("./backend/webgl-backend").WebGLBuffer);
      this.buffer = webglBackend.createBufferFromData(data.buffer as ArrayBuffer, 0) as Buffer;
    } else {
      (this.buffer as { data: Float32Array }).data[index] = value;
    }
    return this;
  }

  /** Concatenate with other. Returns new GPUArray. */
  async concat(other: GPUArray): Promise<GPUArray> {
    this._ensureNotDisposed();
    const a = await this.toArray();
    const b = await other.toArray();
    const combined = new Float32Array(a.length + b.length);
    combined.set(a);
    combined.set(b, a.length);
    const G = (globalThis as any).GPUBufferUsage;
    const usage = G.STORAGE | G.COPY_SRC | G.COPY_DST;
    const newBuffer = this.backend.createBufferFromData(combined.buffer as ArrayBuffer, usage);
    return new GPUArray(this.backend, this.runner, newBuffer, combined.length, [combined.length]);
  }

  /** Split into numSections arrays. Returns array of GPUArrays. */
  async split(numSections: number): Promise<GPUArray[]> {
    this._ensureNotDisposed();
    if (this.length % numSections !== 0) {
      throw new Error(`split: length ${this.length} not divisible by ${numSections}`);
    }
    const sectionLen = this.length / numSections;
    const data = await this.toArray();
    const G = (globalThis as any).GPUBufferUsage;
    const usage = G.STORAGE | G.COPY_SRC | G.COPY_DST;
    const result: GPUArray[] = [];
    for (let i = 0; i < numSections; i++) {
      const section = data.slice(i * sectionLen, (i + 1) * sectionLen);
      const buf = this.backend.createBufferFromData(section.buffer as ArrayBuffer, usage);
      result.push(new GPUArray(this.backend, this.runner, buf, sectionLen, [sectionLen]));
    }
    return result;
  }

  /** Flatten to 1D. Same as reshape(length). */
  flatten(): GPUArray {
    this._ensureNotDisposed();
    this._shape = [this.length];
    return this;
  }

  /** Remove dimensions of size 1. */
  squeeze(): GPUArray {
    this._ensureNotDisposed();
    this._shape = this._shape.filter((d) => d !== 1);
    if (this._shape.length === 0) this._shape = [1];
    return this;
  }

  /** Insert dimension of size 1 at dim. */
  unsqueeze(dim: number): GPUArray {
    this._ensureNotDisposed();
    if (dim < 0) dim = this._shape.length + 1 + dim;
    this._shape.splice(dim, 0, 1);
    return this;
  }

  /** Broadcast to targetShape. Replicates along dims of size 1. Returns new GPUArray. */
  async broadcast(targetShape: number[]): Promise<GPUArray> {
    this._ensureNotDisposed();
    const src = this._shape;
    const dst = targetShape;
    const srcLen = src.length;
    const dstLen = dst.length;
    const nd = Math.max(srcLen, dstLen);
    const s = new Array(nd).fill(1);
    const t = new Array(nd).fill(1);
    for (let i = 0; i < srcLen; i++) s[nd - srcLen + i] = src[i];
    for (let i = 0; i < dstLen; i++) t[nd - dstLen + i] = dst[i];
    for (let i = 0; i < nd; i++) {
      if (s[i] !== t[i] && s[i] !== 1) {
        throw new Error(
          `broadcast: cannot broadcast [${src.join(", ")}] to [${targetShape.join(", ")}]`
        );
      }
    }
    const outSize = t.reduce((a, b) => a * b, 1);
    const data = await this.toArray();
    const out = new Float32Array(outSize);
    const srcStrides: number[] = [];
    let stride = 1;
    for (let i = nd - 1; i >= 0; i--) {
      srcStrides[i] = stride;
      stride *= s[i];
    }
    const dstStrides: number[] = [];
    stride = 1;
    for (let i = nd - 1; i >= 0; i--) {
      dstStrides[i] = stride;
      stride *= t[i];
    }
    for (let idx = 0; idx < outSize; idx++) {
      let srcIdx = 0;
      for (let d = 0; d < nd; d++) {
        const coord = Math.floor(idx / dstStrides[d]) % t[d];
        srcIdx += (s[d] === 1 ? 0 : coord % s[d]) * srcStrides[d];
      }
      out[idx] = data[srcIdx];
    }
    const G = (globalThis as any).GPUBufferUsage;
    const buf = this.backend.createBufferFromData(out.buffer as ArrayBuffer, G.STORAGE | G.COPY_SRC | G.COPY_DST);
    return new GPUArray(this.backend, this.runner, buf, outSize, [...targetShape]);
  }

  /** Sum over axis. axis=undefined reduces all. axis=0 over rows, axis=1 over cols for 2D. */
  async sumAxis(axis?: number): Promise<GPUArray | number> {
    this._ensureNotDisposed();
    if (axis === undefined) return this.sum();
    const s = this._shape;
    if (s.length < 2) return this.sum();
    const data = await this.toArray();
    const ax = axis < 0 ? s.length + axis : axis;
    const outShape = s.filter((_, i) => i !== ax);
    const outSize = outShape.reduce((a, b) => a * b, 1);
    const out = new Float32Array(outSize);
    const reduceDim = s[ax];
    const inner = s.slice(ax + 1).reduce((a, b) => a * b, 1);
    const outer = s.slice(0, ax).reduce((a, b) => a * b, 1);
    const stride = inner * reduceDim;
    for (let o = 0; o < outer; o++) {
      for (let i = 0; i < inner; i++) {
        let sum = 0;
        for (let r = 0; r < reduceDim; r++) {
          sum += data[o * stride + r * inner + i];
        }
        out[o * inner + i] = sum;
      }
    }
    const G = (globalThis as any).GPUBufferUsage;
    const buf = this.backend.createBufferFromData(out.buffer as ArrayBuffer, G.STORAGE | G.COPY_SRC | G.COPY_DST);
    return new GPUArray(this.backend, this.runner, buf, outSize, outShape);
  }

  /** Mean over axis. */
  async meanAxis(axis?: number): Promise<GPUArray | number> {
    const res = await this.sumAxis(axis);
    if (typeof res === "number") return res;
    const ax = axis!;
    const dim = this._shape[ax < 0 ? this._shape.length + ax : ax];
    const copy = await (res as GPUArray).clone();
    await copy.div(dim);
    return copy;
  }

  /** Max over axis. */
  async maxAxis(axis?: number): Promise<GPUArray | number> {
    this._ensureNotDisposed();
    if (axis === undefined) return this.max();
    const s = this._shape;
    if (s.length < 2) return this.max();
    const data = await this.toArray();
    const ax = axis < 0 ? s.length + axis : axis;
    const outShape = s.filter((_, i) => i !== ax);
    const outSize = outShape.reduce((a, b) => a * b, 1);
    const out = new Float32Array(outSize);
    const reduceDim = s[ax];
    const inner = s.slice(ax + 1).reduce((a, b) => a * b, 1);
    const outer = s.slice(0, ax).reduce((a, b) => a * b, 1);
    const stride = inner * reduceDim;
    for (let o = 0; o < outer; o++) {
      for (let i = 0; i < inner; i++) {
        let m = -Infinity;
        for (let r = 0; r < reduceDim; r++) {
          const v = data[o * stride + r * inner + i];
          if (v > m) m = v;
        }
        out[o * inner + i] = m === -Infinity ? 0 : m;
      }
    }
    const G = (globalThis as any).GPUBufferUsage;
    const buf = this.backend.createBufferFromData(out.buffer as ArrayBuffer, G.STORAGE | G.COPY_SRC | G.COPY_DST);
    return new GPUArray(this.backend, this.runner, buf, outSize, outShape);
  }

  /** Normalize along axis. L2 norm per slice. Returns new GPUArray. */
  async normalize(axis?: number): Promise<GPUArray> {
    this._ensureNotDisposed();
    const ax = axis ?? this._shape.length - 1;
    const s = this._shape;
    const dim = s[ax < 0 ? s.length + ax : ax];
    const data = await this.toArray();
    const out = new Float32Array(data);
    const inner = s.slice(ax + 1).reduce((a, b) => a * b, 1);
    const outer = s.slice(0, ax).reduce((a, b) => a * b, 1);
    const stride = inner * dim;
    for (let o = 0; o < outer; o++) {
      for (let i = 0; i < inner; i++) {
        let sumSq = 0;
        for (let r = 0; r < dim; r++) {
          const v = out[o * stride + r * inner + i];
          sumSq += v * v;
        }
        const norm = Math.sqrt(sumSq) || 1;
        for (let r = 0; r < dim; r++) {
          out[o * stride + r * inner + i] /= norm;
        }
      }
    }
    const G = (globalThis as any).GPUBufferUsage;
    const buf = this.backend.createBufferFromData(out.buffer as ArrayBuffer, G.STORAGE | G.COPY_SRC | G.COPY_DST);
    return new GPUArray(this.backend, this.runner, buf, this.length, this.shape);
  }

  /** L2 norm (default). Use ord=1 for L1, ord=2 for L2. */
  async norm(ord: number = 2): Promise<number> {
    this._ensureNotDisposed();
    if (ord === 2) {
      const copy = await this.clone();
      await copy.mul(this);
      const sumSq = (await copy.sum()) as number;
      return Math.sqrt(Math.max(0, sumSq));
    }
    if (ord === 1) {
      const copy = await this.clone();
      await copy.abs();
      return (await copy.sum()) as number;
    }
    throw new Error(`norm: ord=${ord} not supported (use 1 or 2)`);
  }

  /** Outer product. Returns matrix [this.length, other.length]. */
  async outer(other: GPUArray): Promise<GPUArray> {
    this._ensureNotDisposed();
    const a = await this.toArray();
    const b = await other.toArray();
    const out = new Float32Array(this.length * other.length);
    for (let i = 0; i < this.length; i++) {
      for (let j = 0; j < other.length; j++) {
        out[i * other.length + j] = a[i] * b[j];
      }
    }
    const G = (globalThis as any).GPUBufferUsage;
    const usage = G.STORAGE | G.COPY_SRC | G.COPY_DST;
    const buf = this.backend.createBufferFromData(out.buffer as ArrayBuffer, usage);
    return new GPUArray(this.backend, this.runner, buf, out.length, [this.length, other.length]);
  }

  /** Mean squared error with target. */
  async mse(target: GPUArray): Promise<number> {
    this._ensureNotDisposed();
    if (target.length !== this.length) errLengthMismatch("mse", this.length, target.length);
    const copy = await this.clone();
    await copy.sub(target);
    await copy.mul(copy);
    return (await copy.mean()) as number;
  }

  /** Cross-entropy loss. Input: logits or probs. Target: one-hot or class indices (as floats). */
  async crossEntropy(target: GPUArray): Promise<number> {
    this._ensureNotDisposed();
    if (target.length !== this.length) errLengthMismatch("crossEntropy", this.length, target.length);
    const eps = 1e-7;
    const data = await this.toArray();
    const tData = await target.toArray();
    let sum = 0;
    for (let i = 0; i < this.length; i++) {
      const p = Math.max(eps, Math.min(1 - eps, data[i]));
      sum -= tData[i] * Math.log(p);
    }
    return sum;
  }
}
