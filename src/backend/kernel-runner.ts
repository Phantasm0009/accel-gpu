/**
 * Kernel runner - executes pre-built WGSL compute shaders
 */

import type { WebGPUBackend } from "./webgpu";
import {
  ADD_SHADER,
  MUL_SHADER,
  MUL_SCALAR_SHADER,
  REDUCE_SUM_SHADER,
  REDUCE_MAX_SHADER,
  MATMUL_SHADER,
  SOFTMAX_SHADER,
  LAYER_NORM_SHADER,
  ATTENTION_SCORES_SHADER,
} from "../kernels/shaders";

const WORKGROUP_SIZE = 256;

export class KernelRunner {
  private backend: WebGPUBackend;
  private pipelineCache = new Map<string, GPUComputePipeline>();

  constructor(backend: WebGPUBackend) {
    this.backend = backend;
  }

  private async getPipeline(name: string, shader: string): Promise<GPUComputePipeline> {
    let pipeline = this.pipelineCache.get(name);
    if (!pipeline) {
      pipeline = await this.backend.createComputePipeline(shader, "main");
      this.pipelineCache.set(name, pipeline);
    }
    return pipeline;
  }

  async add(a: GPUBuffer, b: GPUBuffer, out: GPUBuffer, length: number): Promise<void> {
    const pipeline = await this.getPipeline("add", ADD_SHADER);
    const bindGroup = this.backend.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: a } },
        { binding: 1, resource: { buffer: b } },
        { binding: 2, resource: { buffer: out } },
      ],
    });
    const workgroups = Math.ceil(length / WORKGROUP_SIZE);
    this.backend.runPipeline(pipeline, [bindGroup], [workgroups]);
  }

  async mul(a: GPUBuffer, b: GPUBuffer, out: GPUBuffer, length: number): Promise<void> {
    const pipeline = await this.getPipeline("mul", MUL_SHADER);
    const bindGroup = this.backend.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: a } },
        { binding: 1, resource: { buffer: b } },
        { binding: 2, resource: { buffer: out } },
      ],
    });
    const workgroups = Math.ceil(length / WORKGROUP_SIZE);
    this.backend.runPipeline(pipeline, [bindGroup], [workgroups]);
  }

  async mulScalar(a: GPUBuffer, scalar: number, out: GPUBuffer, length: number): Promise<void> {
    const pipeline = await this.getPipeline("mul_scalar", MUL_SCALAR_SHADER);
    const uniformBuffer = this.backend.device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.backend.queue.writeBuffer(uniformBuffer, 0, new Float32Array([scalar]).buffer);
    const bindGroup = this.backend.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: a } },
        { binding: 1, resource: { buffer: uniformBuffer } },
        { binding: 2, resource: { buffer: out } },
      ],
    });
    const workgroups = Math.ceil(length / WORKGROUP_SIZE);
    this.backend.runPipeline(pipeline, [bindGroup], [workgroups]);
    uniformBuffer.destroy();
  }

  async reduceSum(input: GPUBuffer, output: GPUBuffer, length: number): Promise<void> {
    const pipeline = await this.getPipeline("reduce_sum", REDUCE_SUM_SHADER);
    const workgroups = Math.ceil(length / WORKGROUP_SIZE);
    const bindGroup = this.backend.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: input } },
        { binding: 1, resource: { buffer: output } },
      ],
    });
    this.backend.runPipeline(pipeline, [bindGroup], [workgroups]);
  }

  async reduceMax(input: GPUBuffer, output: GPUBuffer, length: number): Promise<void> {
    const pipeline = await this.getPipeline("reduce_max", REDUCE_MAX_SHADER);
    const workgroups = Math.ceil(length / WORKGROUP_SIZE);
    const bindGroup = this.backend.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: input } },
        { binding: 1, resource: { buffer: output } },
      ],
    });
    this.backend.runPipeline(pipeline, [bindGroup], [workgroups]);
  }

  async matmul(
    a: GPUBuffer,
    b: GPUBuffer,
    out: GPUBuffer,
    M: number,
    N: number,
    K: number
  ): Promise<void> {
    const pipeline = await this.getPipeline("matmul", MATMUL_SHADER);

    const paramsBuffer = this.backend.device.createBuffer({
      size: 12,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.backend.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([M, N, K]).buffer);

    const bindGroup = this.backend.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: a } },
        { binding: 1, resource: { buffer: b } },
        { binding: 2, resource: { buffer: out } },
        { binding: 3, resource: { buffer: paramsBuffer } },
      ],
    });

    const workgroupsX = Math.ceil(M / 8);
    const workgroupsY = Math.ceil(N / 8);
    this.backend.runPipeline(pipeline, [bindGroup], [workgroupsX, workgroupsY]);

    paramsBuffer.destroy();
  }

  async softmax(input: GPUBuffer, output: GPUBuffer, rows: number, cols: number): Promise<void> {
    const pipeline = await this.getPipeline("softmax", SOFTMAX_SHADER);

    const paramsBuffer = this.backend.device.createBuffer({
      size: 8,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.backend.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([rows, cols]).buffer);

    const bindGroup = this.backend.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: input } },
        { binding: 1, resource: { buffer: output } },
        { binding: 2, resource: { buffer: paramsBuffer } },
      ],
    });

    const workgroups = Math.ceil(rows / 256);
    this.backend.runPipeline(pipeline, [bindGroup], [workgroups]);

    paramsBuffer.destroy();
  }

  async layerNorm(
    input: GPUBuffer,
    gamma: GPUBuffer,
    beta: GPUBuffer,
    output: GPUBuffer,
    rows: number,
    cols: number
  ): Promise<void> {
    const pipeline = await this.getPipeline("layer_norm", LAYER_NORM_SHADER);
    const paramsBuffer = this.backend.device.createBuffer({
      size: 8,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.backend.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([rows, cols]).buffer);
    const bindGroup = this.backend.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: input } },
        { binding: 1, resource: { buffer: gamma } },
        { binding: 2, resource: { buffer: beta } },
        { binding: 3, resource: { buffer: output } },
        { binding: 4, resource: { buffer: paramsBuffer } },
      ],
    });
    const workgroups = Math.ceil(rows / 256);
    this.backend.runPipeline(pipeline, [bindGroup], [workgroups]);
    paramsBuffer.destroy();
  }

  async attentionScores(
    Q: GPUBuffer,
    K: GPUBuffer,
    output: GPUBuffer,
    seq: number,
    dim: number
  ): Promise<void> {
    const pipeline = await this.getPipeline("attention_scores", ATTENTION_SCORES_SHADER);
    const paramsBuffer = this.backend.device.createBuffer({
      size: 12,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.backend.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([seq, dim, 0]).buffer);
    const bindGroup = this.backend.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: Q } },
        { binding: 1, resource: { buffer: K } },
        { binding: 2, resource: { buffer: output } },
        { binding: 3, resource: { buffer: paramsBuffer } },
      ],
    });
    const workgroupsX = Math.ceil(seq / 8);
    const workgroupsY = Math.ceil(seq / 8);
    this.backend.runPipeline(pipeline, [bindGroup], [workgroupsX, workgroupsY]);
    paramsBuffer.destroy();
  }
}
