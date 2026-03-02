/**
 * WebGPU backend - device initialization and buffer management
 */

export interface WebGPUBackend {
  device: GPUDevice;
  queue: GPUQueue;
  createBuffer(size: number, usage: GPUBufferUsageFlags): GPUBuffer;
  createBufferFromData(data: ArrayBuffer, usage: GPUBufferUsageFlags): GPUBuffer;
  writeBuffer(gpuBuffer: GPUBuffer, data: ArrayBuffer): Promise<void>;
  readBuffer(gpuBuffer: GPUBuffer, output: ArrayBuffer): Promise<void>;
  destroyBuffer(buffer: GPUBuffer): void;
  createComputePipeline(shader: string, entryPoint: string): Promise<GPUComputePipeline>;
  runPipeline(
    pipeline: GPUComputePipeline,
    bindGroups: GPUBindGroup[],
    workgroups: [number, number?, number?]
  ): void;
}

export async function createWebGPUBackend(): Promise<WebGPUBackend> {
  if (!navigator.gpu) {
    throw new Error("WebGPU is not supported in this environment");
  }

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new Error("Failed to get WebGPU adapter");
  }

  const device = await adapter.requestDevice();
  const queue = device.queue;

  const bufferPool: { size: number; buffers: GPUBuffer[] }[] = [];
  const POOL_BUCKETS = [256, 1024, 4096, 16384, 65536, 262144, 1048576];

  function getPoolBucket(size: number): number {
    for (const bucket of POOL_BUCKETS) {
      if (size <= bucket) return bucket;
    }
    return size;
  }

  return {
    device,
    queue,

    createBuffer(size: number, usage: GPUBufferUsageFlags): GPUBuffer {
      const alignedSize = Math.max(4, (size + 3) & ~3);
      const bucket = getPoolBucket(alignedSize);
      const pool = bufferPool.find((p) => p.size === bucket);
      if (pool && pool.buffers.length > 0) {
        const buf = pool.buffers.pop()!;
        if (buf.size >= alignedSize) return buf;
        buf.destroy();
      }
      return device.createBuffer({
        size: alignedSize,
        usage,
        mappedAtCreation: false,
      });
    },

    createBufferFromData(data: ArrayBuffer, usage: GPUBufferUsageFlags): GPUBuffer {
      const size = Math.max(4, (data.byteLength + 3) & ~3);
      const buffer = device.createBuffer({
        size,
        usage,
        mappedAtCreation: true,
      });
      new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(data));
      buffer.unmap();
      return buffer;
    },

    async writeBuffer(gpuBuffer: GPUBuffer, data: ArrayBuffer): Promise<void> {
      queue.writeBuffer(gpuBuffer, 0, data);
    },

    async readBuffer(gpuBuffer: GPUBuffer, output: ArrayBuffer): Promise<void> {
      const size = output.byteLength;
      const staging = device.createBuffer({
        size,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });

      const encoder = device.createCommandEncoder();
      encoder.copyBufferToBuffer(gpuBuffer, 0, staging, 0, size);
      queue.submit([encoder.finish()]);

      await staging.mapAsync(GPUMapMode.READ);
      new Uint8Array(output).set(new Uint8Array(staging.getMappedRange()));
      staging.unmap();
      staging.destroy();
    },

    destroyBuffer(buffer: GPUBuffer): void {
      const bucket = getPoolBucket(buffer.size);
      let pool = bufferPool.find((p) => p.size === bucket);
      if (!pool) {
        pool = { size: bucket, buffers: [] };
        bufferPool.push(pool);
      }
      if (pool.buffers.length < 16) {
        pool.buffers.push(buffer);
      } else {
        buffer.destroy();
      }
    },

    async createComputePipeline(shader: string, entryPoint: string): Promise<GPUComputePipeline> {
      const module = device.createShaderModule({ code: shader });
      return device.createComputePipeline({
        layout: "auto",
        compute: {
          module,
          entryPoint,
        },
      });
    },

    runPipeline(
      pipeline: GPUComputePipeline,
      bindGroups: GPUBindGroup[],
      workgroups: [number, number?, number?]
    ): void {
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      bindGroups.forEach((bg, i) => pass.setBindGroup(i, bg));
      pass.dispatchWorkgroups(workgroups[0], workgroups[1] ?? 1, workgroups[2] ?? 1);
      pass.end();
      queue.submit([encoder.finish()]);
    },
  };
}
