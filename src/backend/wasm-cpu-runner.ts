/**
 * WASM CPU runner (experimental).
 *
 * This runner currently keeps CPURunner semantics and records whether a
 * provided wasm module could be instantiated. It is a forward-compatible
 * path for progressively moving CPU kernels to WebAssembly.
 */

import { CPURunner } from "./cpu-runner";

export interface WasmCPUOptions {
  wasmModule?: WebAssembly.Module | ArrayBuffer;
}

export class WasmCPURunner extends CPURunner {
  readonly engine: "wasm" | "js";

  private constructor(engine: "wasm" | "js") {
    super();
    this.engine = engine;
  }

  static async create(options?: WasmCPUOptions): Promise<WasmCPURunner> {
    if (typeof WebAssembly === "undefined") {
      return new WasmCPURunner("js");
    }

    if (!options?.wasmModule) {
      return new WasmCPURunner("js");
    }

    try {
      const module =
        options.wasmModule instanceof ArrayBuffer
          ? await WebAssembly.compile(options.wasmModule)
          : options.wasmModule;

      await WebAssembly.instantiate(module, {});
      return new WasmCPURunner("wasm");
    } catch {
      return new WasmCPURunner("js");
    }
  }
}
