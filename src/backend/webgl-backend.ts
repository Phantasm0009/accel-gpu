/**
 * WebGL2 backend - render-to-texture compute for environments without WebGPU
 */

export interface WebGLBuffer {
  texture: WebGLTexture;
  width: number;
  height: number;
  length: number;
}

function getTextureDimensions(length: number): { width: number; height: number } {
  const pixels = Math.max(1, length);
  const maxSize = 4096;
  let width = Math.ceil(Math.sqrt(pixels));
  let height = Math.ceil(pixels / width);
  if (width > maxSize) width = maxSize;
  if (height > maxSize) height = maxSize;
  return { width, height };
}

export interface WebGLBackend {
  type: "webgl";
  gl: WebGL2RenderingContext;
  canvas: HTMLCanvasElement;
  createBuffer(size: number, _usage?: number): WebGLBuffer;
  createBufferFromData(data: ArrayBuffer, _usage?: number): WebGLBuffer;
  writeBuffer(buffer: WebGLBuffer, data: ArrayBuffer): void;
  readBuffer(buffer: WebGLBuffer, output: ArrayBuffer): Promise<void>;
  destroyBuffer(buffer: WebGLBuffer): void;
}

export function createWebGLBackend(): WebGLBackend {
  const canvas = document.createElement("canvas");
  canvas.width = 4096;
  canvas.height = 4096;

  const glOrNull = canvas.getContext("webgl2", {
    preserveDrawingBuffer: true,
    antialias: false,
    depth: false,
    stencil: false,
  });

  if (!glOrNull) {
    throw new Error("WebGL2 is not supported");
  }

  const gl = glOrNull;

  const texturePool: WebGLBuffer[] = [];

  function createTexture(width: number, height: number, length: number): WebGLBuffer {
    const texture = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, width, height, 0, gl.RED, gl.FLOAT, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.bindTexture(gl.TEXTURE_2D, null);
    return { texture, width, height, length };
  }

  return {
    type: "webgl",
    gl,
    canvas,

    createBuffer(size: number, _usage?: number): WebGLBuffer {
      const length = size / 4;
      const { width, height } = getTextureDimensions(length);
      const reused = texturePool.find((b) => b.width === width && b.height === height && b.length >= length);
      if (reused) {
        texturePool.splice(texturePool.indexOf(reused), 1);
        reused.length = length;
        return reused;
      }
      return createTexture(width, height, length);
    },

    createBufferFromData(data: ArrayBuffer, _usage?: number): WebGLBuffer {
      const floats = new Float32Array(data);
      const length = floats.length;
      const { width, height } = getTextureDimensions(length);
      const buf = createTexture(width, height, length);
      const padded = new Float32Array(width * height);
      padded.set(floats);
      gl.bindTexture(gl.TEXTURE_2D, buf.texture);
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, width, height, gl.RED, gl.FLOAT, padded);
      gl.bindTexture(gl.TEXTURE_2D, null);
      return buf;
    },

    writeBuffer(buffer: WebGLBuffer, data: ArrayBuffer): void {
      const floats = new Float32Array(data);
      gl.bindTexture(gl.TEXTURE_2D, buffer.texture);
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, buffer.width, buffer.height, gl.RED, gl.FLOAT, floats);
      gl.bindTexture(gl.TEXTURE_2D, null);
    },

    async readBuffer(buffer: WebGLBuffer, output: ArrayBuffer): Promise<void> {
      const size = buffer.length * 4;
      const fb = gl.createFramebuffer()!;
      gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, buffer.texture, 0);
      const pixels = new Float32Array(buffer.length);
      gl.readPixels(0, 0, buffer.width, buffer.height, gl.RED, gl.FLOAT, pixels);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.deleteFramebuffer(fb);
      new Float32Array(output).set(pixels.subarray(0, output.byteLength / 4));
    },

    destroyBuffer(buffer: WebGLBuffer): void {
      if (texturePool.length < 32) {
        texturePool.push(buffer);
      } else {
        gl.deleteTexture(buffer.texture);
      }
    },
  };
}
