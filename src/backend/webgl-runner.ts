/**
 * WebGL2 runner - executes ops via render-to-texture
 */

import type { WebGLBackend, WebGLBuffer } from "./webgl-backend";
import {
  VERTEX_SHADER,
  ADD_FRAGMENT,
  MUL_FRAGMENT,
  MUL_SCALAR_FRAGMENT,
  REDUCE_SUM_FRAGMENT,
  REDUCE_MAX_FRAGMENT,
  MATMUL_FRAGMENT,
  SOFTMAX_FRAGMENT,
  LAYER_NORM_FRAGMENT,
  ATTENTION_SCORES_FRAGMENT,
} from "./webgl-shaders";

const QUAD_VERTS = new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]);

function compileProgram(gl: WebGL2RenderingContext, vs: string, fs: string): WebGLProgram {
  const vsh = gl.createShader(gl.VERTEX_SHADER)!;
  gl.shaderSource(vsh, vs);
  gl.compileShader(vsh);
  if (!gl.getShaderParameter(vsh, gl.COMPILE_STATUS)) {
    throw new Error("VS: " + gl.getShaderInfoLog(vsh));
  }
  const fsh = gl.createShader(gl.FRAGMENT_SHADER)!;
  gl.shaderSource(fsh, fs);
  gl.compileShader(fsh);
  if (!gl.getShaderParameter(fsh, gl.COMPILE_STATUS)) {
    throw new Error("FS: " + gl.getShaderInfoLog(fsh));
  }
  const prog = gl.createProgram()!;
  gl.attachShader(prog, vsh);
  gl.attachShader(prog, fsh);
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    throw new Error("Link: " + gl.getProgramInfoLog(prog));
  }
  return prog;
}

function runFragmentShader(
  gl: WebGL2RenderingContext,
  prog: WebGLProgram,
  outBuffer: WebGLBuffer,
  setup: (gl: WebGL2RenderingContext, prog: WebGLProgram) => void
): void {
  const fb = gl.createFramebuffer()!;
  gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
  gl.framebufferTexture2D(
    gl.FRAMEBUFFER,
    gl.COLOR_ATTACHMENT0,
    gl.TEXTURE_2D,
    outBuffer.texture,
    0
  );
  gl.viewport(0, 0, outBuffer.width, outBuffer.height);
  gl.useProgram(prog);
  setup(gl, prog);
  gl.drawArrays(gl.TRIANGLES, 0, 6);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.deleteFramebuffer(fb);
}

export class WebGLRunner {
  private backend: WebGLBackend;
  private gl: WebGL2RenderingContext;
  private quadVbo: WebGLBuffer | null = null;
  private programCache = new Map<string, WebGLProgram>();

  constructor(backend: WebGLBackend) {
    this.backend = backend;
    this.gl = backend.gl;
    this.initQuad();
  }

  private initQuad(): void {
    this.quadVbo = this.gl.createBuffer() as unknown as WebGLBuffer;
    this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.quadVbo as unknown as WebGLBuffer);
    this.gl.bufferData(this.gl.ARRAY_BUFFER, QUAD_VERTS, this.gl.STATIC_DRAW);
  }

  private getProgram(name: string, fragment: string): WebGLProgram {
    let prog = this.programCache.get(name);
    if (!prog) {
      prog = compileProgram(this.gl, VERTEX_SHADER, fragment);
      this.programCache.set(name, prog);
    }
    return prog;
  }

  private renderQuad(): void {
    const prog = this.gl.getParameter(this.gl.CURRENT_PROGRAM) as WebGLProgram;
    const posLoc = this.gl.getAttribLocation(prog, "a_position");
    this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.quadVbo as unknown as WebGLBuffer);
    this.gl.enableVertexAttribArray(posLoc);
    this.gl.vertexAttribPointer(posLoc, 2, this.gl.FLOAT, false, 0, 0);
  }

  async add(a: WebGLBuffer, b: WebGLBuffer, out: WebGLBuffer, length: number): Promise<void> {
    const prog = this.getProgram("add", ADD_FRAGMENT);
    runFragmentShader(this.gl, prog, out, (gl, p) => {
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, a.texture);
      gl.uniform1i(gl.getUniformLocation(p, "u_a"), 0);
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, b.texture);
      gl.uniform1i(gl.getUniformLocation(p, "u_b"), 1);
      gl.uniform2f(gl.getUniformLocation(p, "u_texSize"), a.width, a.height);
      gl.uniform1f(gl.getUniformLocation(p, "u_length"), length);
      this.renderQuad();
    });
  }

  async mul(a: WebGLBuffer, b: WebGLBuffer, out: WebGLBuffer, length: number): Promise<void> {
    const prog = this.getProgram("mul", MUL_FRAGMENT);
    runFragmentShader(this.gl, prog, out, (gl, p) => {
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, a.texture);
      gl.uniform1i(gl.getUniformLocation(p, "u_a"), 0);
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, b.texture);
      gl.uniform1i(gl.getUniformLocation(p, "u_b"), 1);
      gl.uniform2f(gl.getUniformLocation(p, "u_texSize"), a.width, a.height);
      gl.uniform1f(gl.getUniformLocation(p, "u_length"), length);
      this.renderQuad();
    });
  }

  async mulScalar(a: WebGLBuffer, scalar: number, out: WebGLBuffer, length: number): Promise<void> {
    const prog = this.getProgram("mul_scalar", MUL_SCALAR_FRAGMENT);
    runFragmentShader(this.gl, prog, out, (gl, p) => {
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, a.texture);
      gl.uniform1i(gl.getUniformLocation(p, "u_a"), 0);
      gl.uniform1f(gl.getUniformLocation(p, "u_scalar"), scalar);
      gl.uniform2f(gl.getUniformLocation(p, "u_texSize"), a.width, a.height);
      gl.uniform1f(gl.getUniformLocation(p, "u_length"), length);
      this.renderQuad();
    });
  }

  async reduceSum(input: WebGLBuffer, output: WebGLBuffer, length: number): Promise<void> {
    let current = input;
    let currentLen = length;
    const usage = 0;

    while (currentLen > 1) {
      const outLen = Math.ceil(currentLen / 2);
      const outBuf = this.backend.createBuffer(Math.max(4, outLen * 4), usage);
      const prog = this.getProgram("reduce_sum", REDUCE_SUM_FRAGMENT);
      runFragmentShader(this.gl, prog, outBuf, (gl, p) => {
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, current.texture);
        gl.uniform1i(gl.getUniformLocation(p, "u_input"), 0);
        gl.uniform2f(gl.getUniformLocation(p, "u_inputTexSize"), current.width, current.height);
        gl.uniform2f(gl.getUniformLocation(p, "u_outputTexSize"), outBuf.width, outBuf.height);
        gl.uniform1f(gl.getUniformLocation(p, "u_length"), currentLen);
        this.renderQuad();
      });
      if (current !== input) this.backend.destroyBuffer(current);
      current = outBuf;
      currentLen = outLen;
    }

    const data = new ArrayBuffer(4);
    await this.backend.readBuffer(current, data);
    if (current !== input) this.backend.destroyBuffer(current);
    this.backend.writeBuffer(output, data);
  }

  async reduceMax(input: WebGLBuffer, output: WebGLBuffer, length: number): Promise<void> {
    let current = input;
    let currentLen = length;
    const usage = 0;

    while (currentLen > 1) {
      const outLen = Math.ceil(currentLen / 2);
      const outBuf = this.backend.createBuffer(Math.max(4, outLen * 4), usage);
      const prog = this.getProgram("reduce_max", REDUCE_MAX_FRAGMENT);
      runFragmentShader(this.gl, prog, outBuf, (gl, p) => {
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, current.texture);
        gl.uniform1i(gl.getUniformLocation(p, "u_input"), 0);
        gl.uniform2f(gl.getUniformLocation(p, "u_inputTexSize"), current.width, current.height);
        gl.uniform2f(gl.getUniformLocation(p, "u_outputTexSize"), outBuf.width, outBuf.height);
        gl.uniform1f(gl.getUniformLocation(p, "u_length"), currentLen);
        this.renderQuad();
      });
      if (current !== input) this.backend.destroyBuffer(current);
      current = outBuf;
      currentLen = outLen;
    }

    const data = new ArrayBuffer(4);
    await this.backend.readBuffer(current, data);
    if (current !== input) this.backend.destroyBuffer(current);
    this.backend.writeBuffer(output, data);
  }

  async matmul(
    a: WebGLBuffer,
    b: WebGLBuffer,
    out: WebGLBuffer,
    M: number,
    N: number,
    K: number
  ): Promise<void> {
    const prog = this.getProgram("matmul", MATMUL_FRAGMENT);
    runFragmentShader(this.gl, prog, out, (gl, p) => {
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, a.texture);
      gl.uniform1i(gl.getUniformLocation(p, "u_a"), 0);
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, b.texture);
      gl.uniform1i(gl.getUniformLocation(p, "u_b"), 1);
      gl.uniform2f(gl.getUniformLocation(p, "u_texSizeA"), a.width, a.height);
      gl.uniform2f(gl.getUniformLocation(p, "u_texSizeB"), b.width, b.height);
      gl.uniform3f(gl.getUniformLocation(p, "u_params"), M, N, K);
      this.renderQuad();
    });
  }

  async softmax(
    input: WebGLBuffer,
    output: WebGLBuffer,
    rows: number,
    cols: number
  ): Promise<void> {
    const prog = this.getProgram("softmax", SOFTMAX_FRAGMENT);
    runFragmentShader(this.gl, prog, output, (gl, p) => {
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, input.texture);
      gl.uniform1i(gl.getUniformLocation(p, "u_input"), 0);
      gl.uniform2f(gl.getUniformLocation(p, "u_texSize"), input.width, input.height);
      gl.uniform2f(gl.getUniformLocation(p, "u_params"), rows, cols);
      this.renderQuad();
    });
  }

  async layerNorm(
    input: WebGLBuffer,
    gamma: WebGLBuffer,
    beta: WebGLBuffer,
    output: WebGLBuffer,
    rows: number,
    cols: number
  ): Promise<void> {
    const prog = this.getProgram("layer_norm", LAYER_NORM_FRAGMENT);
    runFragmentShader(this.gl, prog, output, (gl, p) => {
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, input.texture);
      gl.uniform1i(gl.getUniformLocation(p, "u_input"), 0);
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, gamma.texture);
      gl.uniform1i(gl.getUniformLocation(p, "u_gamma"), 1);
      gl.activeTexture(gl.TEXTURE2);
      gl.bindTexture(gl.TEXTURE_2D, beta.texture);
      gl.uniform1i(gl.getUniformLocation(p, "u_beta"), 2);
      gl.uniform2f(gl.getUniformLocation(p, "u_texSize"), input.width, input.height);
      gl.uniform2f(gl.getUniformLocation(p, "u_gammaTexSize"), gamma.width, gamma.height);
      gl.uniform2f(gl.getUniformLocation(p, "u_params"), rows, cols);
      this.renderQuad();
    });
  }

  async attentionScores(
    Q: WebGLBuffer,
    K: WebGLBuffer,
    output: WebGLBuffer,
    seq: number,
    dim: number
  ): Promise<void> {
    const prog = this.getProgram("attention_scores", ATTENTION_SCORES_FRAGMENT);
    runFragmentShader(this.gl, prog, output, (gl, p) => {
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, Q.texture);
      gl.uniform1i(gl.getUniformLocation(p, "u_Q"), 0);
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, K.texture);
      gl.uniform1i(gl.getUniformLocation(p, "u_K"), 1);
      gl.uniform2f(gl.getUniformLocation(p, "u_texSizeQ"), Q.width, Q.height);
      gl.uniform2f(gl.getUniformLocation(p, "u_texSizeK"), K.width, K.height);
      gl.uniform2f(gl.getUniformLocation(p, "u_params"), seq, dim);
      this.renderQuad();
    });
  }
}
