/**
 * WebGL2 GLSL fragment shaders for compute via render-to-texture
 * Uses RGBA8 with float bit-packing for universal framebuffer compatibility
 */

const PACK_UNPACK = `
vec4 packFloat(float f) {
  uint bits = floatBitsToUint(f);
  return vec4(
    float(bits & 0xffu) / 255.0,
    float((bits >> 8u) & 0xffu) / 255.0,
    float((bits >> 16u) & 0xffu) / 255.0,
    float((bits >> 24u) & 0xffu) / 255.0
  );
}
float unpackFloat(vec4 rgba) {
  uint r = uint(rgba.r * 255.0 + 0.5);
  uint g = uint(rgba.g * 255.0 + 0.5);
  uint b = uint(rgba.b * 255.0 + 0.5);
  uint a = uint(rgba.a * 255.0 + 0.5);
  uint bits = r | (g << 8u) | (b << 16u) | (a << 24u);
  return uintBitsToFloat(bits);
}
`;

export const VERTEX_SHADER = `#version 300 es
in vec2 a_position;
void main() {
  gl_Position = vec4(a_position, 0.0, 1.0);
}
`;

export const ADD_FRAGMENT = `#version 300 es
precision highp float;
${PACK_UNPACK}
uniform sampler2D u_a;
uniform sampler2D u_b;
uniform vec2 u_texSize;
uniform float u_length;
out vec4 fragColor;
void main() {
  float idx = floor(gl_FragCoord.y) * u_texSize.x + floor(gl_FragCoord.x);
  if (idx >= u_length) {
    fragColor = packFloat(0.0);
    return;
  }
  float aVal = unpackFloat(texelFetch(u_a, ivec2(gl_FragCoord.xy), 0));
  float bVal = unpackFloat(texelFetch(u_b, ivec2(gl_FragCoord.xy), 0));
  fragColor = packFloat(aVal + bVal);
}
`;

export const MUL_FRAGMENT = `#version 300 es
precision highp float;
${PACK_UNPACK}
uniform sampler2D u_a;
uniform sampler2D u_b;
uniform vec2 u_texSize;
uniform float u_length;
out vec4 fragColor;
void main() {
  float idx = floor(gl_FragCoord.y) * u_texSize.x + floor(gl_FragCoord.x);
  if (idx >= u_length) {
    fragColor = packFloat(0.0);
    return;
  }
  float aVal = unpackFloat(texelFetch(u_a, ivec2(gl_FragCoord.xy), 0));
  float bVal = unpackFloat(texelFetch(u_b, ivec2(gl_FragCoord.xy), 0));
  fragColor = packFloat(aVal * bVal);
}
`;

export const MUL_SCALAR_FRAGMENT = `#version 300 es
precision highp float;
${PACK_UNPACK}
uniform sampler2D u_a;
uniform float u_scalar;
uniform vec2 u_texSize;
uniform float u_length;
out vec4 fragColor;
void main() {
  float idx = floor(gl_FragCoord.y) * u_texSize.x + floor(gl_FragCoord.x);
  if (idx >= u_length) {
    fragColor = packFloat(0.0);
    return;
  }
  float aVal = unpackFloat(texelFetch(u_a, ivec2(gl_FragCoord.xy), 0));
  fragColor = packFloat(aVal * u_scalar);
}
`;

export const REDUCE_SUM_FRAGMENT = `#version 300 es
precision highp float;
${PACK_UNPACK}
uniform sampler2D u_input;
uniform vec2 u_inputTexSize;
uniform vec2 u_outputTexSize;
uniform float u_length;
out vec4 fragColor;
void main() {
  float outIdx = floor(gl_FragCoord.y) * u_outputTexSize.x + floor(gl_FragCoord.x);
  if (outIdx * 2.0 >= u_length) {
    fragColor = packFloat(0.0);
    return;
  }
  float inIdx0 = outIdx * 2.0;
  float inIdx1 = outIdx * 2.0 + 1.0;
  int x0 = int(mod(inIdx0, u_inputTexSize.x));
  int y0 = int(floor(inIdx0 / u_inputTexSize.x));
  float a = unpackFloat(texelFetch(u_input, ivec2(x0, y0), 0));
  float b = 0.0;
  if (inIdx1 < u_length) {
    int x1 = int(mod(inIdx1, u_inputTexSize.x));
    int y1 = int(floor(inIdx1 / u_inputTexSize.x));
    b = unpackFloat(texelFetch(u_input, ivec2(x1, y1), 0));
  }
  fragColor = packFloat(a + b);
}
`;

export const REDUCE_MAX_FRAGMENT = `#version 300 es
precision highp float;
${PACK_UNPACK}
uniform sampler2D u_input;
uniform vec2 u_inputTexSize;
uniform vec2 u_outputTexSize;
uniform float u_length;
out vec4 fragColor;
void main() {
  float outIdx = floor(gl_FragCoord.y) * u_outputTexSize.x + floor(gl_FragCoord.x);
  if (outIdx * 2.0 >= u_length) {
    fragColor = packFloat(-1e38);
    return;
  }
  float inIdx0 = outIdx * 2.0;
  float inIdx1 = outIdx * 2.0 + 1.0;
  int x0 = int(mod(inIdx0, u_inputTexSize.x));
  int y0 = int(floor(inIdx0 / u_inputTexSize.x));
  float a = unpackFloat(texelFetch(u_input, ivec2(x0, y0), 0));
  float b = -1e38;
  if (inIdx1 < u_length) {
    int x1 = int(mod(inIdx1, u_inputTexSize.x));
    int y1 = int(floor(inIdx1 / u_inputTexSize.x));
    b = unpackFloat(texelFetch(u_input, ivec2(x1, y1), 0));
  }
  fragColor = packFloat(max(a, b));
}
`;

export const MATMUL_FRAGMENT = `#version 300 es
precision highp float;
${PACK_UNPACK}
uniform sampler2D u_a;
uniform sampler2D u_b;
uniform vec2 u_texSizeA;
uniform vec2 u_texSizeB;
uniform vec3 u_params;
out vec4 fragColor;
void main() {
  float M = u_params.x;
  float N = u_params.y;
  float K = u_params.z;
  float i = floor(gl_FragCoord.y) * u_texSizeB.x + floor(gl_FragCoord.x);
  int row = int(floor(i / N));
  int col = int(mod(i, N));
  if (row >= int(M) || col >= int(N)) {
    fragColor = packFloat(0.0);
    return;
  }
  float sum = 0.0;
  for (int k = 0; k < 1024; k++) {
    if (float(k) >= K) break;
    float aIdx = float(row) * K + float(k);
    float bIdx = float(k) * N + float(col);
    int ax = int(mod(aIdx, u_texSizeA.x));
    int ay = int(floor(aIdx / u_texSizeA.x));
    int bx = int(mod(bIdx, u_texSizeB.x));
    int by = int(floor(bIdx / u_texSizeB.x));
    sum += unpackFloat(texelFetch(u_a, ivec2(ax, ay), 0)) * unpackFloat(texelFetch(u_b, ivec2(bx, by), 0));
  }
  fragColor = packFloat(sum);
}
`;

export const SOFTMAX_FRAGMENT = `#version 300 es
precision highp float;
${PACK_UNPACK}
uniform sampler2D u_input;
uniform vec2 u_texSize;
uniform vec2 u_params;
out vec4 fragColor;
void main() {
  float rows = u_params.x;
  float cols = u_params.y;
  float idx = floor(gl_FragCoord.y) * u_texSize.x + floor(gl_FragCoord.x);
  if (idx >= rows * cols) {
    fragColor = packFloat(0.0);
    return;
  }
  float row = floor(idx / cols);
  float col = mod(idx, cols);
  float maxVal = -1e38;
  for (float c = 0.0; c < cols; c += 1.0) {
    float i = row * cols + c;
    int ix = int(mod(i, u_texSize.x));
    int iy = int(floor(i / u_texSize.x));
    maxVal = max(maxVal, unpackFloat(texelFetch(u_input, ivec2(ix, iy), 0)));
  }
  float sumExp = 0.0;
  for (float c = 0.0; c < cols; c += 1.0) {
    float i = row * cols + c;
    int ix = int(mod(i, u_texSize.x));
    int iy = int(floor(i / u_texSize.x));
    sumExp += exp(unpackFloat(texelFetch(u_input, ivec2(ix, iy), 0)) - maxVal);
  }
  int selfX = int(mod(idx, u_texSize.x));
  int selfY = int(floor(idx / u_texSize.x));
  float selfVal = unpackFloat(texelFetch(u_input, ivec2(selfX, selfY), 0));
  fragColor = packFloat(exp(selfVal - maxVal) / sumExp);
}
`;

export const LAYER_NORM_FRAGMENT = `#version 300 es
precision highp float;
${PACK_UNPACK}
uniform sampler2D u_input;
uniform sampler2D u_gamma;
uniform sampler2D u_beta;
uniform vec2 u_texSize;
uniform vec2 u_gammaTexSize;
uniform vec2 u_params;
out vec4 fragColor;
void main() {
  float rows = u_params.x;
  float cols = u_params.y;
  float idx = floor(gl_FragCoord.y) * u_texSize.x + floor(gl_FragCoord.x);
  if (idx >= rows * cols) {
    fragColor = packFloat(0.0);
    return;
  }
  float row = floor(idx / cols);
  float col = mod(idx, cols);
  float sum = 0.0;
  for (float c = 0.0; c < cols; c += 1.0) {
    float i = row * cols + c;
    int ix = int(mod(i, u_texSize.x));
    int iy = int(floor(i / u_texSize.x));
    sum += unpackFloat(texelFetch(u_input, ivec2(ix, iy), 0));
  }
  float mean = sum / cols;
  float varSum = 0.0;
  for (float c = 0.0; c < cols; c += 1.0) {
    float i = row * cols + c;
    int ix = int(mod(i, u_texSize.x));
    int iy = int(floor(i / u_texSize.x));
    float d = unpackFloat(texelFetch(u_input, ivec2(ix, iy), 0)) - mean;
    varSum += d * d;
  }
  float variance = sqrt(varSum / cols + 1e-5);
  int gx = int(mod(col, u_gammaTexSize.x));
  int gy = int(floor(col / u_gammaTexSize.x));
  float gammaVal = unpackFloat(texelFetch(u_gamma, ivec2(gx, gy), 0));
  float betaVal = unpackFloat(texelFetch(u_beta, ivec2(gx, gy), 0));
  int selfX = int(mod(idx, u_texSize.x));
  int selfY = int(floor(idx / u_texSize.x));
  float selfVal = unpackFloat(texelFetch(u_input, ivec2(selfX, selfY), 0));
  float normalized = (selfVal - mean) / variance;
  fragColor = packFloat(normalized * gammaVal + betaVal);
}
`;

export const ATTENTION_SCORES_FRAGMENT = `#version 300 es
precision highp float;
${PACK_UNPACK}
uniform sampler2D u_Q;
uniform sampler2D u_K;
uniform vec2 u_texSizeQ;
uniform vec2 u_texSizeK;
uniform vec2 u_params;
out vec4 fragColor;
void main() {
  float seq = u_params.x;
  float dim = u_params.y;
  float scale = 1.0 / sqrt(dim);
  float idx = floor(gl_FragCoord.y) * u_texSizeK.x + floor(gl_FragCoord.x);
  if (idx >= seq * seq) {
    fragColor = packFloat(0.0);
    return;
  }
  float i = floor(idx / seq);
  float j = mod(idx, seq);
  float score = 0.0;
  for (float d = 0.0; d < dim; d += 1.0) {
    float qIdx = i * dim + d;
    float kIdx = j * dim + d;
    int qx = int(mod(qIdx, u_texSizeQ.x));
    int qy = int(floor(qIdx / u_texSizeQ.x));
    int kx = int(mod(kIdx, u_texSizeK.x));
    int ky = int(floor(kIdx / u_texSizeK.x));
    score += unpackFloat(texelFetch(u_Q, ivec2(qx, qy), 0)) * unpackFloat(texelFetch(u_K, ivec2(kx, ky), 0));
  }
  fragColor = packFloat(score * scale);
}
`;
