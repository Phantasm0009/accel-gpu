/**
 * Embedded WGSL compute shaders - zero runtime file loading
 */

export const ADD_SHADER = /* wgsl */ `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i < arrayLength(&out)) {
    out[i] = a[i] + b[i];
  }
}
`;

export const MUL_SHADER = /* wgsl */ `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i < arrayLength(&out)) {
    out[i] = a[i] * b[i];
  }
}
`;

export const MUL_SCALAR_SHADER = /* wgsl */ `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<uniform> scalar: f32;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i < arrayLength(&out)) {
    out[i] = a[i] * scalar;
  }
}
`;

export const SUB_SHADER = /* wgsl */ `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i < arrayLength(&out)) {
    out[i] = a[i] - b[i];
  }
}
`;

export const SUB_SCALAR_SHADER = /* wgsl */ `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<uniform> scalar: f32;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i < arrayLength(&out)) {
    out[i] = a[i] - scalar;
  }
}
`;

export const DIV_SHADER = /* wgsl */ `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i < arrayLength(&out)) {
    out[i] = select(0.0, a[i] / b[i], b[i] != 0.0);
  }
}
`;

export const DIV_SCALAR_SHADER = /* wgsl */ `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<uniform> scalar: f32;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i < arrayLength(&out)) {
    out[i] = select(0.0, a[i] / scalar, scalar != 0.0);
  }
}
`;

export const POW_SCALAR_SHADER = /* wgsl */ `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<uniform> exponent: f32;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i < arrayLength(&out)) {
    out[i] = pow(a[i], exponent);
  }
}
`;

export const SQRT_SHADER = /* wgsl */ `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i < arrayLength(&out)) {
    out[i] = sqrt(a[i]);
  }
}
`;

export const ABS_SHADER = /* wgsl */ `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i < arrayLength(&out)) {
    out[i] = abs(a[i]);
  }
}
`;

export const NEG_SHADER = /* wgsl */ `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i < arrayLength(&out)) {
    out[i] = -a[i];
  }
}
`;

export const EXP_SHADER = /* wgsl */ `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i < arrayLength(&out)) {
    out[i] = exp(a[i]);
  }
}
`;

export const LOG_SHADER = /* wgsl */ `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i < arrayLength(&out)) {
    out[i] = select(-1e38, log(a[i]), a[i] > 0.0);
  }
}
`;

export const RELU_SHADER = /* wgsl */ `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i < arrayLength(&out)) {
    out[i] = max(0.0, a[i]);
  }
}
`;

export const SIGMOID_SHADER = /* wgsl */ `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i < arrayLength(&out)) {
    out[i] = 1.0 / (1.0 + exp(-a[i]));
  }
}
`;

export const TANH_SHADER = /* wgsl */ `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i < arrayLength(&out)) {
    out[i] = tanh(a[i]);
  }
}
`;

export const CLAMP_SHADER = /* wgsl */ `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<uniform> params: vec2<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i < arrayLength(&out)) {
    out[i] = clamp(a[i], params.x, params.y);
  }
}
`;

export const GELU_SHADER = /* wgsl */ `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;

let C: f32 = 0.044715;
let SQRT_2_OVER_PI: f32 = 0.7978845608;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i < arrayLength(&out)) {
    let x = a[i];
    out[i] = 0.5 * x * (1.0 + tanh(SQRT_2_OVER_PI * (x + C * x * x * x)));
  }
}
`;

export const LEAKY_RELU_SHADER = /* wgsl */ `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<uniform> alpha: f32;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i < arrayLength(&out)) {
    out[i] = select(alpha * a[i], a[i], a[i] >= 0.0);
  }
}
`;

export const EQUAL_SHADER = /* wgsl */ `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i < arrayLength(&out)) {
    out[i] = select(0.0, 1.0, a[i] == b[i]);
  }
}
`;

export const GREATER_SHADER = /* wgsl */ `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i < arrayLength(&out)) {
    out[i] = select(0.0, 1.0, a[i] > b[i]);
  }
}
`;

export const LESS_SHADER = /* wgsl */ `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i < arrayLength(&out)) {
    out[i] = select(0.0, 1.0, a[i] < b[i]);
  }
}
`;

export const REDUCE_SUM_SHADER = /* wgsl */ `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

var<workgroup> shared: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>
) {
  let idx = gid.x;
  let localIdx = lid.x;
  
  if (idx < arrayLength(&input)) {
    shared[localIdx] = input[idx];
  } else {
    shared[localIdx] = 0.0;
  }
  workgroupBarrier();

  var stride = 128u;
  loop {
    if (localIdx < stride) {
      shared[localIdx] = shared[localIdx] + shared[localIdx + stride];
    }
    workgroupBarrier();
    stride = stride / 2u;
    if (stride == 0u) {
      break;
    }
  }

  if (localIdx == 0u) {
    output[wid.x] = shared[0];
  }
}
`;

export const REDUCE_MAX_SHADER = /* wgsl */ `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

var<workgroup> shared: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>
) {
  let idx = gid.x;
  let localIdx = lid.x;
  
  if (idx < arrayLength(&input)) {
    shared[localIdx] = input[idx];
  } else {
    shared[localIdx] = -1e38;
  }
  workgroupBarrier();

  var stride = 128u;
  loop {
    if (localIdx < stride) {
      shared[localIdx] = max(shared[localIdx], shared[localIdx + stride]);
    }
    workgroupBarrier();
    stride = stride / 2u;
    if (stride == 0u) {
      break;
    }
  }

  if (localIdx == 0u) {
    output[wid.x] = shared[0];
  }
}
`;

export const REDUCE_MIN_SHADER = /* wgsl */ `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

var<workgroup> shared: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>
) {
  let idx = gid.x;
  let localIdx = lid.x;
  
  if (idx < arrayLength(&input)) {
    shared[localIdx] = input[idx];
  } else {
    shared[localIdx] = 1e38;
  }
  workgroupBarrier();

  var stride = 128u;
  loop {
    if (localIdx < stride) {
      shared[localIdx] = min(shared[localIdx], shared[localIdx + stride]);
    }
    workgroupBarrier();
    stride = stride / 2u;
    if (stride == 0u) {
      break;
    }
  }

  if (localIdx == 0u) {
    output[wid.x] = shared[0];
  }
}
`;

export const MATMUL_SHADER = /* wgsl */ `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: vec3<u32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let M = params.x;
  let N = params.y;
  let K = params.z;
  
  let i = gid.x;
  let j = gid.y;
  
  if (i < M && j < N) {
    var sum = 0.0;
    for (var k = 0u; k < K; k++) {
      sum += a[i * K + k] * b[k * N + j];
    }
    out[i * N + j] = sum;
  }
}
`;

export const SOFTMAX_SHADER = /* wgsl */ `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: vec2<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let rows = params.x;
  let cols = params.y;
  let row = gid.x;
  
  if (row >= rows) {
    return;
  }
  
  var maxVal = -1e38;
  for (var c = 0u; c < cols; c++) {
    maxVal = max(maxVal, input[row * cols + c]);
  }
  
  var sumExp = 0.0;
  for (var c = 0u; c < cols; c++) {
    let e = exp(input[row * cols + c] - maxVal);
    output[row * cols + c] = e;
    sumExp += e;
  }
  
  for (var c = 0u; c < cols; c++) {
    output[row * cols + c] = output[row * cols + c] / sumExp;
  }
}
`;

export const LAYER_NORM_SHADER = /* wgsl */ `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> gamma: array<f32>;
@group(0) @binding(2) var<storage, read> beta: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: vec2<u32>;
let eps = 1e-5;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let rows = params.x;
  let cols = params.y;
  let row = gid.x;
  
  if (row >= rows) { return; }
  
  var sum = 0.0;
  for (var c = 0u; c < cols; c++) {
    sum += input[row * cols + c];
  }
  let mean = sum / f32(cols);
  
  var varSum = 0.0;
  for (var c = 0u; c < cols; c++) {
    let d = input[row * cols + c] - mean;
    varSum += d * d;
  }
  let variance = sqrt(varSum / f32(cols) + eps);
  
  for (var c = 0u; c < cols; c++) {
    let normalized = (input[row * cols + c] - mean) / variance;
    output[row * cols + c] = normalized * gamma[c] + beta[c];
  }
}
`;

export const ATTENTION_SCORES_SHADER = /* wgsl */ `
@group(0) @binding(0) var<storage, read> Q: array<f32>;
@group(0) @binding(1) var<storage, read> K: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: vec3<u32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let seq = params.x;
  let dim = params.y;
  let scale = 1.0 / sqrt(f32(dim));
  let i = gid.x;
  let j = gid.y;
  
  if (i >= seq || j >= seq) { return; }
  
  var score = 0.0;
  for (var d = 0u; d < dim; d++) {
    score += Q[i * dim + d] * K[j * dim + d];
  }
  output[i * seq + j] = score * scale;
}
`;

export const CONV2D_SHADER = /* wgsl */ `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> kernel: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: array<u32, 12>;

@compute @workgroup_size(4, 4, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = params[0];
  let H = params[1];
  let W = params[2];
  let C_in = params[3];
  let kH = params[4];
  let kW = params[5];
  let C_out = params[6];
  let outH = params[7];
  let outW = params[8];
  let stride = params[9];
  let padding = params[10];

  let ox = gid.x;
  let oy = gid.y;
  let nz = gid.z;
  let ni = nz / C_out;
  let co = nz % C_out;

  if (ni >= n || oy >= outH || ox >= outW || co >= C_out) {
    return;
  }

  var sum = 0.0;
  for (var ky = 0u; ky < kH; ky++) {
    for (var kx = 0u; kx < kW; kx++) {
      let iy = i32(oy * stride + ky) - i32(padding);
      let ix = i32(ox * stride + kx) - i32(padding);
      if (iy >= 0 && iy < i32(H) && ix >= 0 && ix < i32(W)) {
        for (var ci = 0u; ci < C_in; ci++) {
          let inIdx = ni * H * W * C_in + u32(iy) * W * C_in + u32(ix) * C_in + ci;
          let kIdx = ky * kW * C_in * C_out + kx * C_in * C_out + ci * C_out + co;
          sum += input[inIdx] * kernel[kIdx];
        }
      }
    }
  }

  let outIdx = ni * outH * outW * C_out + oy * outW * C_out + ox * C_out + co;
  output[outIdx] = sum;
}
`;

export const MAX_POOL2D_SHADER = /* wgsl */ `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: array<u32, 10>;

@compute @workgroup_size(4, 4, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = params[0];
  let H = params[1];
  let W = params[2];
  let C = params[3];
  let kernelSize = params[4];
  let stride = params[5];
  let padding = params[6];
  let outH = params[7];
  let outW = params[8];

  let ox = gid.x;
  let oy = gid.y;
  let nz = gid.z;
  let ni = nz / C;
  let ci = nz % C;

  if (ni >= n || oy >= outH || ox >= outW || ci >= C) {
    return;
  }

  var maxVal = -1e38;
  for (var ky = 0u; ky < kernelSize; ky++) {
    for (var kx = 0u; kx < kernelSize; kx++) {
      let iy = i32(oy * stride + ky) - i32(padding);
      let ix = i32(ox * stride + kx) - i32(padding);
      if (iy >= 0 && iy < i32(H) && ix >= 0 && ix < i32(W)) {
        let inIdx = ni * H * W * C + u32(iy) * W * C + u32(ix) * C + ci;
        maxVal = max(maxVal, input[inIdx]);
      }
    }
  }

  let outIdx = ni * outH * outW * C + oy * outW * C + ox * C + ci;
  output[outIdx] = select(0.0, maxVal, maxVal > -1e37);
}
`;

export const AVG_POOL2D_SHADER = /* wgsl */ `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: array<u32, 10>;

@compute @workgroup_size(4, 4, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = params[0];
  let H = params[1];
  let W = params[2];
  let C = params[3];
  let kernelSize = params[4];
  let stride = params[5];
  let padding = params[6];
  let outH = params[7];
  let outW = params[8];

  let ox = gid.x;
  let oy = gid.y;
  let nz = gid.z;
  let ni = nz / C;
  let ci = nz % C;

  if (ni >= n || oy >= outH || ox >= outW || ci >= C) {
    return;
  }

  var sum = 0.0;
  var count = 0u;
  for (var ky = 0u; ky < kernelSize; ky++) {
    for (var kx = 0u; kx < kernelSize; kx++) {
      let iy = i32(oy * stride + ky) - i32(padding);
      let ix = i32(ox * stride + kx) - i32(padding);
      if (iy >= 0 && iy < i32(H) && ix >= 0 && ix < i32(W)) {
        let inIdx = ni * H * W * C + u32(iy) * W * C + u32(ix) * C + ci;
        sum += input[inIdx];
        count += 1u;
      }
    }
  }

  let outIdx = ni * outH * outW * C + oy * outW * C + ox * C + ci;
  output[outIdx] = select(0.0, sum / f32(count), count > 0u);
}
`;

export const FFT_DFT_SHADER = /* wgsl */ `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: vec2<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = params.x;
  let inverse = params.y;
  let k = gid.x;
  if (k >= n) {
    return;
  }

  let sign = select(-1.0, 1.0, inverse == 1u);
  var real = 0.0;
  var imag = 0.0;
  for (var t = 0u; t < n; t++) {
    let angle = sign * 6.283185307179586 * f32(k * t) / f32(n);
    let c = cos(angle);
    let s = sin(angle);
    real += input[t] * c;
    imag += input[t] * s;
  }

  if (inverse == 1u) {
    real = real / f32(n);
    imag = imag / f32(n);
  }

  output[2u * k] = real;
  output[2u * k + 1u] = imag;
}
`;

export const IFFT_DFT_SHADER = /* wgsl */ `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: vec2<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = params.x;
  let k = gid.x;
  if (k >= n) {
    return;
  }

  var real = 0.0;
  var imag = 0.0;
  for (var t = 0u; t < n; t++) {
    let angle = 6.283185307179586 * f32(k * t) / f32(n);
    let c = cos(angle);
    let s = sin(angle);
    let xr = input[2u * t];
    let xi = input[2u * t + 1u];
    real += xr * c - xi * s;
    imag += xr * s + xi * c;
  }

  output[2u * k] = real / f32(n);
  output[2u * k + 1u] = imag / f32(n);
}
`;

export const FFT_MAGNITUDE_SHADER = /* wgsl */ `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i < arrayLength(&output)) {
    let r = input[2u * i];
    let im = input[2u * i + 1u];
    output[i] = sqrt(r * r + im * im);
  }
}
`;
