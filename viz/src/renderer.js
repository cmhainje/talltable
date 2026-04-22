import * as d3 from 'd3';

const TEX_WIDTH = 4096; // each texture row holds this many pixels

// ── Shaders ───────────────────────────────────────────────────────────────────

const VERT_SRC = /* glsl */`#version 300 es
void main() {
    // Full-screen quad from vertex ID — no vertex buffers needed.
    // Two triangles: (0,1,2) and (3,4,5) covering NDC [-1,1]².
    int id = gl_VertexID;
    float x = (id == 1 || id == 2 || id == 4) ? 1.0 : -1.0;
    float y = (id == 2 || id == 4 || id == 5) ? 1.0 : -1.0;
    gl_Position = vec4(x, y, 0.0, 1.0);
}`;

const FRAG_SRC = /* glsl */`#version 300 es
precision highp float;
precision highp int;
precision highp usampler2D;

uniform vec2       u_centerRaDec;     // tangent point (ra, dec) in radians
uniform float      u_scale;           // pixels per radian
uniform vec2       u_canvasSize;      // (width, height)
uniform int        u_nside;           // 2^maplvl
uniform int        u_order;           // maplvl
uniform vec2       u_fluxRange;       // (m, white) — m = black point (≥0), white = 98th pct
uniform float      u_alpha;           // arcsinh stretch: asinh(alpha*Q*(x-m)) / asinh(alpha*Q*(white-m))
uniform float      u_Q;               // stretch nonlinearity (0 → linear, large → log-like)
uniform int        u_pixelCount;
uniform int        u_indicesTexWidth; // = TEX_WIDTH
uniform usampler2D u_indicesTex;      // R32UI: sorted mappix values
uniform sampler2D  u_fluxesTex;       // R32F:  corresponding flux values
uniform sampler2D  u_colormapTex;     // RGB8 256×1: plasma colormap

out vec4 outColor;

const float PI     = 3.141592653589793;
const float TWO_PI = 6.283185307179586;
const vec4  BG     = vec4(0.039, 0.039, 0.059, 1.0); // #0a0a0f

// ── Gnomonic inverse ──────────────────────────────────────────────────────────
// Map screen pixel → (ra, dec) in radians.
// Canvas y grows downward; GL FragCoord y grows upward — no extra flip needed
// because FragCoord.y == 0 at the bottom of the canvas.
vec2 gnomonic_inverse(vec2 frag) {
    float x = -(frag.x - u_canvasSize.x * 0.5) / u_scale; // RA increases left (East left convention)
    float y =  (frag.y - u_canvasSize.y * 0.5) / u_scale; // FragCoord y=0 at bottom
    float rho2 = x*x + y*y;
    if (rho2 < 1e-20) return u_centerRaDec;
    float rho = sqrt(rho2);
    float c    = atan(rho);
    float sinc = sin(c);
    float cosc = cos(c);
    float sinP0 = sin(u_centerRaDec.y);
    float cosP0 = cos(u_centerRaDec.y);
    float dec = asin(cosc * sinP0 + y * (sinc * cosP0 / rho));
    float ra  = u_centerRaDec.x + atan(x * sinc, rho * cosP0 * cosc - y * sinP0 * sinc);
    return vec2(ra, dec);
}

// ── HEALPix NESTED ang2pix ────────────────────────────────────────────────────
// Returns a uvec2(lo, hi) representing the 64-bit NESTED pixel index.
// For levels ≤ 14 the index fits in 32 bits so hi == 0u always.
// theta = colatitude ∈ [0, π], phi = longitude ∈ [0, 2π).
uvec2 ang2pix_nest(float theta, float phi) {
    float z  = cos(theta);
    float za = abs(z);
    // tt ∈ [0, 4): phi normalised by π/2
    float tt = mod(phi * (2.0 / PI), 4.0);

    int  face_num;
    uint ix, iy;
    float ns = float(u_nside);

    if (za <= 0.6666666666666667) {
        // ── Equatorial belt ──
        float temp1 = ns * (0.5 + tt);
        float temp2 = ns * z * 0.75;
        uint jp = uint(max(0.0, floor(temp1 - temp2)));
        uint jm = uint(max(0.0, floor(temp1 + temp2)));
        int ifp = int(jp) / u_nside;
        int ifm = int(jm) / u_nside;
        face_num = (ifp == ifm) ? (ifp | 4)
                 : (ifp <  ifm) ?  ifp
                 :                 (ifm + 8);
        ix =  jm & uint(u_nside - 1);
        iy = uint(u_nside - 1) - (jp & uint(u_nside - 1));
    } else {
        // ── Polar caps ──
        float tt_floor = floor(tt);
        int   ntt = min(3, int(tt_floor));
        float tp  = tt - tt_floor;
        float tmp = ns * sqrt(3.0 * (1.0 - za));
        uint jp = uint(min(ns - 1.0, floor(tp         * tmp)));
        uint jm = uint(min(ns - 1.0, floor((1.0 - tp) * tmp)));
        if (z >= 0.0) {
            face_num = ntt;
            ix = uint(u_nside - 1) - jm;
            iy = uint(u_nside - 1) - jp;
        } else {
            face_num = ntt + 8;
            ix = jp;
            iy = jm;
        }
    }

    // Bit-interleave (ix, iy) → ipf (NESTED local index within face, 2*order bits)
    uint ipf = 0u;
    for (int k = 0; k < 30; k++) {
        if (k >= u_order) break;
        uint kb = uint(k);
        ipf |= ((ix >> kb) & 1u) << (2u * kb);
        ipf |= ((iy >> kb) & 1u) << (2u * kb + 1u);
    }

    // Pack into uvec2(lo, hi): full index = face_num * nside² + ipf
    // face_num << (2*order) may exceed 32 bits at order=15, so carry into hi.
    uint ipf_bits = uint(2 * u_order);
    uint lo = (uint(face_num) << ipf_bits) | ipf;
    uint hi = uint(face_num) >> (32u - ipf_bits);
    return uvec2(lo, hi);
}

// ── Binary search in 2-D texture ─────────────────────────────────────────────
// Indices are stored as RG32UI texels: r=lo, g=hi (little-endian uint64).
// Returns the row index of target in the sorted texture, or -1.
int findIndex(uvec2 target) {
    int lo = 0;
    int hi = u_pixelCount - 1;
    for (int step = 0; step < 32; step++) {
        if (lo > hi) return -1;
        int  mid = (lo + hi) >> 1;
        ivec2 xy  = ivec2(mid % u_indicesTexWidth, mid / u_indicesTexWidth);
        uvec2 val = texelFetch(u_indicesTex, xy, 0).rg;
        if (val == target) return mid;
        bool less = val.y < target.y || (val.y == target.y && val.x < target.x);
        if (less) lo = mid + 1;
        else      hi = mid - 1;
    }
    return -1;
}

void main() {
    // ── 1. Screen → sky ──
    vec2  radec = gnomonic_inverse(gl_FragCoord.xy);
    float ra    = radec.x;
    float dec   = radec.y;

    // Clip pixels more than 90° from the tangent point (gnomonic undefined)
    float cosc = sin(u_centerRaDec.y) * sin(dec)
               + cos(u_centerRaDec.y) * cos(dec) * cos(ra - u_centerRaDec.x);
    if (cosc <= 0.0) { outColor = BG; return; }

    // ── 2. Sky → HEALPix NESTED index ──
    float theta = PI * 0.5 - dec;
    float phi   = mod(ra, TWO_PI);
    if (phi < 0.0) phi += TWO_PI;
    uvec2 ipix = ang2pix_nest(theta, phi);

    // ── 3. Lookup flux ──
    int idx = findIndex(ipix);
    if (idx < 0) { outColor = BG; return; }

    ivec2 fxy  = ivec2(idx % u_indicesTexWidth, idx / u_indicesTexWidth);
    float flux = texelFetch(u_fluxesTex, fxy, 0).r;

    // ── 4. Arcsinh stretch + colormap ──
    // fluxRange.x = m (black point, ≥ 0), fluxRange.y = white point (98th pct)
    float x     = flux - u_fluxRange.x;
    float scale = asinh(u_alpha * u_Q * max(u_fluxRange.y - u_fluxRange.x, 1e-30));
    float t     = clamp(asinh(u_alpha * u_Q * x) / max(scale, 1e-30), 0.0, 1.0);
    outColor = vec4(texture(u_colormapTex, vec2(t, 0.5)).rgb, 1.0);
}`;

// ── WebGL helpers ─────────────────────────────────────────────────────────────

function compileShader(gl, type, src) {
  const sh = gl.createShader(type);
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(sh);
    gl.deleteShader(sh);
    throw new Error(`Shader compile error:\n${log}`);
  }
  return sh;
}

function createProgram(gl, vertSrc, fragSrc) {
  const vs   = compileShader(gl, gl.VERTEX_SHADER,   vertSrc);
  const fs   = compileShader(gl, gl.FRAGMENT_SHADER, fragSrc);
  const prog = gl.createProgram();
  gl.attachShader(prog, vs);
  gl.attachShader(prog, fs);
  gl.linkProgram(prog);
  gl.deleteShader(vs);
  gl.deleteShader(fs);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    const log = gl.getProgramInfoLog(prog);
    gl.deleteProgram(prog);
    throw new Error(`Program link error:\n${log}`);
  }
  return prog;
}

function createColormapTexture(gl) {
  const data = new Uint8Array(256 * 3);
  for (let i = 0; i < 256; i++) {
    const c = d3.color(d3.interpolatePlasma(i / 255));
    data[i * 3]     = c.r;
    data[i * 3 + 1] = c.g;
    data[i * 3 + 2] = c.b;
  }
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, 256, 1, 0, gl.RGB, gl.UNSIGNED_BYTE, data);
  return tex;
}

// ── Public API ────────────────────────────────────────────────────────────────

/**
 * Initialise WebGL2 on the given canvas.
 * Returns a renderer context object to pass to uploadPixels / render.
 * Throws if WebGL2 is unavailable or shaders fail to compile.
 */
export function init(canvas) {
  const gl = canvas.getContext('webgl2', { antialias: false });
  if (!gl) throw new Error('WebGL2 not supported');

  const program = createProgram(gl, VERT_SRC, FRAG_SRC);

  const uniformNames = [
    'u_centerRaDec', 'u_scale', 'u_canvasSize',
    'u_nside', 'u_order',
    'u_fluxRange', 'u_alpha', 'u_Q',
    'u_pixelCount', 'u_indicesTexWidth',
    'u_indicesTex', 'u_fluxesTex', 'u_colormapTex',
  ];
  const uniforms = {};
  for (const name of uniformNames) {
    uniforms[name] = gl.getUniformLocation(program, name);
  }

  // Empty VAO — we drive geometry purely from gl_VertexID
  const vao = gl.createVertexArray();

  const indicesTex  = gl.createTexture();
  const fluxesTex   = gl.createTexture();
  const colormapTex = createColormapTexture(gl);

  return {
    gl, program, uniforms, vao,
    indicesTex, fluxesTex, colormapTex,
    pixelCount: 0,
    texWidth:   TEX_WIDTH,
    fluxRange:  [0, 1],
  };
}

/**
 * Upload a new set of healpix pixels to the GPU.
 * mappix: Uint32Array of interleaved [lo0,hi0, lo1,hi1, ...] (2 u32 per pixel)
 * flux:   Float32Array of corresponding flux values (one per pixel)
 * Sorts by (hi,lo) so the fragment shader can binary-search.
 */
export function uploadPixels(ctx, mappix, flux) {
  const { gl, indicesTex, fluxesTex } = ctx;
  const n = flux.length;  // pixel count; mappix.length == 2*n

  if (n === 0) {
    ctx.pixelCount = 0;
    return;
  }

  // Sort by (hi, lo) ascending
  const order = new Uint32Array(n);
  for (let i = 0; i < n; i++) order[i] = i;
  order.sort((a, b) => {
    const aHi = mappix[a * 2 + 1], bHi = mappix[b * 2 + 1];
    if (aHi !== bHi) return aHi < bHi ? -1 : 1;
    const aLo = mappix[a * 2], bLo = mappix[b * 2];
    return aLo < bLo ? -1 : aLo > bLo ? 1 : 0;
  });

  const sortedIdx  = new Uint32Array(n * 2);  // interleaved lo/hi
  const sortedFlux = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const j = order[i];
    sortedIdx[i * 2]     = mappix[j * 2];      // lo
    sortedIdx[i * 2 + 1] = mappix[j * 2 + 1];  // hi
    sortedFlux[i] = flux[j];
  }

  // Flux range: black point = max(2nd pct, 0), white point = 98th pct
  const tmp = Float32Array.from(sortedFlux).sort();
  ctx.fluxRange = [
    Math.max(d3.quantile(tmp, 0.02) ?? 0, 0),
    d3.quantile(tmp, 0.98) ?? 1,
  ];

  // Pad to a full multiple of TEX_WIDTH rows
  const rows  = Math.ceil(n / TEX_WIDTH);
  const total = rows * TEX_WIDTH;

  const paddedIdx  = new Uint32Array(total * 2).fill(0xFFFFFFFF);
  const paddedFlux = new Float32Array(total).fill(0);
  paddedIdx.set(sortedIdx);
  paddedFlux.set(sortedFlux);

  // Indices texture: RG32UI — r=lo, g=hi (supports indices up to 2^64)
  gl.bindTexture(gl.TEXTURE_2D, indicesTex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RG32UI, TEX_WIDTH, rows, 0,
                gl.RG_INTEGER, gl.UNSIGNED_INT, paddedIdx);

  // Flux texture: R32F (unchanged)
  gl.bindTexture(gl.TEXTURE_2D, fluxesTex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, TEX_WIDTH, rows, 0,
                gl.RED, gl.FLOAT, paddedFlux);

  ctx.pixelCount = n;
}

/**
 * Render one frame.
 * ra, dec: tangent-point sky coordinates in degrees
 * scale:   pixels per radian
 * maplvl:  healpix order (nside = 2^maplvl)
 */
export function render(ctx, ra, dec, scale, maplvl, alpha, Q) {
  const { gl, program, uniforms, vao,
          indicesTex, fluxesTex, colormapTex,
          pixelCount, texWidth, fluxRange } = ctx;

  const W = gl.canvas.width;
  const H = gl.canvas.height;

  gl.viewport(0, 0, W, H);
  gl.useProgram(program);

  const DEG = Math.PI / 180;
  gl.uniform2f(uniforms.u_centerRaDec, ra * DEG, dec * DEG);
  gl.uniform1f(uniforms.u_scale,       scale);
  gl.uniform2f(uniforms.u_canvasSize,  W, H);
  gl.uniform1i(uniforms.u_nside,       1 << maplvl);
  gl.uniform1i(uniforms.u_order,       maplvl);
  gl.uniform2f(uniforms.u_fluxRange,   fluxRange[0], fluxRange[1]);
  gl.uniform1f(uniforms.u_alpha,       alpha);
  gl.uniform1f(uniforms.u_Q,           Q);
  gl.uniform1i(uniforms.u_pixelCount,  pixelCount);
  gl.uniform1i(uniforms.u_indicesTexWidth, texWidth);

  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, indicesTex);
  gl.uniform1i(uniforms.u_indicesTex, 0);

  gl.activeTexture(gl.TEXTURE1);
  gl.bindTexture(gl.TEXTURE_2D, fluxesTex);
  gl.uniform1i(uniforms.u_fluxesTex, 1);

  gl.activeTexture(gl.TEXTURE2);
  gl.bindTexture(gl.TEXTURE_2D, colormapTex);
  gl.uniform1i(uniforms.u_colormapTex, 2);

  gl.bindVertexArray(vao);
  gl.drawArrays(gl.TRIANGLES, 0, 6);
  gl.bindVertexArray(null);
}
