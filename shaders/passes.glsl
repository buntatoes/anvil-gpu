// SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0-OR-Commercial
// Copyright (c) 2026 Buntato
// GLSL 330 core compositor passes. Author: Buntato.
//
// These passes are the device side of src/compose.rs. They do the same math
// in the same order. The shadow blur and bilinear blits are integer math, so
// they give the software bytes on any device; the f32 parts give them on a
// device that does not fuse or reorder those operations. On Linux, src/gpu_gl.rs builds one program per pass in an OpenGL
// 3.3 core context and draws a self-test scene; the device is only used when
// that scene matches the software bytes. See README.md for what runs where.
//
// Compile once per pass with VERTEX_SHADER, or with FRAGMENT_SHADER plus one
// PASS_* symbol:
// - PASS_SHADOW_MASK: rounded-box coverage into an R32F mask.
// - PASS_BOX_BLUR: one box pass over that mask along uAxis. Run it three
//   times along x, then three times along y, ping-ponging two R32F textures.
// - PASS_SHADOW_PAINT: paints uColor through the blurred mask over uDest.
// - PASS_GRADIENT: a linear or radial gradient over uDest.
// - PASS_BLIT_RGBA, PASS_BLIT_NV12, PASS_BLIT_I420: scale and convert a
//   video frame. These replace the destination; they do not read uDest.
//
// Units: colors are 0..255 code values in uniforms and in the math, the same
// as the CPU code, and divided by 255 only on output. Every pass writes to a
// render target the size of its region. Rows are never flipped: row 0 is the
// first row in memory for every texture and render target, and with the
// default lower-left gl_FragCoord origin, ivec2(gl_FragCoord.xy) is that same
// memory pixel. Upload the target with its top row first and read it back the
// same way. uOrigin + the region pixel is the pixel in the full target. uDest
// holds a copy of the target pixels in the region (straight RGBA8). There is
// no fixed-function blending: the passes that blend read uDest and do the
// straight-alpha source-over themselves, which fixed-function blending cannot
// do for a target that is not opaque.
//
// Rounding: every place this file rounds uses floor(x + 0.5), and the CPU
// code it mirrors does the same. Rust's f32::round is not used for those,
// because it differs for 0.5 - 2^-25: round gives 0, while x + 0.5 rounds up
// to 1.0 in f32.

#version 330 core

#ifdef VERTEX_SHADER
// Region in render-target pixels: x, y, width, height. Usually 0, 0, size.
uniform vec4 uRegion;
uniform vec2 uViewportSize;

void main()
{
    // Draw four vertices as a triangle strip. The corner of the unit quad
    // comes from the vertex index, so no vertex buffer is needed.
    vec2 corner = vec2(float(gl_VertexID & 1), float(gl_VertexID >> 1));
    // Unit quad to pixels, then to NDC. No y flip: window row y is memory row y.
    vec2 p = uRegion.xy + corner * uRegion.zw;
    gl_Position = vec4((p / uViewportSize) * 2.0 - 1.0, 0.0, 1.0);
}
#endif

#ifdef FRAGMENT_SHADER
out vec4 FragColor;

// Region origin in full-target pixels. May be negative for the shadow mask.
uniform ivec2 uOrigin;

ivec2 RegionPixel()
{
    return ivec2(gl_FragCoord.xy);
}

// Center of this fragment in full-target pixels.
vec2 TargetCenter()
{
    return vec2(RegionPixel() + uOrigin) + 0.5;
}

// Splitting `x` and `y` into 12-bit halves gives the exact residual `a - x * y`
// in 24-bit floats, and one refinement step rounds `a / b` and `sqrt(x)` to
// 0.5 ULP as on the CPU.
float ExactMulSub(float a, float x, float y)
{
    float xh = uintBitsToFloat(floatBitsToUint(x) & 0xfffff000u);
    float xl = x - xh;
    float yh = uintBitsToFloat(floatBitsToUint(y) & 0xfffff000u);
    float yl = y - yh;
    float p1 = xh * yh;
    float p2 = xh * yl + xl * yh;
    float p3 = xl * yl;
    return ((a - p1) - p2) - p3;
}

float IeeeDiv(float a, float b)
{
    if (a == 0.0)
    {
        return 0.0;
    }
    float q0 = a / b;
    float rem = ExactMulSub(a, q0, b);
    return q0 + rem / b;
}

vec2 IeeeDiv2(vec2 a, vec2 b)
{
    return vec2(IeeeDiv(a.x, b.x), IeeeDiv(a.y, b.y));
}

vec3 IeeeDiv3(vec3 a, float b)
{
    return vec3(IeeeDiv(a.x, b), IeeeDiv(a.y, b), IeeeDiv(a.z, b));
}

float IeeeSqrt(float x)
{
    if (x <= 0.0)
    {
        return 0.0;
    }
    float s0 = sqrt(x);
    float rem = ExactMulSub(x, s0, s0);
    return s0 + rem / (2.0 * s0);
}

// Same as sdf_rounded_box in src/lib.rs.
float SdfRoundedBox(vec2 p, vec2 halfSize, float radius)
{
    float r = max(min(radius, min(halfSize.x, halfSize.y)), 0.0);
    vec2 q = abs(p) - (halfSize - vec2(r));
    vec2 qm = max(q, 0.0);
    float outside = IeeeSqrt(qm.x * qm.x + qm.y * qm.y);
    float inside = min(max(q.x, q.y), 0.0);
    return (outside + inside) - r;
}

// Same as over_straight in src/compose.rs. dst and src are 0..255.
vec4 OverStraight(vec4 dst, vec4 src, float coverage)
{
    float sa = IeeeDiv(src.a, 255.0) * clamp(coverage, 0.0, 1.0);
    if (sa <= 0.0)
    {
        return dst;
    }
    float da = IeeeDiv(dst.a, 255.0);
    float keep = da * (1.0 - sa);
    float oa = sa + keep;
    vec3 num = src.rgb * sa + dst.rgb * keep;
    vec3 rgb = clamp(floor(IeeeDiv3(num, oa) + 0.5), 0.0, 255.0);
    return vec4(rgb, clamp(floor(oa * 255.0 + 0.5), 0.0, 255.0));
}

// Same as bt709_yuv_f32_to_rgb in src/compose.rs. Inputs are 8-bit codes.
vec3 Bt709ToRgb(float y, float u, float v)
{
    float yf = max(y - 16.0, 0.0) * 1.164384;
    float uf = u - 128.0;
    float vf = v - 128.0;
    vec3 rgb = vec3(yf + 1.792741 * vf, (yf - 0.213249 * uf) - 0.532909 * vf, yf + 2.112402 * uf);
    return clamp(floor(rgb + 0.5), 0.0, 255.0);
}

#ifdef PASS_SHADOW_MASK
// Shadow box after offset and spread: center xy, half size zw.
uniform vec4 uShape;
uniform float uCorner;

void main()
{
    float d = SdfRoundedBox(TargetCenter() - uShape.xy, uShape.zw, uCorner);
    FragColor = vec4(clamp(0.5 - d, 0.0, 1.0), 0.0, 0.0, 1.0);
}
#endif

#ifdef PASS_BOX_BLUR
uniform sampler2D uSource;
// (1, 0) for a row pass, (0, 1) for a column pass.
uniform ivec2 uAxis;
// Pass radius from shadow_pass_radius in src/compose.rs.
uniform int uRadius;

// MASK_ONE in src/compose.rs: mask values are whole multiples of 1 / 32768,
// so the sums below are exact integers on every device.
const float MASK_ONE = 32768.0;

void main()
{
    ivec2 p = RegionPixel();
    ivec2 size = textureSize(uSource, 0);
    int acc = 0;
    // Texels outside the mask count as zero, as in box_blur_line.
    for (int k = -uRadius; k <= uRadius; ++k)
    {
        ivec2 q = p + uAxis * k;
        if (q.x >= 0 && q.y >= 0 && q.x < size.x && q.y < size.y)
        {
            acc += int(roundEven(texelFetch(uSource, q, 0).r * MASK_ONE));
        }
    }
    // Mean rounded half up, as BoxRun::emit.
    int w = 2 * uRadius + 1;
    int mean = (2 * acc + w) / (2 * w);
    FragColor = vec4(float(mean) / MASK_ONE, 0.0, 0.0, 1.0);
}
#endif

#ifdef PASS_SHADOW_PAINT
uniform sampler2D uMask;
uniform sampler2D uDest;
// Shadow color, 0..255 straight.
uniform vec4 uColor;
// Mask texels between the mask edge and the paint region: 3 * radius.
uniform int uMargin;

void main()
{
    ivec2 p = RegionPixel();
    vec4 dst = floor(texelFetch(uDest, p, 0) * 255.0 + 0.5);
    float coverage = texelFetch(uMask, p + ivec2(uMargin), 0).r;
    FragColor = OverStraight(dst, uColor, coverage) / 255.0;
}
#endif

#ifdef PASS_GRADIENT
const int MAX_STOPS = 16;
uniform sampler2D uDest;
// 0: linear from uGeometry.xy to uGeometry.zw.
// 1: radial, center uGeometry.xy, radii uGeometry.zw.
uniform int uKind;
uniform vec4 uGeometry;
uniform int uStopCount;
// Offsets already clamped and ordered, as Stops::new does on the CPU.
uniform float uStopOffsets[MAX_STOPS];
// Colors, 0..255 straight.
uniform vec4 uStopColors[MAX_STOPS];

// Same as sample_multi_stop_gradient in src/lib.rs, including the rounding
// of each mix to whole codes.
vec4 SampleStops(float t)
{
    if (uStopCount == 1)
    {
        return uStopColors[0];
    }
    float c = clamp(t, 0.0, 1.0);
    if (c <= uStopOffsets[0])
    {
        return uStopColors[0];
    }
    for (int i = 1; i < MAX_STOPS; ++i)
    {
        if (i >= uStopCount)
        {
            break;
        }
        if (c <= uStopOffsets[i])
        {
            float prevOff = uStopOffsets[i - 1];
            float span = max(uStopOffsets[i] - prevOff, 1e-5);
            float local = clamp(IeeeDiv(c - prevOff, span), 0.0, 1.0);
            return floor(uStopColors[i - 1] * (1.0 - local) + uStopColors[i] * local + 0.5);
        }
    }
    return uStopColors[uStopCount - 1];
}

void main()
{
    ivec2 p = RegionPixel();
    vec2 c = TargetCenter();
    float t;
    if (uKind == 0)
    {
        // Same as linear_gradient_t in src/compose.rs.
        vec2 d = uGeometry.zw - uGeometry.xy;
        float len2 = d.x * d.x + d.y * d.y;
        vec2 rel = c - uGeometry.xy;
        float proj = rel.x * d.x + rel.y * d.y;
        t = len2 < 1e-12 ? 1.0 : clamp(IeeeDiv(proj, len2), 0.0, 1.0);
    }
    else
    {
        // Same as sample_radial_gradient in src/lib.rs.
        vec2 n = IeeeDiv2(c - uGeometry.xy, max(uGeometry.zw, vec2(0.5)));
        t = clamp(IeeeSqrt(n.x * n.x + n.y * n.y), 0.0, 1.0);
    }
    vec4 dst = floor(texelFetch(uDest, p, 0) * 255.0 + 0.5);
    FragColor = OverStraight(dst, SampleStops(t), 1.0) / 255.0;
}
#endif

#if defined(PASS_BLIT_RGBA) || defined(PASS_BLIT_NV12) || defined(PASS_BLIT_I420)
// Blit rectangle in full-target pixels: x, y, width, height.
uniform ivec4 uDstRect;
// 0: nearest, 1: bilinear.
uniform int uFilter;

// Nearest texel along one axis, integer math as in nearest_index.
int NearestIndex(int d, int src, int dst)
{
    return min(((2 * d + 1) * src) / (2 * dst), src - 1);
}

// Bilinear taps along one axis, integer math as in fixed_taps: the two
// clamped texel indices and the weight of the second, in 0..4095. The
// division is split so every value stays below 2^31; the result is the same
// as round((2d + 1) * src * 4096 / (2 dst)), halves up.
ivec3 FixedTaps(int d, int src, int dst, int size)
{
    int n = (2 * d + 1) * src;
    int den = 2 * dst;
    int q = n / den;
    int p = q * 4096 + (2 * (n - q * den) * 4096 + den) / (2 * den) + 2048;
    int i = p / 4096 - 1;
    return ivec3(clamp(i, 0, size - 1), clamp(i + 1, 0, size - 1), p - (p / 4096) * 4096);
}

// As bilinear_rgba: texels tl, tr, bl, br as 0..255 codes. Rows first, in
// premultiplied color rounded to 1/16 steps, then the two rows; alpha
// exactly. Every value stays below 2^32.
uvec4 BilinearRgba(uvec4 tl, uvec4 tr, uvec4 bl, uvec4 br, uint wx, uint wy)
{
    uint ix = 4096u - wx;
    uint iy = 4096u - wy;
    uvec3 top = (tl.rgb * (tl.a * ix) + tr.rgb * (tr.a * wx) + 128u) >> 8;
    uvec3 bottom = (bl.rgb * (bl.a * ix) + br.rgb * (br.a * wx) + 128u) >> 8;
    uint alpha = (tl.a * ix + tr.a * wx) * iy + (bl.a * ix + br.a * wx) * wy;
    uint alpha16 = (alpha + 128u) >> 8;
    if (alpha16 == 0u)
    {
        return uvec4(0u, 0u, 0u, 0u);
    }
    uvec3 rgb = min((top * iy + bottom * wy + alpha16 / 2u) / alpha16, uvec3(255u, 255u, 255u));
    return uvec4(rgb, min((alpha + 8388608u) >> 24, 255u));
}

// As bilinear_plane: four codes (tl, tr, bl, br) to one value in 1/256 steps.
uint BilinearPlane(uvec4 t, uint wx, uint wy)
{
    uint ix = 4096u - wx;
    uint iy = 4096u - wy;
    // At most 255 * 2^24, which fits in 32 bits.
    uint sum = t.x * (ix * iy) + t.y * (wx * iy) + t.z * (ix * wy) + t.w * (wx * wy);
    return (sum + 32768u) >> 16;
}

// Destination pixel relative to the blit rectangle.
ivec2 LocalPixel()
{
    return RegionPixel() + uOrigin - uDstRect.xy;
}
#endif

#ifdef PASS_BLIT_RGBA
uniform sampler2D uFrame;

vec4 Texel(int x, int y)
{
    return floor(texelFetch(uFrame, ivec2(x, y), 0) * 255.0 + 0.5);
}

void main()
{
    ivec2 size = textureSize(uFrame, 0);
    ivec2 l = LocalPixel();
    vec4 outColor;
    if (uFilter == 0)
    {
        outColor = Texel(NearestIndex(l.x, size.x, uDstRect.z), NearestIndex(l.y, size.y, uDstRect.w));
    }
    else
    {
        ivec3 tx = FixedTaps(l.x, size.x, uDstRect.z, size.x);
        ivec3 ty = FixedTaps(l.y, size.y, uDstRect.w, size.y);
        uvec4 c = BilinearRgba(uvec4(Texel(tx.x, ty.x)), uvec4(Texel(tx.y, ty.x)),
                               uvec4(Texel(tx.x, ty.y)), uvec4(Texel(tx.y, ty.y)), uint(tx.z), uint(ty.z));
        outColor = vec4(c);
    }
    FragColor = outColor / 255.0;
}
#endif

#if defined(PASS_BLIT_NV12) || defined(PASS_BLIT_I420)
uniform sampler2D uLuma;
#ifdef PASS_BLIT_NV12
// Interleaved Cb, Cr in the red and green channels.
uniform sampler2D uChroma;
#else
uniform sampler2D uChromaU;
uniform sampler2D uChromaV;
#endif

float Luma(int x, int y)
{
    return floor(texelFetch(uLuma, ivec2(x, y), 0).r * 255.0 + 0.5);
}

vec2 Chroma(int x, int y)
{
#ifdef PASS_BLIT_NV12
    return floor(texelFetch(uChroma, ivec2(x, y), 0).rg * 255.0 + 0.5);
#else
    return floor(vec2(texelFetch(uChromaU, ivec2(x, y), 0).r, texelFetch(uChromaV, ivec2(x, y), 0).r) * 255.0 + 0.5);
#endif
}

// As bt709_fixed_to_rgb in src/compose.rs: code values in 1/256 steps,
// integer coefficients in 1/8192 steps, rounded half up.
vec3 Bt709FixedToRgb(int y, int u, int v)
{
    int yf = max(y - 16 * 256, 0) * 9539;
    int uf = u - 128 * 256;
    int vf = v - 128 * 256;
    ivec3 s = max(ivec3(yf + 14686 * vf, yf - 1747 * uf - 4366 * vf, yf + 17305 * uf), ivec3(0));
    return vec3(min((s + (1 << 20)) >> 21, ivec3(255)));
}

void main()
{
    ivec2 size = textureSize(uLuma, 0);
    ivec2 csize = (size + 1) / 2;
    ivec2 l = LocalPixel();
    vec3 rgb;
    if (uFilter == 0)
    {
        int tx = NearestIndex(l.x, size.x, uDstRect.z);
        int ty = NearestIndex(l.y, size.y, uDstRect.w);
        vec2 uv = Chroma(tx / 2, ty / 2);
        rgb = Bt709ToRgb(Luma(tx, ty), uv.x, uv.y);
    }
    else
    {
        ivec3 tx = FixedTaps(l.x, size.x, uDstRect.z, size.x);
        ivec3 ty = FixedTaps(l.y, size.y, uDstRect.w, size.y);
        uint y = BilinearPlane(uvec4(Luma(tx.x, ty.x), Luma(tx.y, ty.x), Luma(tx.x, ty.y), Luma(tx.y, ty.y)),
                               uint(tx.z), uint(ty.z));
        // Chroma texels are twice as wide, so their centers sit at the middle
        // of each 2 by 2 luma block.
        ivec3 cx = FixedTaps(l.x, size.x, 2 * uDstRect.z, csize.x);
        ivec3 cy = FixedTaps(l.y, size.y, 2 * uDstRect.w, csize.y);
        vec2 c0 = Chroma(cx.x, cy.x);
        vec2 c1 = Chroma(cx.y, cy.x);
        vec2 c2 = Chroma(cx.x, cy.y);
        vec2 c3 = Chroma(cx.y, cy.y);
        uint u = BilinearPlane(uvec4(c0.x, c1.x, c2.x, c3.x), uint(cx.z), uint(cy.z));
        uint v = BilinearPlane(uvec4(c0.y, c1.y, c2.y, c3.y), uint(cx.z), uint(cy.z));
        rgb = Bt709FixedToRgb(int(y), int(u), int(v));
    }
    FragColor = vec4(rgb, 255.0) / 255.0;
}
#endif
#endif
