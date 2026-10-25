// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 Buntato
// GLSL 330 core compositor passes. Author: Buntato.
//
// These passes are the device side of src/compose.rs. They do the same math
// in the same order, so a device that runs them matches the software results
// within one step per channel. Nothing in this repository runs them yet: the
// crate embeds this text with include_str! and the Linux probe does not create
// a GL context. See README.md for what runs where and how they were checked.
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
// Rounding: Rust f32::round and floor(x + 0.5) agree on every value that is
// later clamped to 0..255, which is every place this file rounds.

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

// Same as sdf_rounded_box in src/lib.rs.
float SdfRoundedBox(vec2 p, vec2 halfSize, float radius)
{
    float r = max(min(radius, min(halfSize.x, halfSize.y)), 0.0);
    vec2 q = abs(p) - (halfSize - vec2(r));
    return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r;
}

// Same as over_straight in src/compose.rs. dst and src are 0..255.
vec4 OverStraight(vec4 dst, vec4 src, float coverage)
{
    float sa = (src.a / 255.0) * clamp(coverage, 0.0, 1.0);
    if (sa <= 0.0)
    {
        return dst;
    }
    float da = dst.a / 255.0;
    float keep = da * (1.0 - sa);
    float oa = sa + keep;
    vec3 rgb = clamp(floor((src.rgb * sa + dst.rgb * keep) / oa + 0.5), 0.0, 255.0);
    return vec4(rgb, clamp(floor(oa * 255.0 + 0.5), 0.0, 255.0));
}

// Same as bt709_yuv_f32_to_rgb in src/compose.rs. Inputs are 8-bit codes.
vec3 Bt709ToRgb(float y, float u, float v)
{
    float yf = max(y - 16.0, 0.0) * 1.164384;
    float uf = u - 128.0;
    float vf = v - 128.0;
    vec3 rgb = vec3(yf + 1.792741 * vf, yf - 0.213249 * uf - 0.532909 * vf, yf + 2.112402 * uf);
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
            float span = max(uStopOffsets[i] - uStopOffsets[i - 1], 1e-5);
            float local = clamp((c - uStopOffsets[i - 1]) / span, 0.0, 1.0);
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
        float len2 = dot(d, d);
        t = len2 < 1e-12 ? 1.0 : clamp(dot(c - uGeometry.xy, d) / len2, 0.0, 1.0);
    }
    else
    {
        // Same as sample_radial_gradient in src/lib.rs.
        vec2 n = (c - uGeometry.xy) / max(uGeometry.zw, vec2(0.5));
        t = clamp(length(n), 0.0, 1.0);
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

// Two clamped taps and the weight of the second, as in taps.
vec3 Taps(float f, int size)
{
    float fl = floor(f);
    float last = float(size - 1);
    return vec3(clamp(fl, 0.0, last), clamp(fl + 1.0, 0.0, last), f - fl);
}

// Destination pixel relative to the blit rectangle.
ivec2 LocalPixel()
{
    return RegionPixel() + uOrigin - uDstRect.xy;
}

// Texel-space position of the pixel center for bilinear sampling.
vec2 SourcePos(ivec2 l, ivec2 srcSize)
{
    return (vec2(l) + 0.5) * (vec2(srcSize) / vec2(uDstRect.zw)) - 0.5;
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
        vec2 f = SourcePos(l, size);
        vec3 tx = Taps(f.x, size.x);
        vec3 ty = Taps(f.y, size.y);
        vec4 w = vec4((1.0 - tx.z) * (1.0 - ty.z), tx.z * (1.0 - ty.z), (1.0 - tx.z) * ty.z, tx.z * ty.z);
        vec4 s0 = Texel(int(tx.x), int(ty.x));
        vec4 s1 = Texel(int(tx.y), int(ty.x));
        vec4 s2 = Texel(int(tx.x), int(ty.y));
        vec4 s3 = Texel(int(tx.y), int(ty.y));
        // Alpha-weighted, as on the CPU.
        vec4 a = vec4(s0.a, s1.a, s2.a, s3.a) * w;
        float alpha = a.x + a.y + a.z + a.w;
        if (alpha <= 0.0)
        {
            outColor = vec4(0.0);
        }
        else
        {
            vec3 rgb = (s0.rgb * a.x + s1.rgb * a.y + s2.rgb * a.z + s3.rgb * a.w) / alpha;
            outColor = vec4(clamp(floor(rgb + 0.5), 0.0, 255.0), clamp(floor(alpha + 0.5), 0.0, 255.0));
        }
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

void main()
{
    ivec2 size = textureSize(uLuma, 0);
    ivec2 csize = (size + 1) / 2;
    ivec2 l = LocalPixel();
    float y;
    vec2 uv;
    if (uFilter == 0)
    {
        int tx = NearestIndex(l.x, size.x, uDstRect.z);
        int ty = NearestIndex(l.y, size.y, uDstRect.w);
        y = Luma(tx, ty);
        uv = Chroma(tx / 2, ty / 2);
    }
    else
    {
        vec2 f = SourcePos(l, size);
        vec3 tx = Taps(f.x, size.x);
        vec3 ty = Taps(f.y, size.y);
        vec4 w = vec4((1.0 - tx.z) * (1.0 - ty.z), tx.z * (1.0 - ty.z), (1.0 - tx.z) * ty.z, tx.z * ty.z);
        y = Luma(int(tx.x), int(ty.x)) * w.x + Luma(int(tx.y), int(ty.x)) * w.y
          + Luma(int(tx.x), int(ty.y)) * w.z + Luma(int(tx.y), int(ty.y)) * w.w;
        // Chroma centers sit at the middle of each 2 by 2 luma block.
        vec3 cx = Taps((f.x + 0.5) * 0.5 - 0.5, csize.x);
        vec3 cy = Taps((f.y + 0.5) * 0.5 - 0.5, csize.y);
        vec4 cw = vec4((1.0 - cx.z) * (1.0 - cy.z), cx.z * (1.0 - cy.z), (1.0 - cx.z) * cy.z, cx.z * cy.z);
        uv = Chroma(int(cx.x), int(cy.x)) * cw.x + Chroma(int(cx.y), int(cy.x)) * cw.y
           + Chroma(int(cx.x), int(cy.y)) * cw.z + Chroma(int(cx.y), int(cy.y)) * cw.w;
    }
    FragColor = vec4(Bt709ToRgb(y, uv.x, uv.y), 255.0) / 255.0;
}
#endif
#endif
