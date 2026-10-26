// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 Buntato
// HLSL Shader Model 5.0 compositor passes. Author: Buntato.
//
// These passes are the device side of src/compose.rs and match
// shaders/passes.glsl line for line. On Windows, src/gpu_d3d.rs compiles every
// entry point below to DXBC with d3dcompiler_47.dll (see PASS_ENTRY_POINTS in
// src/lib.rs), runs them on a Direct3D 11 device, and uses the device only if
// a self-test scene matches the software bytes. That path has been built and
// type-checked but not yet run on a Windows machine.
//
// Entry points:
// - VSPass: unit quad over the pass region.
// - PSShadowMask: rounded-box coverage into an R32_FLOAT mask.
// - PSBoxBlur: one box pass along Axis. Run three times along x, then three
//   times along y, ping-ponging two R32_FLOAT textures.
// - PSShadowPaint: paints ShadowColor through the blurred mask over Dest.
// - PSGradient: a linear or radial gradient over Dest.
// - PSBlitRgba, PSBlitNv12, PSBlitI420: scale and convert a video frame. These
//   replace the destination; they do not read Dest.
//
// Units: colors are 0..255 code values in constants and in the math, the same
// as the CPU code, and divided by 255 only on output. Every pass writes to a
// render target the size of its region. Row 0 is the first row in memory for
// every texture and render target, as in the GLSL file. SV_Position is the
// pixel center with a top-left origin, which is that same memory row, so
// int2(SV_Position.xy) is the pixel inside the region and Origin plus that
// pixel is the pixel in the full target. VSPass flips y for that reason. Dest holds a copy of
// the target pixels in the region. Blending is off: the passes that blend read
// Dest and do the straight-alpha source-over themselves.

#define MAX_STOPS 16

cbuffer PassConstants : register(b0)
{
    float4 Region;        // render-target pixels: x, y, width, height
    float2 ViewportSize;
    int2 Origin;          // region origin in full-target pixels
    float4 Shape;         // shadow box center xy, half size zw
    float4 ShadowColor;   // 0..255 straight
    float4 Geometry;      // gradient line or ellipse
    int4 DstRect;         // blit rectangle in full-target pixels
    int2 Axis;            // (1, 0) rows or (0, 1) columns
    int Radius;           // box pass radius
    int Margin;           // 3 * Radius
    float Corner;
    int Kind;             // 0 linear, 1 radial
    int StopCount;
    int FilterMode;       // 0 nearest, 1 bilinear
    float4 StopOffsets[MAX_STOPS / 4];
    float4 StopColors[MAX_STOPS];
};

Texture2D<float> Source : register(t0);
Texture2D<float> Mask : register(t1);
Texture2D<float4> Dest : register(t2);
Texture2D<float4> Frame : register(t3);
Texture2D<float> Luma : register(t4);
Texture2D<float2> ChromaUV : register(t5);
Texture2D<float> ChromaU : register(t6);
Texture2D<float> ChromaV : register(t7);

// Draw four vertices as a triangle strip. The corner of the unit quad comes
// from the vertex index, so no vertex buffer or input layout is needed.
float4 VSPass(uint id : SV_VertexID) : SV_Position
{
    float2 corner = float2(float(id & 1), float(id >> 1));
    float2 p = Region.xy + corner * Region.zw;
    float2 ndc = (p / ViewportSize) * 2.0 - 1.0;
    ndc.y = -ndc.y;
    return float4(ndc, 0.0, 1.0);
}

float2 TargetCenter(int2 p)
{
    return float2(p + Origin) + 0.5;
}

// Same as sdf_rounded_box in src/lib.rs.
float SdfRoundedBox(float2 p, float2 halfSize, float radius)
{
    float r = max(min(radius, min(halfSize.x, halfSize.y)), 0.0);
    float2 q = abs(p) - (halfSize - float2(r, r));
    return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r;
}

// Same as over_straight in src/compose.rs. dst and src are 0..255.
float4 OverStraight(float4 dst, float4 src, float coverage)
{
    float sa = (src.a / 255.0) * saturate(coverage);
    if (sa <= 0.0)
    {
        return dst;
    }
    float da = dst.a / 255.0;
    float keep = da * (1.0 - sa);
    float oa = sa + keep;
    float3 rgb = clamp(floor((src.rgb * sa + dst.rgb * keep) / oa + 0.5), 0.0, 255.0);
    return float4(rgb, clamp(floor(oa * 255.0 + 0.5), 0.0, 255.0));
}

// Same as bt709_yuv_f32_to_rgb in src/compose.rs. Inputs are 8-bit codes.
float3 Bt709ToRgb(float y, float u, float v)
{
    float yf = max(y - 16.0, 0.0) * 1.164384;
    float uf = u - 128.0;
    float vf = v - 128.0;
    float3 rgb = float3(yf + 1.792741 * vf, yf - 0.213249 * uf - 0.532909 * vf, yf + 2.112402 * uf);
    return clamp(floor(rgb + 0.5), 0.0, 255.0);
}

float4 LoadDest(int2 p)
{
    return floor(Dest.Load(int3(p, 0)) * 255.0 + 0.5);
}

float4 PSShadowMask(float4 pos : SV_Position) : SV_Target
{
    int2 p = int2(pos.xy);
    float d = SdfRoundedBox(TargetCenter(p) - Shape.xy, Shape.zw, Corner);
    return float4(saturate(0.5 - d), 0.0, 0.0, 1.0);
}

// MASK_ONE in src/compose.rs: mask values are whole multiples of 1 / 32768,
// so the sums below are exact integers on every device.
static const float MaskOne = 32768.0;

float4 PSBoxBlur(float4 pos : SV_Position) : SV_Target
{
    int2 p = int2(pos.xy);
    uint w, h;
    Source.GetDimensions(w, h);
    int2 size = int2(w, h);
    int acc = 0;
    // Texels outside the mask count as zero, as in box_blur_line.
    [loop]
    for (int k = -Radius; k <= Radius; ++k)
    {
        int2 q = p + Axis * k;
        if (q.x >= 0 && q.y >= 0 && q.x < size.x && q.y < size.y)
        {
            // HLSL round() rounds halfway cases to even, as roundEven.
            acc += (int)round(Source.Load(int3(q, 0)) * MaskOne);
        }
    }
    // Mean rounded half up, as BoxRun::emit.
    int width = 2 * Radius + 1;
    int mean = (2 * acc + width) / (2 * width);
    return float4(float(mean) / MaskOne, 0.0, 0.0, 1.0);
}

float4 PSShadowPaint(float4 pos : SV_Position) : SV_Target
{
    int2 p = int2(pos.xy);
    float coverage = Mask.Load(int3(p + int2(Margin, Margin), 0));
    return OverStraight(LoadDest(p), ShadowColor, coverage) / 255.0;
}

// Offsets are packed four to a register.
float StopOffset(int i)
{
    float4 v = StopOffsets[i >> 2];
    int c = i & 3;
    return c == 0 ? v.x : (c == 1 ? v.y : (c == 2 ? v.z : v.w));
}

// Same as sample_multi_stop_gradient in src/lib.rs, including the rounding
// of each mix to whole codes.
float4 SampleStops(float t)
{
    if (StopCount == 1)
    {
        return StopColors[0];
    }
    float c = saturate(t);
    if (c <= StopOffset(0))
    {
        return StopColors[0];
    }
    [loop]
    for (int i = 1; i < MAX_STOPS; ++i)
    {
        if (i >= StopCount)
        {
            break;
        }
        if (c <= StopOffset(i))
        {
            float span = max(StopOffset(i) - StopOffset(i - 1), 1e-5);
            float local = saturate((c - StopOffset(i - 1)) / span);
            return floor(StopColors[i - 1] * (1.0 - local) + StopColors[i] * local + 0.5);
        }
    }
    return StopColors[StopCount - 1];
}

float4 PSGradient(float4 pos : SV_Position) : SV_Target
{
    int2 p = int2(pos.xy);
    float2 c = TargetCenter(p);
    float t;
    if (Kind == 0)
    {
        // Same as linear_gradient_t in src/compose.rs.
        float2 d = Geometry.zw - Geometry.xy;
        float len2 = dot(d, d);
        t = len2 < 1e-12 ? 1.0 : saturate(dot(c - Geometry.xy, d) / len2);
    }
    else
    {
        // Same as sample_radial_gradient in src/lib.rs.
        float2 n = (c - Geometry.xy) / max(Geometry.zw, float2(0.5, 0.5));
        t = saturate(length(n));
    }
    return OverStraight(LoadDest(p), SampleStops(t), 1.0) / 255.0;
}

// Nearest texel along one axis, integer math as in nearest_index.
int NearestIndex(int d, int src, int dst)
{
    return min(((2 * d + 1) * src) / (2 * dst), src - 1);
}

// Bilinear taps along one axis, integer math as in fixed_taps: the two
// clamped texel indices and the weight of the second, in 0..4095. The
// division is split so every value stays below 2^31; the result is the same
// as round((2d + 1) * src * 4096 / (2 dst)), halves up.
int3 FixedTaps(int d, int src, int dst, int size)
{
    int n = (2 * d + 1) * src;
    int den = 2 * dst;
    int q = n / den;
    int p = q * 4096 + (2 * (n - q * den) * 4096 + den) / (2 * den) + 2048;
    int i = p / 4096 - 1;
    return int3(clamp(i, 0, size - 1), clamp(i + 1, 0, size - 1), p - (p / 4096) * 4096);
}

// As bilinear_rgba: texels tl, tr, bl, br as 0..255 codes. Rows first, in
// premultiplied color rounded to 1/16 steps, then the two rows; alpha
// exactly. Every value stays below 2^32.
uint4 BilinearRgba(uint4 tl, uint4 tr, uint4 bl, uint4 br, uint wx, uint wy)
{
    uint ix = 4096u - wx;
    uint iy = 4096u - wy;
    uint3 top = (tl.rgb * (tl.a * ix) + tr.rgb * (tr.a * wx) + 128u) >> 8;
    uint3 bottom = (bl.rgb * (bl.a * ix) + br.rgb * (br.a * wx) + 128u) >> 8;
    uint alpha = (tl.a * ix + tr.a * wx) * iy + (bl.a * ix + br.a * wx) * wy;
    uint alpha16 = (alpha + 128u) >> 8;
    if (alpha16 == 0u)
    {
        return uint4(0u, 0u, 0u, 0u);
    }
    uint3 rgb = min((top * iy + bottom * wy + alpha16 / 2u) / alpha16, uint3(255u, 255u, 255u));
    return uint4(rgb, min((alpha + 8388608u) >> 24, 255u));
}

// As bilinear_plane: four codes (tl, tr, bl, br) to one value in 1/256 steps.
uint BilinearPlane(uint4 t, uint wx, uint wy)
{
    uint ix = 4096u - wx;
    uint iy = 4096u - wy;
    // At most 255 * 2^24, which fits in 32 bits.
    uint sum = t.x * (ix * iy) + t.y * (wx * iy) + t.z * (ix * wy) + t.w * (wx * wy);
    return (sum + 32768u) >> 16;
}

// As bt709_fixed_to_rgb in src/compose.rs: code values in 1/256 steps,
// integer coefficients in 1/8192 steps, rounded half up.
float3 Bt709FixedToRgb(int y, int u, int v)
{
    int yf = max(y - 16 * 256, 0) * 9539;
    int uf = u - 128 * 256;
    int vf = v - 128 * 256;
    int3 s = max(int3(yf + 14686 * vf, yf - 1747 * uf - 4366 * vf, yf + 17305 * uf), int3(0, 0, 0));
    return float3(min((s + (1 << 20)) >> 21, int3(255, 255, 255)));
}

float4 FrameTexel(int x, int y)
{
    return floor(Frame.Load(int3(x, y, 0)) * 255.0 + 0.5);
}

float4 PSBlitRgba(float4 pos : SV_Position) : SV_Target
{
    uint w, h;
    Frame.GetDimensions(w, h);
    int2 size = int2(w, h);
    int2 l = int2(pos.xy) + Origin - DstRect.xy;
    float4 outColor;
    if (FilterMode == 0)
    {
        outColor = FrameTexel(NearestIndex(l.x, size.x, DstRect.z), NearestIndex(l.y, size.y, DstRect.w));
    }
    else
    {
        int3 tx = FixedTaps(l.x, size.x, DstRect.z, size.x);
        int3 ty = FixedTaps(l.y, size.y, DstRect.w, size.y);
        uint4 c = BilinearRgba(uint4(FrameTexel(tx.x, ty.x)), uint4(FrameTexel(tx.y, ty.x)),
                               uint4(FrameTexel(tx.x, ty.y)), uint4(FrameTexel(tx.y, ty.y)), uint(tx.z), uint(ty.z));
        outColor = float4(c);
    }
    return outColor / 255.0;
}

float LumaTexel(int x, int y)
{
    return floor(Luma.Load(int3(x, y, 0)) * 255.0 + 0.5);
}

// planar is 0 for NV12 (ChromaUV) and 1 for I420 (ChromaU and ChromaV).
float2 ChromaTexel(int x, int y, int planar)
{
    if (planar == 0)
    {
        return floor(ChromaUV.Load(int3(x, y, 0)) * 255.0 + 0.5);
    }
    return floor(float2(ChromaU.Load(int3(x, y, 0)), ChromaV.Load(int3(x, y, 0))) * 255.0 + 0.5);
}

float4 BlitYuv(float4 pos, int planar)
{
    uint w, h;
    Luma.GetDimensions(w, h);
    int2 size = int2(w, h);
    int2 csize = (size + 1) / 2;
    int2 l = int2(pos.xy) + Origin - DstRect.xy;
    float3 rgb;
    if (FilterMode == 0)
    {
        int tx = NearestIndex(l.x, size.x, DstRect.z);
        int ty = NearestIndex(l.y, size.y, DstRect.w);
        float2 uv = ChromaTexel(tx / 2, ty / 2, planar);
        rgb = Bt709ToRgb(LumaTexel(tx, ty), uv.x, uv.y);
    }
    else
    {
        int3 tx = FixedTaps(l.x, size.x, DstRect.z, size.x);
        int3 ty = FixedTaps(l.y, size.y, DstRect.w, size.y);
        uint y = BilinearPlane(uint4(LumaTexel(tx.x, ty.x), LumaTexel(tx.y, ty.x), LumaTexel(tx.x, ty.y), LumaTexel(tx.y, ty.y)),
                               uint(tx.z), uint(ty.z));
        // Chroma texels are twice as wide, so their centers sit at the middle
        // of each 2 by 2 luma block.
        int3 cx = FixedTaps(l.x, size.x, 2 * DstRect.z, csize.x);
        int3 cy = FixedTaps(l.y, size.y, 2 * DstRect.w, csize.y);
        float2 c0 = ChromaTexel(cx.x, cy.x, planar);
        float2 c1 = ChromaTexel(cx.y, cy.x, planar);
        float2 c2 = ChromaTexel(cx.x, cy.y, planar);
        float2 c3 = ChromaTexel(cx.y, cy.y, planar);
        uint u = BilinearPlane(uint4(c0.x, c1.x, c2.x, c3.x), uint(cx.z), uint(cy.z));
        uint v = BilinearPlane(uint4(c0.y, c1.y, c2.y, c3.y), uint(cx.z), uint(cy.z));
        rgb = Bt709FixedToRgb(int(y), int(u), int(v));
    }
    return float4(rgb, 255.0) / 255.0;
}

float4 PSBlitNv12(float4 pos : SV_Position) : SV_Target
{
    return BlitYuv(pos, 0);
}

float4 PSBlitI420(float4 pos : SV_Position) : SV_Target
{
    return BlitYuv(pos, 1);
}
