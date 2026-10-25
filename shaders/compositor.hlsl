// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 Buntato
// HLSL Shader Model 5.0 source text. Author: Buntato.
//
// Nothing in this repository compiles this file or creates a Direct3D device.
// The crate embeds this text with include_str!. CPU code in src/lib.rs draws
// rounded boxes, box shadows, and linear gradients, shelf-packs an RGBA atlas,
// and converts RGBA and limited-range BT.709 NV12. anvil-raster calls those CPU
// SDF helpers. anvil-video uses Nv12Surface for CPU color conversion.
//
// Entry points (not invoked from this repository):
// - VSMain: expands a unit quad into the rect, flips y into clip space, and
//   forwards colors, clip, UV, and params.
// - PSMain: rejects fragments outside the clip, evaluates the rounded-box SDF,
//   and writes a shadow, a solid fill, or a linear gradient. It can multiply
//   that color by an atlas sample or by an NV12 BT.709 sample.
//
// Inputs:
// - ViewportConstants (b0): ViewportSize in pixels, plus unused Padding.
// - VSInput: LocalPos (POSITION), Rect (TEXCOORD0, x/y/width/height),
//   ClipRect (TEXCOORD1, x/y/width/height), FillColor (COLOR0),
//   FillColorEnd (COLOR1), BorderColor (COLOR2), UvRect (TEXCOORD2, u0/v0/u1/v1),
//   Params (TEXCOORD3), Params2 (TEXCOORD4).
// - AtlasTexture (t0), VideoLumaTexture (t1), VideoChromaTexture (t2),
//   sampled with LinearSampler (s0). This file does not set the sampler filter.
//
// Params.x is the corner radius. Params.y is the border width.
// Params.z above 0.5 multiplies by the atlas. Params.z above 1.5 multiplies by
// the NV12 luma/chroma pair instead. Params.w above 0.5 selects the gradient.
// Params.w above 1.5 selects the shadow. Params2.x is the gradient angle in
// radians. Params2.y is the shadow blur radius.
// World y grows downward. Clip-space y is flipped. Colors written by PSMain
// keep RGB straight and scale alpha; this file does not read a destination pixel.

cbuffer ViewportConstants : register(b0)
{
    float2 ViewportSize;
    float2 Padding;
};

struct VSInput
{
    float2 LocalPos     : POSITION;
    float4 Rect         : TEXCOORD0; // x, y, width, height
    float4 ClipRect     : TEXCOORD1; // x, y, width, height
    float4 FillColor    : COLOR0;
    float4 FillColorEnd : COLOR1;
    float4 BorderColor  : COLOR2;
    float4 UvRect       : TEXCOORD2; // u0, v0, u1, v1
    float4 Params       : TEXCOORD3; // x = corner_radius, y = border_width, z = has_texture, w = prim_kind
    float4 Params2      : TEXCOORD4; // x = grad_angle_rad, y = shadow_blur
};

struct PSInput
{
    float4 Position     : SV_POSITION;
    float2 PixelPos     : TEXCOORD0;
    float4 Rect         : TEXCOORD1;
    float4 ClipRect     : TEXCOORD2;
    float4 FillColor    : COLOR0;
    float4 FillColorEnd : COLOR1;
    float4 BorderColor  : COLOR2;
    float2 Uv           : TEXCOORD3;
    float4 Params       : TEXCOORD4;
    float4 Params2      : TEXCOORD5;
};

Texture2D AtlasTexture : register(t0);
Texture2D VideoLumaTexture : register(t1);
Texture2D VideoChromaTexture : register(t2);
SamplerState LinearSampler : register(s0);

// Limited-range BT.709 inverse in normalized 0..1 units.
// ySample is luma. uvSample is Cb, Cr. Luma below 16/255 is treated as 16/255
// and expanded by 255/219. Chroma is centered at 0.5 and scaled by the
// studio-swing inverse (255/224 on the BT.709 axes). RGB is saturated to 0..1.
// Luma is not clamped to 235/255 before the inverse. Alpha is 1.
float4 ConvertNv12Bt709ToRgba(float ySample, float2 uvSample)
{
    float y = max(ySample - (16.0 / 255.0), 0.0) * 1.16438356;
    float u = uvSample.x - 0.5;
    float v = uvSample.y - 0.5;
    float r = saturate(y + 1.79274107 * v);
    float g = saturate(y - 0.21324861 * u - 0.53290933 * v);
    float b = saturate(y + 2.11240179 * u);
    return float4(r, g, b, 1.0);
}

// Limited-range BT.709 forward matrix in normalized 0..1 units.
// rgb is 0..1. Luma is offset by 16/255 and scaled by 219/255
// (Kr 0.2126, Kg 0.7152, Kb 0.0722). Chroma is offset by 0.5 and scaled by 224/255.
// The result is clamped to 0..1, not to the narrower 16..235 and 16..240 codes.
// PSMain does not call this function.
float3 ConvertRgbaToYuvBt709(float3 rgb)
{
    float y = (16.0 / 255.0) + 0.18258588 * rgb.r + 0.61423059 * rgb.g + 0.06200706 * rgb.b;
    float u = 0.5 - 0.10064373 * rgb.r - 0.33857195 * rgb.g + 0.43921569 * rgb.b;
    float v = 0.5 + 0.43921569 * rgb.r - 0.39894216 * rgb.g - 0.04027352 * rgb.b;
    return saturate(float3(y, u, v));
}

// LocalPos is the unit quad, 0..1. World position is the rect minimum plus
// LocalPos times the size. NDC is that position over the viewport, mapped to
// -1..1, then y is flipped so world +y (down) becomes clip -y.
PSInput VSMain(VSInput input)
{
    PSInput output;
    float2 worldPos = input.Rect.xy + input.LocalPos * input.Rect.zw;
    float2 ndc = (worldPos / ViewportSize) * 2.0 - 1.0;
    ndc.y = -ndc.y;

    output.Position = float4(ndc, 0.0, 1.0);
    output.PixelPos = worldPos;
    output.Rect = input.Rect;
    output.ClipRect = input.ClipRect;
    output.FillColor = input.FillColor;
    output.FillColorEnd = input.FillColorEnd;
    output.BorderColor = input.BorderColor;
    output.Uv = lerp(input.UvRect.xy, input.UvRect.zw, input.LocalPos);
    output.Params = input.Params;
    output.Params2 = input.Params2;
    return output;
}

// Signed distance to a rounded box centered at the origin.
// p is relative to the center. Negative inside, zero on the boundary, positive outside.
// The radius is clamped so the corner arcs fit inside halfSize.
float SdfRoundedBox(float2 p, float2 halfSize, float radius)
{
    float r = min(radius, min(halfSize.x, halfSize.y));
    float2 q = abs(p) - (halfSize - float2(r, r));
    return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r;
}

// Hermite smoothstep of (0.5 - dist / sigma), with sigma at least 0.5.
// Negative dist (inside) approaches 1. Coverage reaches 0 at dist >= 0.5 * sigma.
float SdfBoxShadowAlpha(float dist, float blurRadius)
{
    float sigma = max(blurRadius, 0.5);
    float t = saturate(0.5 - dist / sigma);
    return t * t * (3.0 - 2.0 * t);
}

// Two-stop gradient. Direction is (sin(angleRad), -cos(angleRad)).
// Zero radians points toward decreasing y. The box extent along that direction
// maps one side to startColor and the opposite side to endColor.
float4 EvaluateLinearGradient(float2 relPos, float2 halfSize, float angleRad, float4 startColor, float4 endColor)
{
    float2 dir = float2(sin(angleRad), -cos(angleRad));
    float maxExtent = max(abs(dir.x) * halfSize.x + abs(dir.y) * halfSize.y, 0.5);
    float proj = dot(relPos, dir);
    float t = saturate((proj / maxExtent) * 0.5 + 0.5);
    return lerp(startColor, endColor, t);
}

float4 PSMain(PSInput input) : SV_TARGET
{
    // Inclusive on the clip maximum. PixelPos is the interpolated world position, y down.
    float2 clipMin = input.ClipRect.xy;
    float2 clipMax = input.ClipRect.xy + input.ClipRect.zw;
    if (input.PixelPos.x < clipMin.x || input.PixelPos.y < clipMin.y ||
        input.PixelPos.x > clipMax.x || input.PixelPos.y > clipMax.y)
    {
        discard;
    }

    float2 halfSize = input.Rect.zw * 0.5;
    float2 center = input.Rect.xy + halfSize;
    float2 relPos = input.PixelPos - center;

    float radius = input.Params.x;
    float borderWidth = input.Params.y;
    float hasTexture = input.Params.z;
    float primKind = input.Params.w;

    float dist = SdfRoundedBox(relPos, halfSize, radius);

    // primKind above 1.5: shadow. Alpha is fill alpha times the smoothstep. RGB stays straight.
    if (primKind > 1.5)
    {
        float blurRadius = input.Params2.y;
        float shadowAlpha = SdfBoxShadowAlpha(dist, blurRadius);
        if (shadowAlpha <= 0.0)
        {
            discard;
        }
        float4 shadowColor = input.FillColor;
        shadowColor.a *= shadowAlpha;
        return shadowColor;
    }

    // One-pixel coverage ramp around the zero isosurface.
    float outerAlpha = saturate(0.5 - dist);
    if (outerAlpha <= 0.0)
    {
        discard;
    }

    float4 baseColor = input.FillColor;
    // primKind above 0.5 and at most 1.5: gradient. Shadow already returned above.
    if (primKind > 0.5)
    {
        baseColor = EvaluateLinearGradient(relPos, halfSize, input.Params2.x, input.FillColor, input.FillColorEnd);
    }
    // hasTexture above 1.5: NV12 pair. Above 0.5 only: atlas. Multiply is straight, not a blend.
    if (hasTexture > 1.5)
    {
        float yVal = VideoLumaTexture.Sample(LinearSampler, input.Uv).r;
        float2 uvVal = VideoChromaTexture.Sample(LinearSampler, input.Uv).rg;
        baseColor *= ConvertNv12Bt709ToRgba(yVal, uvVal);
    }
    else if (hasTexture > 0.5)
    {
        baseColor *= AtlasTexture.Sample(LinearSampler, input.Uv);
    }

    if (borderWidth > 0.0)
    {
        // innerDist is zero one stroke-width inside the outer edge. The mix is a one-pixel ramp.
        float innerDist = dist + borderWidth;
        float borderMix = saturate(0.5 + innerDist);
        baseColor = lerp(baseColor, input.BorderColor, borderMix);
    }

    baseColor.a *= outerAlpha;
    return baseColor;
}
