// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 Buntato
// GLSL 330 core source text. Author: Buntato.
//
// Nothing in this repository compiles this file or creates an OpenGL or Vulkan
// device. The crate embeds this text with include_str!. CPU code in src/lib.rs
// draws rounded boxes, box shadows, and linear gradients, shelf-packs an RGBA
// atlas, and converts RGBA and limited-range BT.709 NV12. anvil-raster calls
// those CPU SDF helpers. anvil-video uses Nv12Surface for CPU color conversion.
//
// VERTEX_SHADER and FRAGMENT_SHADER each wrap one main. A host would compile
// this text twice, defining one symbol at a time. No build in this repository
// defines either symbol or links the result.
//
// Entry points (not invoked from this repository):
// - VERTEX_SHADER main: expands a unit quad into the rect, flips y into clip
//   space, and forwards colors, clip, UV, and params.
// - FRAGMENT_SHADER main: rejects fragments outside the clip, evaluates the
//   rounded-box SDF, and writes a shadow, a solid fill, or a linear gradient.
//   It can multiply that color by an atlas sample or by an NV12 BT.709 sample.
//
// Uniforms: uViewportSize (pixels), uAtlasTexture, uVideoLumaTexture,
// uVideoChromaTexture. This file does not set sampler filters.
// Vertex inputs: aLocalPos (location 0), aRect (1, x/y/width/height),
// aClipRect (2), aFillColor (3), aFillColorEnd (4), aBorderColor (5),
// aUvRect (6, u0/v0/u1/v1), aParams (7), aParams2 (8).
// aParams.x is the corner radius. aParams.y is the border width.
// aParams.z above 0.5 multiplies by the atlas. aParams.z above 1.5 multiplies
// by the NV12 pair instead. aParams.w above 0.5 selects the gradient.
// aParams.w above 1.5 selects the shadow. aParams2.x is the gradient angle in
// radians. aParams2.y is the shadow blur radius.
// World y grows downward. Clip-space y is flipped. Fragment output keeps RGB
// straight and scales alpha; this file does not read a destination pixel.

#version 330 core

uniform vec2 uViewportSize;
uniform sampler2D uAtlasTexture;
uniform sampler2D uVideoLumaTexture;
uniform sampler2D uVideoChromaTexture;

#ifdef VERTEX_SHADER
layout(location = 0) in vec2 aLocalPos;
layout(location = 1) in vec4 aRect;         // x, y, width, height
layout(location = 2) in vec4 aClipRect;     // x, y, width, height
layout(location = 3) in vec4 aFillColor;
layout(location = 4) in vec4 aFillColorEnd;
layout(location = 5) in vec4 aBorderColor;
layout(location = 6) in vec4 aUvRect;       // u0, v0, u1, v1
layout(location = 7) in vec4 aParams;       // x = corner_radius, y = border_width, z = has_texture, w = prim_kind
layout(location = 8) in vec4 aParams2;      // x = grad_angle_rad, y = shadow_blur

out vec2 vPixelPos;
out vec4 vRect;
out vec4 vClipRect;
out vec4 vFillColor;
out vec4 vFillColorEnd;
out vec4 vBorderColor;
out vec2 vUv;
out vec4 vParams;
out vec4 vParams2;

void main()
{
    // aLocalPos is the unit quad, 0..1. World position is the rect minimum plus
    // aLocalPos times the size. NDC maps that over the viewport to -1..1, then
    // flips y so world +y (down) becomes clip -y.
    vec2 worldPos = aRect.xy + aLocalPos * aRect.zw;
    vec2 ndc = (worldPos / uViewportSize) * 2.0 - 1.0;
    ndc.y = -ndc.y;

    gl_Position = vec4(ndc, 0.0, 1.0);
    vPixelPos = worldPos;
    vRect = aRect;
    vClipRect = aClipRect;
    vFillColor = aFillColor;
    vFillColorEnd = aFillColorEnd;
    vBorderColor = aBorderColor;
    vUv = mix(aUvRect.xy, aUvRect.zw, aLocalPos);
    vParams = aParams;
    vParams2 = aParams2;
}
#endif

#ifdef FRAGMENT_SHADER
in vec2 vPixelPos;
in vec4 vRect;
in vec4 vClipRect;
in vec4 vFillColor;
in vec4 vFillColorEnd;
in vec4 vBorderColor;
in vec2 vUv;
in vec4 vParams;
in vec4 vParams2;

out vec4 FragColor;

// Signed distance to a rounded box centered at the origin.
// p is relative to the center. Negative inside, zero on the boundary, positive outside.
// The radius is clamped so the corner arcs fit inside halfSize.
float SdfRoundedBox(vec2 p, vec2 halfSize, float radius)
{
    float r = min(radius, min(halfSize.x, halfSize.y));
    vec2 q = abs(p) - (halfSize - vec2(r, r));
    return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r;
}

// Hermite smoothstep of (0.5 - dist / sigma), with sigma at least 0.5.
// Negative dist (inside) approaches 1. Coverage reaches 0 at dist >= 0.5 * sigma.
float SdfBoxShadowAlpha(float dist, float blurRadius)
{
    float sigma = max(blurRadius, 0.5);
    float t = clamp(0.5 - dist / sigma, 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

// Two-stop gradient. Direction is (sin(angleRad), -cos(angleRad)).
// Zero radians points toward decreasing y. The box extent along that direction
// maps one side to startColor and the opposite side to endColor.
vec4 EvaluateLinearGradient(vec2 relPos, vec2 halfSize, float angleRad, vec4 startColor, vec4 endColor)
{
    vec2 dir = vec2(sin(angleRad), -cos(angleRad));
    float maxExtent = max(abs(dir.x) * halfSize.x + abs(dir.y) * halfSize.y, 0.5);
    float proj = dot(relPos, dir);
    float t = clamp((proj / maxExtent) * 0.5 + 0.5, 0.0, 1.0);
    return mix(startColor, endColor, t);
}

// Limited-range BT.709 inverse in normalized 0..1 units.
// ySample is luma. uvSample is Cb, Cr. Luma below 16/255 is treated as 16/255
// and expanded by 255/219. Chroma is centered at 0.5 and scaled by the
// studio-swing inverse (255/224 on the BT.709 axes). RGB is saturated to 0..1.
// Luma is not clamped to 235/255 before the inverse. Alpha is 1.
vec4 ConvertNv12Bt709ToRgba(float ySample, vec2 uvSample)
{
    float y = max(ySample - (16.0 / 255.0), 0.0) * 1.16438356;
    float u = uvSample.x - 0.5;
    float v = uvSample.y - 0.5;
    float r = clamp(y + 1.79274107 * v, 0.0, 1.0);
    float g = clamp(y - 0.21324861 * u - 0.53290933 * v, 0.0, 1.0);
    float b = clamp(y + 2.11240179 * u, 0.0, 1.0);
    return vec4(r, g, b, 1.0);
}

// Limited-range BT.709 forward matrix in normalized 0..1 units.
// rgb is 0..1. Luma is offset by 16/255 and scaled by 219/255
// (Kr 0.2126, Kg 0.7152, Kb 0.0722). Chroma is offset by 0.5 and scaled by 224/255.
// The result is clamped to 0..1, not to the narrower 16..235 and 16..240 codes.
// The fragment main does not call this function.
vec3 ConvertRgbaToYuvBt709(vec3 rgb)
{
    float y = (16.0 / 255.0) + 0.18258588 * rgb.r + 0.61423059 * rgb.g + 0.06200706 * rgb.b;
    float u = 0.5 - 0.10064373 * rgb.r - 0.33857195 * rgb.g + 0.43921569 * rgb.b;
    float v = 0.5 + 0.43921569 * rgb.r - 0.39894216 * rgb.g - 0.04027352 * rgb.b;
    return clamp(vec3(y, u, v), vec3(0.0), vec3(1.0));
}

void main()
{
    // Inclusive on the clip maximum. vPixelPos is the interpolated world position, y down.
    vec2 clipMin = vClipRect.xy;
    vec2 clipMax = vClipRect.xy + vClipRect.zw;
    if (vPixelPos.x < clipMin.x || vPixelPos.y < clipMin.y ||
        vPixelPos.x > clipMax.x || vPixelPos.y > clipMax.y)
    {
        discard;
    }

    vec2 halfSize = vRect.zw * 0.5;
    vec2 center = vRect.xy + halfSize;
    vec2 relPos = vPixelPos - center;

    float radius = vParams.x;
    float borderWidth = vParams.y;
    float hasTexture = vParams.z;
    float primKind = vParams.w;

    float dist = SdfRoundedBox(relPos, halfSize, radius);

    // primKind above 1.5: shadow. Alpha is fill alpha times the smoothstep. RGB stays straight.
    if (primKind > 1.5)
    {
        float blurRadius = vParams2.y;
        float shadowAlpha = SdfBoxShadowAlpha(dist, blurRadius);
        if (shadowAlpha <= 0.0)
        {
            discard;
        }
        vec4 shadowColor = vFillColor;
        shadowColor.a *= shadowAlpha;
        FragColor = shadowColor;
        return;
    }

    // One-pixel coverage ramp around the zero isosurface.
    float outerAlpha = clamp(0.5 - dist, 0.0, 1.0);
    if (outerAlpha <= 0.0)
    {
        discard;
    }

    vec4 baseColor = vFillColor;
    // primKind above 0.5 and at most 1.5: gradient. Shadow already returned above.
    if (primKind > 0.5)
    {
        baseColor = EvaluateLinearGradient(relPos, halfSize, vParams2.x, vFillColor, vFillColorEnd);
    }
    // hasTexture above 1.5: NV12 pair. Above 0.5 only: atlas. Multiply is straight, not a blend.
    if (hasTexture > 1.5)
    {
        float yVal = texture(uVideoLumaTexture, vUv).r;
        vec2 uvVal = texture(uVideoChromaTexture, vUv).rg;
        baseColor *= ConvertNv12Bt709ToRgba(yVal, uvVal);
    }
    else if (hasTexture > 0.5)
    {
        baseColor *= texture(uAtlasTexture, vUv);
    }

    if (borderWidth > 0.0)
    {
        // innerDist is zero one stroke-width inside the outer edge. The mix is a one-pixel ramp.
        float innerDist = dist + borderWidth;
        float borderMix = clamp(0.5 + innerDist, 0.0, 1.0);
        baseColor = mix(baseColor, vBorderColor, borderMix);
    }

    baseColor.a *= outerAlpha;
    FragColor = baseColor;
}
#endif
