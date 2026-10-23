// Proprietary 2D SDF Quad Compositor Shader (GLSL 330 Core — Linux / Cross-Platform OpenGL & Vulkan)
// Author: Buntos

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

float SdfRoundedBox(vec2 p, vec2 halfSize, float radius)
{
    float r = min(radius, min(halfSize.x, halfSize.y));
    vec2 q = abs(p) - (halfSize - vec2(r, r));
    return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r;
}

float SdfBoxShadowAlpha(float dist, float blurRadius)
{
    float sigma = max(blurRadius, 0.5);
    float t = clamp(0.5 - dist / sigma, 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

vec4 EvaluateLinearGradient(vec2 relPos, vec2 halfSize, float angleRad, vec4 startColor, vec4 endColor)
{
    vec2 dir = vec2(sin(angleRad), -cos(angleRad));
    float maxExtent = max(abs(dir.x) * halfSize.x + abs(dir.y) * halfSize.y, 0.5);
    float proj = dot(relPos, dir);
    float t = clamp((proj / maxExtent) * 0.5 + 0.5, 0.0, 1.0);
    return mix(startColor, endColor, t);
}

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

vec3 ConvertRgbaToYuvBt709(vec3 rgb)
{
    float y = (16.0 / 255.0) + 0.18258588 * rgb.r + 0.61423059 * rgb.g + 0.06200706 * rgb.b;
    float u = 0.5 - 0.10064373 * rgb.r - 0.33857195 * rgb.g + 0.43921569 * rgb.b;
    float v = 0.5 + 0.43921569 * rgb.r - 0.39894216 * rgb.g - 0.04027352 * rgb.b;
    return clamp(vec3(y, u, v), vec3(0.0), vec3(1.0));
}

void main()
{
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

    float outerAlpha = clamp(0.5 - dist, 0.0, 1.0);
    if (outerAlpha <= 0.0)
    {
        discard;
    }

    vec4 baseColor = vFillColor;
    if (primKind > 0.5)
    {
        baseColor = EvaluateLinearGradient(relPos, halfSize, vParams2.x, vFillColor, vFillColorEnd);
    }
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
        float innerDist = dist + borderWidth;
        float borderMix = clamp(0.5 + innerDist, 0.0, 1.0);
        baseColor = mix(baseColor, vBorderColor, borderMix);
    }

    baseColor.a *= outerAlpha;
    FragColor = baseColor;
}
#endif
