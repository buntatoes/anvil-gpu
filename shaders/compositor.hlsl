// Proprietary 2D SDF Quad Compositor Shader (HLSL Shader Model 5.0)
// Author: Buntos

cbuffer ViewportConstants : register(b0)
{
    float2 ViewportSize;
    float2 Padding;
};

struct VSInput
{
    float2 LocalPos    : POSITION;
    float4 Rect        : TEXCOORD0; // x, y, width, height
    float4 ClipRect    : TEXCOORD1; // x, y, width, height
    float4 FillColor   : COLOR0;
    float4 BorderColor : COLOR1;
    float4 UvRect      : TEXCOORD2; // u0, v0, u1, v1
    float4 Params      : TEXCOORD3; // x = corner_radius, y = border_width, z = has_texture
};

struct PSInput
{
    float4 Position    : SV_POSITION;
    float2 PixelPos    : TEXCOORD0;
    float4 Rect        : TEXCOORD1;
    float4 ClipRect    : TEXCOORD2;
    float4 FillColor   : COLOR0;
    float4 BorderColor : COLOR1;
    float2 Uv          : TEXCOORD3;
    float4 Params      : TEXCOORD4;
};

Texture2D AtlasTexture : register(t0);
SamplerState LinearSampler : register(s0);

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
    output.BorderColor = input.BorderColor;
    output.Uv = lerp(input.UvRect.xy, input.UvRect.zw, input.LocalPos);
    output.Params = input.Params;
    return output;
}

float SdfRoundedBox(float2 p, float2 halfSize, float radius)
{
    float r = min(radius, min(halfSize.x, halfSize.y));
    float2 q = abs(p) - (halfSize - float2(r, r));
    return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r;
}

float4 PSMain(PSInput input) : SV_TARGET
{
    // Scissor clip check
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

    float dist = SdfRoundedBox(relPos, halfSize, radius);
    float outerAlpha = saturate(0.5 - dist);
    if (outerAlpha <= 0.0)
    {
        discard;
    }

    float4 baseColor = input.FillColor;
    if (hasTexture > 0.5)
    {
        baseColor *= AtlasTexture.Sample(LinearSampler, input.Uv);
    }

    if (borderWidth > 0.0)
    {
        float innerDist = dist + borderWidth;
        float borderMix = saturate(0.5 + innerDist);
        baseColor = lerp(baseColor, input.BorderColor, borderMix);
    }

    baseColor.a *= outerAlpha;
    return baseColor;
}
