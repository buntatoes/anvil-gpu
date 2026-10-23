<div align="center">
  <h1>flux-gpu</h1>
  <p><strong>2D SDF Quad Compositor, NV12 BT.709 GPU Video Surface Pipeline, Texture Atlas &amp; HLSL/GLSL Shaders (Proof of Concept)</strong></p>
  <p>
    <a href="https://github.com/buntatoes/flux-gpu/releases/tag/v.04"><img src="https://img.shields.io/badge/release-v.04-e8c37a?style=flat-square" alt="Release v.04" /></a>
    <img src="https://img.shields.io/badge/status-Proof_of_Concept_(PoC)-d4a359?style=flat-square" alt="Proof of Concept" />
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-MPL--2.0-blue?style=flat-square" alt="License MPL-2.0" /></a>
    <a href="https://github.com/buntatoes/flux-gpu"><img src="https://img.shields.io/badge/rust-2024_edition-orange?style=flat-square" alt="Rust 2024 Edition" /></a>
  </p>
</div>

> [!NOTE]
> **Proof of Concept (PoC) Notice**: `flux-gpu` is a standalone, zero-dependency **Proof of Concept** 2D quad compositor and `NV12` BT.709 GPU video surface library authored from scratch by **Buntos** as part of the [**Flux Browser**](https://github.com/buntatoes/flux-browser) engine ecosystem. It is implemented in **Rust**, **HLSL Shader Model 5.0** ([`shaders/compositor.hlsl`](./shaders/compositor.hlsl) for Windows Direct3D), and **GLSL 330 Core** ([`shaders/compositor.glsl`](./shaders/compositor.glsl) for Linux / cross-platform OpenGL & Vulkan).

---

## Features (`v.04`)

- **Signed-Distance-Field (SDF) Quad Primitives (`GpuQuad`)**:
  - Solid fills (`GpuQuad::solid`), linear gradients at arbitrary angles (`GpuQuad::linear_gradient`), soft SDF box shadows (`GpuQuad::box_shadow`), textured quads, anti-aliased rounded corners (`corner_radius`), borders (`border_width`, `border_color`), and per-quad scissor clip rectangles.
- **Semi-Planar 4:2:0 `NV12` GPU Video Surfaces & BT.709 Shaders (`Nv12Surface`, `ConvertNv12Bt709ToRgba`, `ConvertRgbaToYuvBt709`)**:
  - Hardware video decoders and encoders (Direct3D 11 Video / DXVA2 on Windows and VA-API on Linux) operate on `NV12` (`Y` luma + interleaved `UV` chroma) surfaces. `flux-gpu` provides `Nv12Surface::from_rgba_bt709` / `to_rgba_bt709` alongside GPU `ConvertNv12Bt709ToRgba` and `ConvertRgbaToYuvBt709` shaders in both HLSL and GLSL.
- **Shelf-Packing RGBA Texture Atlas (`TextureAtlas`)**:
  - Packs decoded RGBA images and glyph bitmaps into a single GPU-ready texture sheet and computes normalized `[u0, v0, u1, v1]` UV coordinates.
- **Cross-Platform Embedded Shaders (`COMPOSITOR_HLSL` & `COMPOSITOR_GLSL`)**:
  - Direct3D 11/12 HLSL (`shaders/compositor.hlsl`) and OpenGL 3.3 Core / Vulkan GLSL (`shaders/compositor.glsl`) vertex and fragment shaders evaluating `SdfRoundedBox` coverage, `SdfBoxShadowAlpha` smoothstep shadow falloff, `EvaluateLinearGradient` color ramps, `ConvertNv12Bt709ToRgba` hardware video sampling, border blending, scissor clipping, and atlas texture modulation.
- **Matching Software SDF Compositor (`render_frame_rgba`)**:
  - Evaluates the exact same SDF rounded-rectangle, linear gradient, box shadow, border, clipping, and texture sampling pipeline into a CPU RGBA8888 framebuffer for headless testing and fallback rendering.

---

## Quick Start

```rust
use flux_gpu::{GpuColor, GpuQuad, GpuRect, render_frame_rgba};

let shadow = GpuQuad::box_shadow(
    GpuRect::new(10.0, 10.0, 120.0, 60.0),
    0.0,
    4.0,
    8.0,
    0.0,
    8.0,
    GpuColor::rgba(0, 0, 0, 160),
);
let card = GpuQuad::linear_gradient(
    GpuRect::new(10.0, 10.0, 120.0, 60.0),
    GpuColor::rgba(48, 43, 38, 255),
    GpuColor::rgba(28, 25, 22, 255),
    135.0,
)
.with_rounded_border(8.0, 2.0, GpuColor::rgba(232, 195, 122, 255));

let framebuffer = render_frame_rgba(
    140,
    80,
    GpuColor::rgba(28, 26, 23, 255),
    &[shadow, card],
    None,
);
assert_eq!(framebuffer.len(), 140 * 80 * 4);
```

---

## API Overview

| Type / Function | Description |
| :--- | :--- |
| `COMPOSITOR_HLSL: &str` | Embedded HLSL Shader Model 5.0 vertex and pixel shader source (Windows Direct3D). |
| `COMPOSITOR_GLSL: &str` | Embedded GLSL 330 Core vertex and fragment shader source (Linux OpenGL / Vulkan). |
| `GpuQuad` | 2D quad primitive (`solid`, `linear_gradient`, `box_shadow`, `with_rounded_border`, `with_clip`, `with_uv`). |
| `TextureAtlas` | Shelf-packing RGBA8888 texture atlas (`allocate`, `uv_bounds`, `sample_uv`). |
| `Nv12Surface` | Semi-planar 4:2:0 `NV12` GPU video surface (`from_rgba_bt709`, `to_rgba_bt709`). |
| `bt709_rgba_to_yuv` / `bt709_yuv_to_rgba` | BT.709 color-space conversion between `RGBA8888` and `YUV` (`NV12`). |
| `sdf_rounded_box(rel_x, rel_y, half_w, half_h, radius) -> f32` | Signed distance function for a 2D box with rounded corners. |
| `sdf_box_shadow_alpha(dist, blur_radius) -> f32` | Smoothstep SDF falloff alpha (`0.0..=1.0`) for a soft box shadow. |
| `sample_linear_gradient(...) -> GpuColor` | Evaluates a two-stop linear gradient at an arbitrary angle in degrees. |
| `render_frame_rgba(width, height, clear, quads, atlas) -> Vec<u8>` | Renders a slice of `GpuQuad` primitives into an RGBA8888 framebuffer. |

---

## Building and Testing

```powershell
cargo test
```

---

## Security Policy

See [`SECURITY.md`](./SECURITY.md) for supported versions, surface/atlas bounds, and vulnerability reporting instructions.

## License

Licensed under the Mozilla Public License Version 2.0 ([`MPL-2.0`](./LICENSE)).

## Author

**Buntos** ([@buntatoes](https://github.com/buntatoes))
