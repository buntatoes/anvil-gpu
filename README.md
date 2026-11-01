<div align="center">
  <h1>Anvil GPU</h1>
  <p>Compositor passes for boxes, shadows, gradients, glyph coverage, and NV12/I420 video blits in Go.</p>
  <p>
    <img src="https://img.shields.io/badge/release-v.12-e8c37a?style=flat-square" alt="v.12" />
    <img src="https://img.shields.io/badge/status-Proof_of_Concept-d4a359?style=flat-square" alt="Proof of Concept" />
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Noncommercial_%28based_on_PolyForm_1.0.0%29_%7C_Paid_Commercial_%28Non--Compete%29-blue?style=flat-square" alt="Noncommercial (based on PolyForm 1.0.0) | Paid Commercial (Non-Compete)" /></a>
    <a href="./SECURITY.md"><img src="https://img.shields.io/badge/security-policy-2ea44f?style=flat-square" alt="Security" /></a>
    <img src="https://img.shields.io/badge/go-1.22%2B-00ADD8?style=flat-square" alt="Go 1.22+" />
  </p>
</div>

Version 0.12.0 (`v.12`).

This is the 2D compositor, scene batch, texture atlas, and BT.709 video surface library for Anvil (`github.com/buntatoes/anvil-gpu`), written from scratch in pure Go with embedded HLSL Shader Model 5.0 (`shaders/*.hlsl`) and GLSL 330 core (`shaders/*.glsl`) pass sources.

## How the engine and browser use it

- `anvil-engine` (`Forge`) calls `Compositor` and `GpuSceneBatch` for box shadows, linear/radial/rounded gradients, solid and bordered SDF rounded boxes, glyph coverage blits, and `GpuDevice` 16x16 tiled quad rasterization.
- `anvil-video` calls `Compositor` (`DrawBlit`) for scaled RGBA, NV12, and I420 frame blits and `Nv12Surface` / `I420Surface` for limited-range BT.709 color conversion.

## What it does

- **`Compositor` & `GpuSceneBatch`** (`compose.go`):
  - **Box shadows** (`DrawBoxShadow`): SDF rounded rectangle mask streamed through three fixed-point (`MASK_ONE = 1 << 15`) horizontal and vertical box blur passes.
  - **Gradients** (`DrawGradient`, `DrawRoundedGradient`): 1 to 16 stops (`MaxGradientStops`), linear and elliptical radial shapes, optional SDF rounded-corner clipping.
  - **Solid & Rounded Boxes** (`DrawSolidRect`): Axis-aligned fast fill or SDF rounded rectangle with inner border stroke.
  - **Video & Image Blits** (`DrawBlit`): Scaled `VideoFrame` (`RGBA`, `NV12`, `I420`) blits with `FilterNearest` and fixed-point (`BILINEAR_ONE = 4096`) `FilterBilinear` resampling.
  - **Batched Scene Execution** (`GpuSceneBatch`, `DrawSceneBatch`): Records `Clear`, `PushClip`, `PopClip`, `SolidRect`, `BoxShadow`, `Gradient`, `RoundedGradient`, `BlitRGBA`, `BlitNV12`, `BlitI420`, and `GlyphCoverage` operations and executes them transactionally.
- **`GpuDevice` & `TextureAtlas`** (`device.go`, `atlas.go`, `gpu.go`): 16x16 tiled `GpuQuad` rasterizer (`RenderQuads`, `RenderFrameRGBA`) and shelf-packed straight RGBA8 `TextureAtlas`.
- **BT.709 `Nv12Surface` & `I420Surface`** (`yuv.go`): Studio-swing BT.709 (`Y` in `16..235`, `Cb/Cr` in `16..240`) conversions between straight RGBA8, semi-planar 4:2:0 `NV12`, and planar 4:2:0 `I420`.
- **Embedded Shaders** (`shaders.go`): Embeds `shaders/compositor.hlsl`, `shaders/compositor.glsl`, `shaders/passes.hlsl`, and `shaders/passes.glsl`.

## Build

```sh
go vet ./...
go test -v ./...
go build ./cmd/anvil-gpu
```

Go 1.22 or newer (`CGO_ENABLED=0`). Zero external Go module dependencies.

## The other repositories

The browser is [anvil-browser](https://github.com/buntatoes/anvil-browser) and the page engine is [anvil-engine](https://github.com/buntatoes/anvil-engine), built alongside the nine companion libraries below.

| Repository | What it is |
| --- | --- |
| [anvil-cipher](https://github.com/buntatoes/anvil-cipher) | AES-128, AES-256, CTR, AES-256-GCM, SHA-256, HMAC-SHA256, HKDF-SHA256 |
| [anvil-compress](https://github.com/buntatoes/anvil-compress) | DEFLATE, zlib, gzip (Go) |
| [anvil-hpack](https://github.com/buntatoes/anvil-hpack) | HPACK header compression for HTTP/2 (Go) |
| [anvil-image](https://github.com/buntatoes/anvil-image) | PNG, JPEG, GIF, WebP (`VP8L` + `VP8 `), ICO, BMP, PPM |
| [anvil-json](https://github.com/buntatoes/anvil-json) | JSON & RFC 6901 JSON Pointer (Go) |
| [anvil-pqc](https://github.com/buntatoes/anvil-pqc) | ML-KEM-768, X25519, X25519MLKEM768, SHA-3 (Go) |
| [anvil-ttf](https://github.com/buntatoes/anvil-ttf) | TrueType and OpenType (`glyf`, `CFF `, `kern`, `GPOS`) |
| [anvil-gpu](https://github.com/buntatoes/anvil-gpu) | Compositor passes for boxes, shadows, gradients, and NV12/I420 (Go) |
| [anvil-video](https://github.com/buntatoes/anvil-video) | Opus, VP8, VP9, AV1, AVID/ADCT, and MP4/fMP4/WebM containers (Go) |

## License

Dual-licensed under **noncommercial terms based on the PolyForm Noncommercial License 1.0.0** and a **Paid Commercial License with a Non-Compete Clause**. Copyright (c) 2026 Buntato. See [LICENSE](./LICENSE).
