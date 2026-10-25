<div align="center">
  <h1>anvil-gpu</h1>
  <p>Tile drawing of rounded boxes, shadows, gradients, and NV12.</p>
  <p>
    <img src="https://img.shields.io/badge/release-v.07-e8c37a?style=flat-square" alt="v.07" />
    <img src="https://img.shields.io/badge/status-Proof_of_Concept-d4a359?style=flat-square" alt="Proof of Concept" />
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-MPL--2.0-blue?style=flat-square" alt="MPL-2.0" /></a>
    <a href="./SECURITY.md"><img src="https://img.shields.io/badge/security-policy-2ea44f?style=flat-square" alt="Security" /></a>
    <img src="https://img.shields.io/badge/rust-2024-orange?style=flat-square" alt="Rust 2024" />
  </p>
</div>

Version 0.7.0.

Draw lists are counted in 16 by 16 tiles, then shaded once. The bytes match `render_frame_rgba`. On Linux the crate loads `libEGL.so.1` when that library is present and a display comes back. It does not create a GL context when that probe fails. The HLSL and GLSL files are still source text. This build does not compile them onto a device. Proof of concept.

## What it does

- `GpuDevice::render_quads` counts the 16 by 16 tiles a quad touches, then shades the frame once. Solid fills, linear gradients, rounded corners, borders, box shadows, clips, and atlas samples match `render_frame_rgba`.
- `composite_rgba_frame` draws an RGBA image through one full-frame quad. The bytes match the source.
- `convert_rgba_on_device` and `present_nv12_on_device` convert BT.709 NV12. The plane bytes match `Nv12Surface`.
- `sdf_rounded_box`, `sdf_box_shadow_alpha`, and `sample_linear_gradient` are the math behind that.
- `TextureAtlas` packs rectangles and returns UV bounds.
- `Nv12Surface` converts RGBA to NV12 and back with BT.709 coefficients, on the CPU.
- `COMPOSITOR_HLSL` and `COMPOSITOR_GLSL` are the shader source strings. On Windows, `compile_hlsl_entry` and `probe_and_compile_d3d11_compositor` compile `COMPOSITOR_HLSL` (`VSMain` and `PSMain`) into `DXBC` bytecode via `d3dcompiler_47.dll` and probe `d3d11.dll`.
- `sample_multi_stop_gradient` and `sample_radial_gradient` evaluate arbitrary multi-stop linear and radial color ramps.

## Build

```
cargo test
```

Rust 2024. No crates.io dependencies.

## The other repositories

The browser is [anvil-browser](https://github.com/buntatoes/anvil-browser). These libraries sit next to it. None of them come from crates.io.

| Repository | What it is |
| --- | --- |
| [anvil-cipher](https://github.com/buntatoes/anvil-cipher) | AES-128, AES-256, CTR, AES-256-GCM |
| [anvil-compress](https://github.com/buntatoes/anvil-compress) | DEFLATE, zlib, gzip |
| [anvil-image](https://github.com/buntatoes/anvil-image) | PNG, JPEG, GIF, WebP, ICO, BMP, PPM |
| [anvil-json](https://github.com/buntatoes/anvil-json) | JSON |
| [anvil-ttf](https://github.com/buntatoes/anvil-ttf) | TrueType and OpenType |
| [anvil-gpu](https://github.com/buntatoes/anvil-gpu) | CPU drawing of boxes, shadows, and NV12 |
| [anvil-video](https://github.com/buntatoes/anvil-video) | CPU frames in a small MP4 container |

## License

MPL-2.0. Copyright (c) 2026 Buntos.
