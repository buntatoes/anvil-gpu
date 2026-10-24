<div align="center">
  <h1>anvil-gpu</h1>
  <p>CPU drawing of rounded boxes, shadows, gradients, and NV12.</p>
  <p>
    <img src="https://img.shields.io/badge/release-v.05-e8c37a?style=flat-square" alt="v.05" />
    <img src="https://img.shields.io/badge/status-Proof_of_Concept-d4a359?style=flat-square" alt="Proof of Concept" />
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-MPL--2.0-blue?style=flat-square" alt="MPL-2.0" /></a>
    <a href="./SECURITY.md"><img src="https://img.shields.io/badge/security-policy-2ea44f?style=flat-square" alt="Security" /></a>
    <img src="https://img.shields.io/badge/rust-2024-orange?style=flat-square" alt="Rust 2024" />
  </p>
</div>

Version 0.5.0.

The name is historical. This crate does not open a GPU device. The HLSL and GLSL files are source text stored in the repository. Nothing here compiles them or submits them to Direct3D, OpenGL, or Vulkan.

## What it does

- `render_frame_rgba` draws quads into an RGBA buffer on the CPU: solid fills, linear gradients, rounded corners, borders, box shadows, clips, and atlas samples.
- `sdf_rounded_box`, `sdf_box_shadow_alpha`, and `sample_linear_gradient` are the math behind that.
- `TextureAtlas` packs rectangles and returns UV bounds.
- `Nv12Surface` converts RGBA to NV12 and back with BT.709 coefficients, on the CPU.
- `COMPOSITOR_HLSL` and `COMPOSITOR_GLSL` are the shader source strings. They are not executed.

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
