# flux-gpu

CPU drawing helpers: rounded rectangles, shadows, gradients, a texture atlas, and NV12 color conversion. Version 0.4.0.

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

The browser is [flux-browser](https://github.com/buntatoes/flux-browser). These libraries sit next to it. None of them come from crates.io.

| Repository | What it is |
| --- | --- |
| [flux-cipher](https://github.com/buntatoes/flux-cipher) | AES-128, AES-256, CTR, AES-256-GCM |
| [flux-compress](https://github.com/buntatoes/flux-compress) | DEFLATE, zlib, gzip |
| [flux-image](https://github.com/buntatoes/flux-image) | PNG, JPEG, GIF, WebP, ICO, BMP, PPM |
| [flux-json](https://github.com/buntatoes/flux-json) | JSON |
| [flux-ttf](https://github.com/buntatoes/flux-ttf) | TrueType and OpenType |
| [flux-gpu](https://github.com/buntatoes/flux-gpu) | CPU drawing of boxes, shadows, and NV12 |
| [flux-video](https://github.com/buntatoes/flux-video) | CPU frames in a small MP4 container |

## License

MPL-2.0. Copyright (c) 2026 Buntos.
