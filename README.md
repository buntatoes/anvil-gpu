<div align="center">
  <h1>anvil-gpu</h1>
  <p>CPU compositor passes: blurred shadows, gradients, video frame blits, and tile quads.</p>
  <p>
    <img src="https://img.shields.io/badge/release-v.07-e8c37a?style=flat-square" alt="v.07" />
    <img src="https://img.shields.io/badge/status-Proof_of_Concept-d4a359?style=flat-square" alt="Proof of Concept" />
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-MPL--2.0-blue?style=flat-square" alt="MPL-2.0" /></a>
    <a href="./SECURITY.md"><img src="https://img.shields.io/badge/security-policy-2ea44f?style=flat-square" alt="Security" /></a>
    <img src="https://img.shields.io/badge/rust-2024-orange?style=flat-square" alt="Rust 2024" />
  </p>
</div>

Version 0.8.0. The last release tag is v.07.

CPU drawing for the compositor: blurred box shadows, linear and radial gradients, and video frame blits, plus the older tile quad drawing. The same compositor passes are written out in GLSL and HLSL, but nothing in this crate runs them on a GPU yet. Every pixel you get today is drawn on the CPU. Proof of concept.

## What runs where

| Pass | CPU (used today) | GPU |
| --- | --- | --- |
| Box shadow with blur | `draw_box_shadow`, `Compositor::box_shadow` | Shader passes written, not run |
| Linear and radial gradients, up to 16 stops | `draw_gradient`, `Compositor::fill_gradient` | Shader pass written, not run |
| RGBA, NV12, and I420 frame blit with scaling | `draw_blit`, `Compositor::blit_frame` | Shader passes written, not run |
| Rounded boxes, borders, two-stop gradients, atlas quads | `render_frame_rgba`, `GpuDevice::render_quads` | `compositor.glsl` and `compositor.hlsl`, not run |

What is real on the hardware side:

- On Linux, `probe_hardware` loads `libEGL.so.1` and calls `eglInitialize` on the default display, once per process. It reports the EGL version. It does not create a context, so it cannot tell a GPU from a software driver.
- On Windows, it creates a Direct3D 11 device (hardware, then WARP) and compiles every pass in `shaders/passes.hlsl` to DXBC with `d3dcompiler_47.dll`.
- `select_backend` only picks the GPU when a device opened, the passes compiled, and a draw path exists. No draw path exists yet (no upload, no draw, no read back), so `Compositor::open` always draws on the CPU, and `HardwareProbe::reason` says why.

The probe is cached per process, not per machine. On Linux, the first `GpuDevice::open`, `Compositor::open`, or `probe_hardware` in a process loads EGL and initializes the default display, which connects that process to the display server, and the connection stays open until the process exits. A program that runs one renderer process per tab should probe once in the main process and pass the decision to each renderer, for example by calling `Compositor::software()` there when the main process picked software, rather than letting every renderer open its own connection.

How the shader passes were checked: every GLSL pass and every HLSL entry point passes `glslangValidator`. The tests in this repository run the CPU passes only; nothing here runs a shader. The GLSL passes were also compared against the CPU output once on Mesa's llvmpipe driver, by a separate program that is not part of this repository and is not a test. The HLSL passes have not been run on a device, and on Windows they are only compiled.

## What it does

- `Compositor` takes a shadow, gradient, or blit, checks it, and draws it. It can hold a device backend (`ComposeBackend`). If that backend returns an error, the compositor puts back the pixels the call could have touched, drops the device, and draws the call on the CPU, so the result is the same either way.
- `draw_box_shadow` draws the rounded box into a coverage mask, blurs it with three box passes along rows and three along columns (close to a gaussian with sigma `blur / 2`), then paints the color through it. Offset, spread, corner radius, and clip are close to CSS box shadows, but the blur is a three-box approximation of the CSS gaussian, and the corner radius simply grows by the spread. A sharp box keeps sharp corners when it grows. The mask is made and blurred a row at a time, so it is never held whole: at the largest blur on a 16384 pixel wide target the working memory stays under 60 MB.
- `draw_gradient` fills a rectangle with a linear gradient (start and end point) or an elliptical radial gradient, with 1 to 16 stops.
- `draw_blit` copies an RGBA, NV12, or I420 frame into a rectangle, with nearest or bilinear sampling. YUV uses limited-range BT.709, the same as `Nv12Surface`. A 1:1 NV12 blit gives the same bytes as `Nv12Surface::to_rgba_bt709`. Bilinear sampling treats chroma as sitting at the center of each 2 by 2 luma block. Most BT.709 video sites chroma on the left edge of the block, so for that video the scaled chroma lands half a luma pixel to the right, which only shows as slight fringes on sharp color edges. The shader passes use the same siting, so the output does not depend on the backend.
- Targets are straight RGBA8 with a row stride (`RgbaTarget`). Sizes, strides, plane lengths, and values are checked first. Bad input returns `ComposeError` and draws nothing. Each side is capped at 16384 pixels and blur at 256.
- `GpuDevice::render_quads` counts the 16 by 16 tiles a quad touches, then shades the frame once. The bytes match `render_frame_rgba`.
- `composite_rgba_frame`, `convert_rgba_on_device`, and `present_nv12_on_device` work as before and run on the CPU.
- `sdf_rounded_box`, `sdf_box_shadow_alpha`, `sample_linear_gradient`, `sample_multi_stop_gradient`, and `sample_radial_gradient` are the shared math.
- `TextureAtlas` packs rectangles and returns UV bounds. `Nv12Surface` converts RGBA to NV12 and back.
- `COMPOSITOR_GLSL`, `COMPOSITOR_HLSL`, `PASSES_GLSL`, and `PASSES_HLSL` are the shader sources.

## Unsafe code

All of it is in `src/sys.rs`. The rest of the crate is built with `#![deny(unsafe_code)]`. Each block has a comment saying why it is sound.

- Linux: `dlopen` of `libEGL.so.1`, `dlsym` of `eglGetDisplay` and `eglInitialize`, casting those two symbols to function pointers, and calling them. The library stays loaded and the display stays initialized for the process.
- Windows: `LoadLibraryA` and `GetProcAddress` for `D3DCompile` and `D3D11CreateDevice`, the calls themselves, reading and releasing the compiler blobs, and releasing the device and context.

No pointer from these calls leaves `sys.rs`.

## Build

```
cargo test
```

Rust 2024. No crates.io dependencies. The robustness tests in `tests/robustness.rs` run for 1.5 seconds each. Set `ANVIL_ROBUSTNESS_MS` for a longer run.

## The other repositories

The browser is [anvil-browser](https://github.com/buntatoes/anvil-browser) (Apache-2.0). The page engine is `anvil-engine` (proprietary, not MPL-2.0). These libraries sit next to both. None of them come from crates.io.

| Repository | What it is |
| --- | --- |
| [anvil-cipher](https://github.com/buntatoes/anvil-cipher) | AES-128, AES-256, CTR, AES-256-GCM |
| [anvil-compress](https://github.com/buntatoes/anvil-compress) | DEFLATE, zlib, gzip |
| [anvil-image](https://github.com/buntatoes/anvil-image) | PNG, JPEG, GIF, WebP, ICO, BMP, PPM |
| [anvil-json](https://github.com/buntatoes/anvil-json) | JSON |
| [anvil-ttf](https://github.com/buntatoes/anvil-ttf) | TrueType and OpenType (`glyf`, `CFF `, `kern`, `GPOS`) |
| [anvil-gpu](https://github.com/buntatoes/anvil-gpu) | Software drawing of boxes, shadows, gradients, and NV12 |
| [anvil-video](https://github.com/buntatoes/anvil-video) | CPU frames and PCM/ADPCM audio in a small MP4 container |

## License

MPL-2.0. Copyright (c) 2026 Buntato.
