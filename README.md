<div align="center">
  <h1>anvil-gpu</h1>
  <p>Compositor passes: blurred shadows, gradients, video frame blits, and tile quads, on the CPU or a GPU.</p>
  <p>
    <img src="https://img.shields.io/badge/release-v.07-e8c37a?style=flat-square" alt="v.07" />
    <img src="https://img.shields.io/badge/status-Proof_of_Concept-d4a359?style=flat-square" alt="Proof of Concept" />
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-MPL--2.0-blue?style=flat-square" alt="MPL-2.0" /></a>
    <a href="./SECURITY.md"><img src="https://img.shields.io/badge/security-policy-2ea44f?style=flat-square" alt="Security" /></a>
    <img src="https://img.shields.io/badge/rust-2024-orange?style=flat-square" alt="Rust 2024" />
  </p>
</div>

Version 0.9.0. The last release tag is v.07.

Drawing for the compositor: blurred box shadows, linear and radial gradients, and video frame blits, plus the older tile quad drawing. The CPU passes are the reference. The same passes are written in GLSL and HLSL, and since 0.9.0 they run on a GPU when one is there and gives exactly the same bytes as the CPU. Proof of concept.

## What runs where

| Pass | CPU | Linux, OpenGL 3.3 core | Windows, Direct3D 11 |
| --- | --- | --- | --- |
| Box shadow with blur | `draw_box_shadow` | runs (mask, six blur passes, paint) | written, not yet run on Windows |
| Linear and radial gradients, up to 16 stops | `draw_gradient` | runs | written, not yet run on Windows |
| RGBA, NV12, and I420 frame blit with scaling | `draw_blit` | runs | written, not yet run on Windows |
| Rounded boxes, borders, two-stop gradients, atlas quads | `render_frame_rgba`, `GpuDevice::render_quads` | CPU only | CPU only |

`Compositor::box_shadow`, `fill_gradient`, and `blit_frame` use the device when there is one, and the CPU otherwise. `Compositor::software()` always uses the CPU.

### How a device is picked

`probe_hardware` runs once per process, and `Compositor::open` follows it:

1. Linux: load `libEGL.so.1` with `dlopen` and open a display. The surfaceless platform (`EGL_MESA_platform_surfaceless`) is tried first, so no display server is needed; otherwise the default display. Windows: load `d3d11.dll` and make a device at feature level 11_0, hardware first, then WARP.
2. Make an offscreen OpenGL 3.3 core context (surfaceless, or a 1x1 pbuffer), or use the Direct3D device, and build every pass from `shaders/passes.glsl` or `shaders/passes.hlsl` (compiled with `d3dcompiler_47.dll` and IEEE strictness).
3. Draw a self-test scene that covers every pass (blurred, sharp, and clipped shadows, both gradient shapes, all three frame formats with both filters, on a padded target) on the device and on the CPU, and compare every byte.

The device is used only if all three steps succeed. Otherwise everything is drawn on the CPU and `HardwareProbe::reason` says which step failed. Opening runs on a helper thread and counts as failed after 20 seconds, so a driver that hangs cannot hang the caller.

### Fallback

If a device call fails (a GL error, a failed `Map`, a removed device), the compositor puts back the pixels the call could have touched, drops the device, and draws that call and every later one on the CPU. The result is the same bytes either way. `Compositor::fallbacks` counts how often that happened.

### Limits

- A call may make textures of at most 2^24 texels (64 MB for RGBA8 or R32F) each, no larger than the device's maximum side. Bigger calls are drawn on the CPU by the device backend, with the same result.
- Gradient coordinates or radii past 1e15 are drawn on the CPU, because f32 math on the device could overflow where the CPU does not.
- One shadow uses up to four textures at once (two R32F mask textures for the blur, the destination copy, and the output). The device does not stream rows the way the CPU passes do, so its memory grows with the shadow size up to that limit. Textures are freed at the end of each call. The CPU side also keeps a copy of the region for the upload and the read back.
- One device per process, and all GL work goes through one lock. Calls from several threads are correct but take turns.
- OpenGL ES only drivers are not used; the passes need desktop OpenGL 3.3 core.
- Every call uploads, draws, and reads back synchronously. This is about being correct on a device first, not about speed. On llvmpipe here, a 1024x768 target with a large blurred shadow, a full-size radial gradient, and a scaled 640x360 blit took about the same time on the device as on the CPU (81 ms against 91 ms, release build). Real hardware has not been measured.

### What was checked, and how

- Linux: all of it ran on Mesa's llvmpipe, a software OpenGL driver, through EGL 1.5 surfaceless. The device and CPU bytes are equal in every test below. It has not been run on real GPU hardware. A hardware driver that rounds differently should fail the self-test and stay on the CPU. The self-test is one small scene, though, so it cannot prove that a driver matches the CPU everywhere.
- Windows: the Direct3D 11 path is built for `x86_64-pc-windows-gnu` and passes clippy, and the constant buffer layout is checked against `passes.hlsl` by a test that runs everywhere. It has never been run on a Windows machine, so its tests have never executed. Until they have, the self-test is what keeps it from being used if it is wrong.
- Every GLSL pass and HLSL entry point passes `glslangValidator`.

### Since 0.8.0

The shadow mask is now blurred in fixed point (multiples of 1/32768) instead of in floating point, so a device can give the same bytes as the CPU. CPU shadow output changed slightly: in random tests about 0.04% of bytes differ from 0.8.0, alpha and premultiplied color by at most one code value. The color of pixels with alpha 2 or less can change more, where a tiny coverage now rounds to zero.

### The probe and processes

The probe is cached per process, not per machine. It now also makes a context, compiles the passes, and draws the self-test, which took about 0.1 seconds on llvmpipe here. When the surfaceless platform is missing, the first `GpuDevice::open`, `Compositor::open`, or `probe_hardware` in a process initializes the default EGL display, which connects that process to the display server until it exits. A program that runs one renderer process per tab should probe once in the main process and pass the decision to each renderer, for example by calling `Compositor::software()` there when the main process picked software, rather than letting every renderer open its own connection and device.

## What it does

- `Compositor` takes a shadow, gradient, or blit, checks it, and draws it, on the device picked above or on the CPU. It can also hold any other `ComposeBackend`. `device_pass_draws` counts pass draws done on a device in this process, which shows whether calls really ran there.
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

The rest of the crate is built with `#![deny(unsafe_code)]`. Unsafe code is only in these three modules, which only wrap system calls. Each block has a comment saying why it is sound, every returned pointer, handle, length, and status is checked, and no raw pointer leaves the module.

- `src/sys_gl.rs` (Linux): `dlopen` and `dlsym` for `libEGL.so.1`, and for GL entry points that `eglGetProcAddress` does not return, `libOpenGL.so.0` or `libGL.so.1`.
  - EGL: `eglGetProcAddress`, `eglQueryString`, `eglGetPlatformDisplayEXT`, `eglGetDisplay`, `eglInitialize`, `eglBindAPI`, `eglChooseConfig`, `eglCreateContext`, `eglDestroyContext`, `eglCreatePbufferSurface`, `eglDestroySurface`, `eglMakeCurrent`, `eglGetCurrentContext`, `eglGetCurrentDisplay`, `eglGetCurrentSurface`.
  - OpenGL (42): `glGetError`, `glGetIntegerv`, `glPixelStorei`, `glViewport`, `glCreateShader`, `glShaderSource`, `glCompileShader`, `glGetShaderiv`, `glGetShaderInfoLog`, `glDeleteShader`, `glCreateProgram`, `glAttachShader`, `glLinkProgram`, `glGetProgramiv`, `glGetProgramInfoLog`, `glDeleteProgram`, `glUseProgram`, `glGetUniformLocation`, `glUniform1i`, `glUniform2i`, `glUniform4i`, `glUniform1f`, `glUniform2f`, `glUniform4f`, `glUniform1fv`, `glUniform4fv`, `glGenTextures`, `glDeleteTextures`, `glBindTexture`, `glActiveTexture`, `glTexParameteri`, `glTexImage2D`, `glGenFramebuffers`, `glDeleteFramebuffers`, `glBindFramebuffer`, `glFramebufferTexture2D`, `glCheckFramebufferStatus`, `glGenVertexArrays`, `glDeleteVertexArrays`, `glBindVertexArray`, `glDrawArrays`, `glReadPixels`.
  - The context is only current while the GL lock is held, and whatever was current before is made current again afterwards.
- `src/sys_d3d.rs` (Windows): `LoadLibraryA` and `GetProcAddress` for `D3D11CreateDevice`, then methods called through their vtable slots: `ID3D11Device` `CreateBuffer`, `CreateTexture2D`, `CreateShaderResourceView`, `CreateRenderTargetView`, `CreateVertexShader`, `CreatePixelShader`, `CreateRasterizerState`, `GetDeviceRemovedReason`; `ID3D11DeviceContext` `VSSetConstantBuffers`, `PSSetShaderResources`, `PSSetShader`, `VSSetShader`, `Draw`, `Map`, `Unmap`, `PSSetConstantBuffers`, `IASetPrimitiveTopology`, `OMSetRenderTargets`, `RSSetState`, `RSSetViewports`, `CopyResource`, `UpdateSubresource`; and `IUnknown::Release`, once per object.
- `src/sys.rs` (Windows): `LoadLibraryA` and `GetProcAddress` for `D3DCompile` and `D3D11CreateDevice`, the calls themselves, reading and releasing the compiler blobs, and releasing the probe device and context.

## Build

```
cargo test
```

Rust 2024, builds with Rust 1.85. No crates.io dependencies, and nothing to link at build time besides `libdl` on Linux and `kernel32` on Windows.

- `tests/gpu_equality.rs` draws 1080 seeded random shadows, gradients, and blits (odd sizes, padded strides, 1 to 16 stops, all frame formats, both filters) on the device and on the CPU and compares every byte, row padding included. Each case has a 20 second limit.
- `tests/robustness.rs` throws seeded random and broken input at the CPU and device paths for 1.5 seconds each and requires the same result from both. Set `ANVIL_ROBUSTNESS_MS` for a longer run.
- The unit tests in `src/gpu_gl.rs` force each failure (shaders that do not compile, a context that cannot be made, a draw that fails partway) and check that the output still equals the CPU output.

Tests that need a device print `skipped: no usable device here (...)` and pass when there is none, so they do not fail on a machine without OpenGL. On Linux without a GPU, Mesa's llvmpipe is enough to run them.

## The other repositories

The browser is [anvil-browser](https://github.com/buntatoes/anvil-browser) (Apache-2.0). The page engine is `anvil-engine` (proprietary, not MPL-2.0). These libraries sit next to both. None of them come from crates.io.

| Repository | What it is |
| --- | --- |
| [anvil-cipher](https://github.com/buntatoes/anvil-cipher) | AES-128, AES-256, CTR, AES-256-GCM |
| [anvil-compress](https://github.com/buntatoes/anvil-compress) | DEFLATE, zlib, gzip |
| [anvil-image](https://github.com/buntatoes/anvil-image) | PNG, JPEG, GIF, WebP, ICO, BMP, PPM |
| [anvil-json](https://github.com/buntatoes/anvil-json) | JSON |
| [anvil-ttf](https://github.com/buntatoes/anvil-ttf) | TrueType and OpenType (`glyf`, `CFF `, `kern`, `GPOS`) |
| [anvil-gpu](https://github.com/buntatoes/anvil-gpu) | Drawing of boxes, shadows, gradients, and NV12, on the CPU or a GPU |
| [anvil-video](https://github.com/buntatoes/anvil-video) | CPU frames and PCM/ADPCM audio in a small MP4 container |

## License

MPL-2.0. Copyright (c) 2026 Buntato.
