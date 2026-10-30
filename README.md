<div align="center">
  <h1>Anvil GPU</h1>
  <p>Compositor passes and batched scene command buffers: blurred shadows, gradients, video frame blits, rounded SDF rects, glyph coverage, and tile quads, on the CPU or a GPU.</p>
  <p>
    <img src="https://img.shields.io/badge/release-v.11-e8c37a?style=flat-square" alt="v.11" />
    <img src="https://img.shields.io/badge/status-Proof_of_Concept-d4a359?style=flat-square" alt="Proof of Concept" />
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-PolyForm_Noncommercial_1.0.0_%7C_Paid_Commercial_%28Non--Compete%29-blue?style=flat-square" alt="PolyForm Noncommercial 1.0.0 | Paid Commercial (Non-Compete)" /></a>
    <a href="./SECURITY.md"><img src="https://img.shields.io/badge/security-policy-2ea44f?style=flat-square" alt="Security" /></a>
    <img src="https://img.shields.io/badge/rust-2024-orange?style=flat-square" alt="Rust 2024" />
  </p>
</div>

Version 0.11.0 (`v.11`).

Drawing for the compositor: batched scene command buffers (`GpuSceneBatch`, `GpuSceneOp`, `Compositor::execute_scene_batch`), blurred box shadows, linear and radial gradients, RGBA/NV12/I420 video and canvas frame blits (`Nv12Surface`, `I420Surface`), rounded SDF boxes, 8-bit glyph coverage compositing, plus the tile quad drawing (`GpuDevice`). The CPU passes are the reference. The shader passes are written in GLSL and HLSL, and run on a GPU when one is there and passes a self-test against the CPU. Every test in this repository gets the same bytes from the device as from the CPU on both Linux (Mesa llvmpipe, OpenGL 3.3 core) and Windows (Direct3D 11 hardware GPU). Proof of concept.

## What runs where

| Pass | CPU | Linux, OpenGL 3.3 core | Windows, Direct3D 11 |
| --- | --- | --- | --- |
| Box shadow with blur | `draw_box_shadow` | runs (mask, six blur passes, paint) | runs (mask, six blur passes, paint) |
| Linear and radial gradients, up to 16 stops (including rounded-corner SDF gradients via `draw_rounded_gradient` / `fill_rounded_gradient` / `GpuSceneOp::RoundedGradient`) | `draw_gradient`, `draw_rounded_gradient` | runs | runs |
| RGBA, NV12, and I420 frame blit with scaling | `draw_blit` | runs | runs |
| Rounded boxes (`border_width == 0.0` in `GpuSceneBatch`), borders, two-stop gradients, atlas quads, and `GpuSceneBatch` clear/solid/rounded/glyph ops | `render_frame_rgba`, `GpuDevice::render_quads`, `Compositor::execute_scene_batch` | Shadow, gradient, rounded gradient, rounded solid (`border_width == 0.0`), and blit ops in `GpuSceneBatch` use the device; bordered rects and glyph coverage run on CPU | Shadow, gradient, rounded gradient, rounded solid (`border_width == 0.0`), and blit ops in `GpuSceneBatch` use the device; bordered rects and glyph coverage run on CPU |

`Compositor::box_shadow`, `fill_gradient`, and `blit_frame` use the device when there is one, and the CPU otherwise. `Compositor::software()` always uses the CPU.

### How a device is picked

`probe_hardware` runs once per process, and `Compositor::open` follows it:

1. Linux: load `libEGL.so.1` with `dlopen` and open a display. The surfaceless platform (`EGL_MESA_platform_surfaceless`) is tried first, so no display server is needed; otherwise the default display. Windows: load `d3d11.dll` and make a device at feature level 11_0, hardware first, then WARP.
2. Make an offscreen OpenGL 3.3 core context (surfaceless, or a 1x1 pbuffer), or use the Direct3D device, and build every pass from `shaders/passes.glsl` or `shaders/passes.hlsl` (compiled with `d3dcompiler_47.dll` and IEEE strictness).
3. Draw a self-test scene that covers every pass (blurred, sharp, and clipped shadows, both gradient shapes, all three frame formats with both filters, on a padded target) on the device and on the CPU, and compare every byte.

The device is used only if all three steps succeed. Otherwise everything is drawn on the CPU and `HardwareProbe::reason` says which step failed. Opening runs on a helper thread and counts as failed after 20 seconds, so a driver that hangs cannot hang the caller.

### Fallback

If a device call fails (a GL error, a failed `Map`, a removed device), the compositor puts back the pixels the call could have touched, drops the device, and draws that call and every later one on the CPU. That call then gets exactly the CPU bytes. `Compositor::fallbacks` counts how often that happened.

### Limits

- A call may make textures of at most 2^24 texels (64 MB for RGBA8 or R32F) each, no larger than the device's maximum side. Bigger calls are drawn on the CPU by the device backend, with the same result.
- Gradient coordinates or radii past 1e15 are drawn on the CPU, because f32 math on the device could overflow where the CPU does not.
- One shadow uses up to four textures at once (two R32F mask textures for the blur, the destination copy, and the output). The device does not stream rows the way the CPU passes do, so its memory grows with the shadow size up to that limit. Textures are freed at the end of each call. The CPU side also keeps a copy of the region for the upload and the read back.
- One device per process, and all GL or D3D11 work goes through one lock. Calls from several threads are correct but take turns.
- OpenGL ES only drivers are not used; the passes need desktop OpenGL 3.3 core.
- Opening a device has a 20 second limit, but a single draw does not. A driver that hangs in the middle of a draw hangs that call, and every other call waits on the device lock behind it.
- Every call uploads, draws, and reads back synchronously. This is about being correct on a device first, not about speed.

### What was checked, and how

- Linux: all of it ran on Mesa's llvmpipe, a software OpenGL driver, through EGL 1.5 surfaceless. The device and CPU bytes are equal in every test below.
- Windows: all of it ran on a physical Direct3D 11 hardware GPU (feature level 11_0) and on WARP. Every equality, property, and seeded differential test matches the CPU byte-for-byte.
- Which passes are exact by construction, and how floating-point division and square root are refined:
  - The shadow blur and the bilinear weights and sums of blits use integer math on both sides, so they cannot round differently on a device that follows the integer rules of GLSL and HLSL and reads 8-bit texels back as whole code values.
  - The shadow mask and paint, gradients, and the BT.709 conversion of nearest YUV blits use f32 math in the same order as the CPU. Because GPU hardware `div`/`rcp` and `sqrt` allow up to 1.0 ULP error (and FXC folds `/ 255.0` into `* (1.0 / 255.0)` unless inhibited), `passes.hlsl` and `passes.glsl` split 24-bit significands into 12-bit halves (`ExactMulSub`) and refine `IeeeDiv` and `IeeeSqrt` with one Newton-Raphson step to 0.5-ULP IEEE-754 rounding. That closed four 1-ULP differences found on Direct3D 11 hardware and the `seed 1234, case 83` gradient difference on llvmpipe.
- Every GLSL pass and HLSL entry point passes `glslangValidator`.

### CPU changes on the way to 0.9.0

The shadow, gradient, and blit passes first showed up in a 0.8.0 version that was never released. Before 0.9.0, the CPU output of three of them changed so that a device can give the same bytes. If you built from that 0.8.0 source, this is what moved:

- The shadow mask is blurred in fixed point (multiples of 1/32768) instead of in floating point. Over 3000 random shadows on targets of 1 to 80 pixels a side, 0.039% of all target bytes differ from the 0.8.0 source. (A separate check over 20,000 mixed scenes with a different generator found 0.019%; the share depends mostly on how much of each target the shadows cover.) Alpha and premultiplied color differ by at most one code value. The color of pixels with alpha 2 or less can change more, where a tiny coverage now rounds to zero.
- Bilinear blits use integer weights in 1/4096 steps, with the sample position rounded to that step, and integer sums. RGBA is mixed row by row in premultiplied color kept in 1/16 steps; YUV planes are mixed to 1/256 of a code value and converted with integer BT.709 coefficients in 1/8192 steps. Over 3000 random bilinear blits (frames up to 40 by 40, scaled to 1 to 120 pixels a side), 0.64% of the written bytes differ from the 0.8.0 source for RGBA, 0.74% for NV12 and 0.72% for I420. NV12 and I420 bytes, RGBA alpha, and RGBA color where the output alpha is above 2 differ by one code value at most. The color of RGBA pixels with output alpha 2 or less can change much more, because it is divided back out of such a tiny alpha: in that run 64 such bytes moved, by up to 243. Nearest blits did not change.
- Shadow paint and gradient colors are rounded as `floor(x + 0.5)`, the way the shaders do it, instead of with Rust's `f32::round`. The two only disagree when the value before rounding is exactly 0.5 - 2^-25, which is rare, and then one channel moves by one code value.

### The probe and processes

The probe is cached per process, not per machine. It now also makes a context, compiles the passes, and draws the self-test, which took about 0.1 seconds on llvmpipe here. When the surfaceless platform is missing, the first `GpuDevice::open`, `Compositor::open`, or `probe_hardware` in a process initializes the default EGL display, which connects that process to the display server until it exits. A program that runs one renderer process per tab should probe once in the main process and pass the decision to each renderer, for example by calling `Compositor::software()` there when the main process picked software, rather than letting every renderer open its own connection and device.

## What it does

- `Compositor` takes a shadow, gradient, or blit, checks it, and draws it, on the device picked above or on the CPU. It can also hold any other `ComposeBackend`. `device_pass_draws` counts pass draws done on a device in this process, which shows whether calls really ran there.
- `draw_box_shadow` draws the rounded box into a coverage mask, blurs it with three box passes along rows and three along columns (close to a gaussian with sigma `blur / 2`), then paints the color through it. Offset, spread, corner radius, and clip are close to CSS box shadows, but the blur is a three-box approximation of the CSS gaussian, and the corner radius simply grows by the spread. A sharp box keeps sharp corners when it grows. The mask is made and blurred a row at a time, so it is never held whole: at the largest blur on a 16384 pixel wide target the working memory stays under 60 MB.
- `draw_gradient` fills a rectangle with a linear gradient (start and end point) or an elliptical radial gradient, with 1 to 16 stops.
- `draw_blit` copies an RGBA, NV12, or I420 frame into a rectangle, with nearest or bilinear sampling. YUV uses limited-range BT.709, the same as `Nv12Surface`. A 1:1 nearest NV12 blit gives the same bytes as `Nv12Surface::to_rgba_bt709`; bilinear blits use the integer conversion described above, which can be one code value away from it. Bilinear sampling treats chroma as sitting at the center of each 2 by 2 luma block. Most BT.709 video sites chroma on the left edge of the block, so for that video the scaled chroma lands half a luma pixel to the right, which only shows as slight fringes on sharp color edges. The shader passes use the same siting and the same integer math.
- Targets are straight RGBA8 with a row stride (`RgbaTarget`). Sizes, strides, plane lengths, and values are checked first. Bad input returns `ComposeError` and draws nothing. Each side is capped at 16384 pixels and blur at 256.
- `GpuDevice::render_quads` counts the 16 by 16 tiles a quad touches, then shades the frame once. The bytes match `render_frame_rgba`.
- `composite_rgba_frame`, `convert_rgba_on_device`, and `present_nv12_on_device` work as before and run on the CPU.
- `sdf_rounded_box`, `sdf_box_shadow_alpha`, `sample_linear_gradient`, `sample_multi_stop_gradient`, and `sample_radial_gradient` are the shared math.
- `TextureAtlas` packs rectangles and returns UV bounds. `Nv12Surface` and `I420Surface` convert RGBA to NV12/I420 and back.
- `COMPOSITOR_GLSL`, `COMPOSITOR_HLSL`, `PASSES_GLSL`, and `PASSES_HLSL` are the shader sources.

## Unsafe code

The rest of the crate is built with `#![deny(unsafe_code)]`. Unsafe code is only in these three modules, which only wrap system calls. Each block has a comment saying why it is sound, every returned pointer, handle, length, and status is checked, and no raw pointer leaves the module.

- `src/sys_gl.rs` (Linux): `dlopen` and `dlsym` for `libEGL.so.1`, and for GL entry points that `eglGetProcAddress` does not return, `libOpenGL.so.0` or `libGL.so.1`.
  - EGL: `eglGetProcAddress`, `eglQueryString`, `eglGetPlatformDisplayEXT`, `eglGetDisplay`, `eglInitialize`, `eglBindAPI`, `eglQueryAPI`, `eglChooseConfig`, `eglCreateContext`, `eglDestroyContext`, `eglCreatePbufferSurface`, `eglDestroySurface`, `eglMakeCurrent`, `eglGetCurrentContext`, `eglGetCurrentDisplay`, `eglGetCurrentSurface`.
  - OpenGL (42): `glGetError`, `glGetIntegerv`, `glPixelStorei`, `glViewport`, `glCreateShader`, `glShaderSource`, `glCompileShader`, `glGetShaderiv`, `glGetShaderInfoLog`, `glDeleteShader`, `glCreateProgram`, `glAttachShader`, `glLinkProgram`, `glGetProgramiv`, `glGetProgramInfoLog`, `glDeleteProgram`, `glUseProgram`, `glGetUniformLocation`, `glUniform1i`, `glUniform2i`, `glUniform4i`, `glUniform1f`, `glUniform2f`, `glUniform4f`, `glUniform1fv`, `glUniform4fv`, `glGenTextures`, `glDeleteTextures`, `glBindTexture`, `glActiveTexture`, `glTexParameteri`, `glTexImage2D`, `glGenFramebuffers`, `glDeleteFramebuffers`, `glBindFramebuffer`, `glFramebufferTexture2D`, `glCheckFramebufferStatus`, `glGenVertexArrays`, `glDeleteVertexArrays`, `glBindVertexArray`, `glDrawArrays`, `glReadPixels`.
  - The context is only current while the GL lock is held, and whatever was current before is made current again afterwards, along with the thread's EGL rendering API, so a caller using OpenGL ES on the same thread keeps it.
- `src/sys_d3d.rs` (Windows): `LoadLibraryA` and `GetProcAddress` for `D3D11CreateDevice`, then methods called through their vtable slots: `ID3D11Device` `CreateBuffer`, `CreateTexture2D`, `CreateShaderResourceView`, `CreateRenderTargetView`, `CreateVertexShader`, `CreatePixelShader`, `CreateRasterizerState`, `GetDeviceRemovedReason`; `ID3D11DeviceContext` `VSSetConstantBuffers`, `PSSetShaderResources`, `PSSetShader`, `VSSetShader`, `Draw`, `Map`, `Unmap`, `PSSetConstantBuffers`, `IASetPrimitiveTopology`, `OMSetRenderTargets`, `RSSetState`, `RSSetViewports`, `CopyResource`, `UpdateSubresource`; and `IUnknown::Release`, once per object.
- `src/sys.rs` (Windows): `LoadLibraryA` and `GetProcAddress` for `D3DCompile` and `D3D11CreateDevice`, the calls themselves, reading and releasing the compiler blobs, and releasing the probe device and context.

## Build

```
cargo test
```

Rust 2024 edition, Rust 1.99.1 or newer. No crates.io dependencies, and nothing to link at build time besides `libdl` on Linux and `kernel32` on Windows.

- `tests/gpu_equality.rs` draws 2020 seeded random cases on the device and on the CPU and compares every byte, row padding included: 400 shadows, 320 gradients with 1 to 16 stops, 360 blits of all formats with both filters, 900 bilinear RGBA, NV12, and I420 blits scaled far down, a little down, and up to eight times up, with 1-pixel frames, edge clamping, and rectangles off every side, and 40 with the geometry of a bilinear case that used to differ. Each case has a 20 second limit.
- `tests/device_differential.rs` draws seeded scenes of 2 to 8 calls on targets up to 301 by 211 on both sides for 1.5 seconds per test, broken inputs included, and also replays the scenes that used to be one code value apart. Set `ANVIL_SEED` to run different scenes and `ANVIL_FIRST_CASE` to replay one.
- `tests/compose_properties.rs` checks what the CPU passes draw, not just that they finish: a flat gradient fills exactly its clipped rectangle, a shadow changes nothing past its blur or clip, a centered shadow is mirror symmetric, and NV12, I420, padded, and packed frames give the same pixels. Each test runs for 1.5 seconds; set `ANVIL_ROBUSTNESS_MS` for a longer run.
- `tests/robustness.rs` throws seeded random and broken input at the CPU and device paths for 1.5 seconds each and requires the same result from both. Set `ANVIL_ROBUSTNESS_MS` for a longer run.
- The unit tests in `src/gpu_gl.rs` and `src/gpu_d3d.rs` force each failure (shaders that do not compile, a context that cannot be made, a draw that fails partway) and check that the output still equals the CPU output.

Tests that need a device pass when there is none, so they do not fail on a machine without OpenGL or Direct3D 11. The equality and scene tests then write a `SKIPPED` line that shows in a normal `cargo test` run; set `ANVIL_REQUIRE_DEVICE=1` to make a missing device fail them instead. On Linux without a GPU, Mesa's llvmpipe is enough to run them.

## The other repositories

The browser is [anvil-browser](https://github.com/buntatoes/anvil-browser) and the page engine is [anvil-engine](https://github.com/buntatoes/anvil-engine), built alongside the nine companion libraries below. None of them come from crates.io.

| Repository | What it is |
| --- | --- |
| [anvil-cipher](https://github.com/buntatoes/anvil-cipher) | AES-128, AES-256, CTR, AES-256-GCM, SHA-256, HMAC-SHA256, HKDF-SHA256 |
| [anvil-compress](https://github.com/buntatoes/anvil-compress) | DEFLATE, zlib, gzip |
| [anvil-hpack](https://github.com/buntatoes/anvil-hpack) | HPACK header compression for HTTP/2 |
| [anvil-image](https://github.com/buntatoes/anvil-image) | PNG, JPEG, GIF, WebP (`VP8L` + `VP8 `), ICO, BMP, PPM |
| [anvil-json](https://github.com/buntatoes/anvil-json) | JSON |
| [anvil-pqc](https://github.com/buntatoes/anvil-pqc) | ML-KEM-768, X25519, X25519MLKEM768, SHA-3 |
| [anvil-ttf](https://github.com/buntatoes/anvil-ttf) | TrueType and OpenType (`glyf`, `CFF `, `kern`, `GPOS`) |
| [anvil-gpu](https://github.com/buntatoes/anvil-gpu) | Compositor passes and batched scene command buffers for boxes, shadows, gradients, NV12, and I420 |
| [anvil-video](https://github.com/buntatoes/anvil-video) | CPU frames (AVID/ADCT) and PCM/ADPCM/FLAC/MDCT audio in MP4, fMP4, and WebM containers |

## License

Dual-licensed under the **PolyForm Noncommercial License 1.0.0** and a **Paid Commercial License with a Non-Compete Clause**. Copyright (c) 2026 Buntato. See [LICENSE](./LICENSE).
