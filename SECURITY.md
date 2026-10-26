# Security

Report problems in private:

https://github.com/buntatoes/anvil-gpu/security/advisories/new

The browser repository takes the same kind of report:

https://github.com/buntatoes/anvil-browser/security/advisories/new

Do not put the details in a public issue.

## What this is

Drawing runs on the CPU, or on a GPU when one passes the self-test. `Compositor` draws shadows, gradients, and frame blits, and `GpuDevice` shades quads in 16 by 16 tiles on the CPU. Every size, stride, plane length, and value is checked before anything is drawn, on either path.

On Linux the crate loads `libEGL.so.1` and the OpenGL entry points with `dlopen`, makes an offscreen OpenGL 3.3 core context, and runs the passes in `shaders/passes.glsl`. On Windows it loads `d3d11.dll` and `d3dcompiler_47.dll`, makes a Direct3D 11 device, and runs `shaders/passes.hlsl`; both the Linux OpenGL 3.3 core path and the Windows Direct3D 11 hardware path are validated bit-for-bit against the CPU across all 2020 equality cases and seeded differential scenes. Pixels go to the device as textures and come back with `glReadPixels` or a staging texture copy. No pixel data stays on the device after a call; only the compiled passes do.

All unsafe code is in `src/sys.rs`, `src/sys_gl.rs`, and `src/sys_d3d.rs`, which hold those system calls and nothing else.

There is no persistently mapped GPU memory and no video decoder in this crate.

Version 0.9.1 is current.
