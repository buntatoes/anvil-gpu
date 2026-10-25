# Security

Report problems in private:

https://github.com/buntatoes/anvil-gpu/security/advisories/new

The browser repository takes the same kind of report:

https://github.com/buntatoes/anvil-browser/security/advisories/new

Do not put the details in a public issue.

## What this is

Drawing runs on the CPU. `Compositor` draws shadows, gradients, and frame blits, and `GpuDevice` shades quads in 16 by 16 tiles. Every size, stride, plane length, and value is checked before anything is drawn. On Linux the crate loads `libEGL.so.1` and calls `eglInitialize` once per process, without opening a context. On Windows it compiles `shaders/compositor.hlsl` and `shaders/passes.hlsl` with `d3dcompiler_47.dll` and checks whether a Direct3D 11 device opens. Nothing is drawn on a GPU.

All unsafe code is in `src/sys.rs`, which holds those system calls and nothing else.

There is no mapped GPU memory and no video decoder in this crate.

Version 0.8.0 is current.
