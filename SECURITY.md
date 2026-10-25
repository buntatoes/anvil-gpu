# Security

Report problems in private:

https://github.com/buntatoes/anvil-gpu/security/advisories/new

The browser repository takes the same kind of report:

https://github.com/buntatoes/anvil-browser/security/advisories/new

Do not put the details in a public issue.

## What this is

Drawing runs on the CPU. `GpuDevice` counts 16 by 16 tiles and shades with `render_frame_rgba`. On Linux it probes `libEGL.so.1` and does not open a context when that probe fails. On Windows, `compile_hlsl_entry` and `probe_and_compile_d3d11_compositor` compile `shaders/compositor.hlsl` with `d3dcompiler_47.dll` and check whether a Direct3D 11 device opens. Quad drawing itself still runs on the CPU.

There is no mapped GPU memory and no video decoder in this crate.

Version 0.7.0 is current.
