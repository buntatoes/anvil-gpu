# Security

Report problems in private:

https://github.com/buntatoes/anvil-gpu/security/advisories/new

The browser repository takes the same kind of report:

https://github.com/buntatoes/anvil-browser/security/advisories/new

Do not put the details in a public issue.

## What this is

Drawing runs on the CPU. `GpuDevice` counts 16 by 16 tiles and shades with `render_frame_rgba`. On Linux it probes `libEGL.so.1` and does not open a context when that probe fails. The shader files are not compiled and are not sent to a driver.

There is no mapped GPU memory and no video decoder in this crate.

Version 0.6.0 is current.
