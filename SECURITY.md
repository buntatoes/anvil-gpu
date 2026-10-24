# Security

Report problems in private:

https://github.com/buntatoes/anvil-gpu/security/advisories/new

The browser repository takes the same kind of report:

https://github.com/buntatoes/anvil-browser/security/advisories/new

Do not put the details in a public issue.

## What this is

Drawing runs on the CPU. The shader files are not compiled and are not sent to a driver.

There is no device handle, no mapped GPU memory, and no video decoder in this crate.

Version 0.5.0 is current.
