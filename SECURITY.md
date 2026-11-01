# Security

Report problems in private:

https://github.com/buntatoes/anvil-gpu/security/advisories/new

Do not put the details in a public issue.

## What is checked

- Target and frame dimensions are capped at `MaxSide` (`16,384` pixels) and validated against row strides and buffer lengths before any draw.
- Non-finite (`NaN` or `Inf`) coordinates, negative radii/sizes, and out-of-range gradient stop counts return `ErrBadInput` without modifying target pixels.
- `Compositor` saves the affected target bounds before invoking any device backend and restores them transactionally before falling back to `SoftwareBackend` on any error.

Version 0.12.0 is current.
