# Security Policy

## Proof of Concept (PoC) Notice

`flux-gpu` is a standalone, zero-dependency **Proof of Concept (PoC)** 2D signed-distance-field (SDF) quad compositor, `NV12` BT.709 GPU video surface pipeline, shelf-packing RGBA texture atlas, and HLSL/GLSL shader library built from scratch as part of the [**Flux Browser**](https://github.com/buntatoes/flux-browser) engine ecosystem.

---

## Supported Versions

| Version | Supported |
| :--- | :---: |
| `v.04` (Latest) | Yes |
| `< v.04` | No |

---

## Reporting a Vulnerability

Please **do not** report security vulnerabilities through public GitHub issues or pull requests.

- **GitHub Private Vulnerability Reporting (Recommended)**: [`https://github.com/buntatoes/flux-gpu/security/advisories/new`](https://github.com/buntatoes/flux-gpu/security/advisories/new)
- **Central Ecosystem Security Portal**: [`https://github.com/buntatoes/flux-browser/security/advisories/new`](https://github.com/buntatoes/flux-browser/security/advisories/new)
- **Author Attribution**: **Buntos** ([@buntatoes](https://github.com/buntatoes) · `Buntos <324143321+buntatoes@users.noreply.github.com>`)

*(Note: `324143321+buntatoes@users.noreply.github.com` is a GitHub privacy `noreply` commit-attribution address and does not accept direct inbound email; please use GitHub Private Vulnerability Reporting above to contact the maintainer privately.)*

---

## Security Controls in `flux-gpu`

- **Bounded Atlas & Surface Allocations**:
  - Validates `TextureAtlas` shelf regions, UV coordinates, and `Nv12Surface` luma (`Y`) / chroma (`UV`) plane dimensions before buffer allocation or sampling.
- **Deterministic Scissor Clipping & Safe Rust**:
  - Clamps all quad rasterization and scissor clip rectangles to framebuffer bounds in 100% safe Rust.
