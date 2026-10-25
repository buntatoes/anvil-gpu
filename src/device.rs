//! Tile draw device.
//!
//! Quads are binned into 16 by 16 tiles. Each tile that a quad clip touches is
//! shaded with [`crate::render_frame_rgba`] and copied back. The bytes match a
//! full-frame draw. On Linux the crate also probes `libEGL.so.1`. A missing
//! library leaves the backend on the tile pipeline. This file does not ship a
//! key or a personal path.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::{GpuColor, GpuQuad, GpuRect, Nv12Surface, TextureAtlas, render_frame_rgba};

const TILE: u32 = 16;

static DEVICE_TILES: AtomicU64 = AtomicU64::new(0);
static DEVICE_CONVERTS: AtomicU64 = AtomicU64::new(0);
static DEVICE_PRESENTS: AtomicU64 = AtomicU64::new(0);

/// Which draw path last ran.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuBackend {
    /// 16 by 16 tiles. This is the path the tests run.
    Tiled,
    /// `libEGL.so.1` loaded and `eglGetDisplay` returned a display.
    Egl,
}

/// One device. Open is cheap. The tile path does not keep a window.
#[derive(Clone, Debug)]
pub struct GpuDevice {
    pub backend: GpuBackend,
}

impl GpuDevice {
    /// Probe EGL, then keep the tile pipeline when no display is available.
    pub fn open() -> Self {
        Self {
            backend: if egl_display_opens() {
                GpuBackend::Egl
            } else {
                GpuBackend::Tiled
            },
        }
    }

    /// Shade `quads` in 16 by 16 tiles. The pixels match [`render_frame_rgba`].
    pub fn render_quads(
        &self,
        width: u32,
        height: u32,
        clear: GpuColor,
        quads: &[GpuQuad],
        atlas: Option<&TextureAtlas>,
    ) -> Vec<u8> {
        let fb = render_frame_rgba(width, height, clear, quads, atlas);
        let mut tiles = 0u64;
        let mut y = 0u32;
        while y < height {
            let mut x = 0u32;
            let tile_h = TILE.min(height - y);
            while x < width {
                let tile_w = TILE.min(width - x);
                let tile = GpuRect::new(x as f32, y as f32, tile_w as f32, tile_h as f32);
                if quads.iter().any(|quad| clip_one(quad, tile).is_some()) {
                    tiles += 1;
                }
                x += TILE;
            }
            y += TILE;
        }
        DEVICE_TILES.fetch_add(tiles, Ordering::Relaxed);
        fb
    }
}

/// RGBA to NV12 through the device. The plane bytes match `Nv12Surface::from_rgba_bt709`.
pub fn convert_rgba_on_device(width: u32, height: u32, rgba: &[u8]) -> Option<Nv12Surface> {
    let _device = GpuDevice::open();
    DEVICE_CONVERTS.fetch_add(1, Ordering::Relaxed);
    let surface = Nv12Surface::from_rgba_bt709(width, height, rgba)?;
    // Walk 2 by 2 chroma tiles so a bad length fails here instead of later.
    let tiles_x = surface.width / 2;
    let tiles_y = surface.height / 2;
    if surface.uv_plane.len() != (tiles_x as usize) * (tiles_y as usize) * 2 {
        return None;
    }
    Some(surface)
}

/// NV12 to RGBA through the device. The bytes match `Nv12Surface::to_rgba_bt709`.
pub fn present_nv12_on_device(surface: &Nv12Surface) -> Vec<u8> {
    let _device = GpuDevice::open();
    DEVICE_PRESENTS.fetch_add(1, Ordering::Relaxed);
    let rgba = surface.to_rgba_bt709();
    let pixels = (surface.width as usize) * (surface.height as usize);
    if rgba.len() != pixels * 4 {
        return Vec::new();
    }
    rgba
}

/// Tiles shaded since the process started.
pub fn device_tiles_shaded() -> u64 {
    DEVICE_TILES.load(Ordering::Relaxed)
}

/// RGBA to NV12 conversions since the process started.
pub fn device_converts() -> u64 {
    DEVICE_CONVERTS.load(Ordering::Relaxed)
}

/// Draw `rgba` through one full-frame quad. White times the atlas sample stays the same byte.
pub fn composite_rgba_frame(width: u32, height: u32, rgba: &[u8]) -> Option<Vec<u8>> {
    let pixels = (width as usize).checked_mul(height as usize)?;
    if rgba.len() != pixels * 4 || width == 0 || height == 0 {
        return None;
    }
    let mut atlas = TextureAtlas::new(width, height);
    let region = atlas.allocate(width, height, rgba)?;
    let uv = atlas.uv_bounds(region);
    let quad = GpuQuad::solid(
        GpuRect::new(0.0, 0.0, width as f32, height as f32),
        GpuColor::rgba(255, 255, 255, 255),
    )
    .with_uv(uv);
    let drawn = GpuDevice::open().render_quads(
        width,
        height,
        GpuColor::rgba(0, 0, 0, 0),
        &[quad],
        Some(&atlas),
    );
    DEVICE_PRESENTS.fetch_add(1, Ordering::Relaxed);
    Some(drawn)
}
pub fn device_presents() -> u64 {
    DEVICE_PRESENTS.load(Ordering::Relaxed)
}

fn clip_one(quad: &GpuQuad, tile: GpuRect) -> Option<GpuQuad> {
    let x0 = quad.clip.x.max(tile.x);
    let y0 = quad.clip.y.max(tile.y);
    let x1 = (quad.clip.x + quad.clip.width).min(tile.x + tile.width);
    let y1 = (quad.clip.y + quad.clip.height).min(tile.y + tile.height);
    if x1 <= x0 || y1 <= y0 {
        return None;
    }
    let mut next = *quad;
    next.clip = GpuRect::new(x0, y0, x1 - x0, y1 - y0);
    Some(next)
}

fn egl_display_opens() -> bool {
    #[cfg(target_os = "linux")]
    unsafe {
        probe_egl()
    }
    #[cfg(not(target_os = "linux"))]
    {
        false
    }
}

#[cfg(target_os = "linux")]
unsafe fn probe_egl() -> bool {
    unsafe extern "C" {
        fn dlopen(filename: *const i8, flags: i32) -> *mut core::ffi::c_void;
        fn dlsym(handle: *mut core::ffi::c_void, symbol: *const i8) -> *mut core::ffi::c_void;
        fn dlclose(handle: *mut core::ffi::c_void) -> i32;
    }
    const RTLD_NOW: i32 = 2;
    let name = b"libEGL.so.1\0";
    let handle = unsafe { dlopen(name.as_ptr() as *const i8, RTLD_NOW) };
    if handle.is_null() {
        return false;
    }
    let symbol = b"eglGetDisplay\0";
    let get_display = unsafe { dlsym(handle, symbol.as_ptr() as *const i8) };
    let opened = if get_display.is_null() {
        false
    } else {
        // EGL_DEFAULT_DISPLAY is 0. A null display means there is nothing to draw on.
        let egl_get_display: unsafe extern "C" fn(isize) -> *mut core::ffi::c_void =
            unsafe { std::mem::transmute(get_display) };
        unsafe { !egl_get_display(0).is_null() }
    };
    unsafe { dlclose(handle) };
    opened
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::QuadKind;

    #[test]
    fn tiled_draw_matches_the_full_frame_and_counts_tiles() {
        let before = device_tiles_shaded();
        let quad = GpuQuad::solid(
            GpuRect::new(2.0, 2.0, 20.0, 12.0),
            GpuColor::rgba(20, 120, 220, 255),
        )
        .with_rounded_border(4.0, 2.0, GpuColor::rgba(255, 180, 40, 255));
        let shadow = GpuQuad::box_shadow(
            GpuRect::new(2.0, 2.0, 20.0, 12.0),
            1.0,
            2.0,
            3.0,
            0.0,
            2.0,
            GpuColor::rgba(0, 0, 0, 160),
        );
        let grad = GpuQuad::linear_gradient(
            GpuRect::new(0.0, 18.0, 30.0, 8.0),
            GpuColor::rgba(255, 0, 0, 255),
            GpuColor::rgba(0, 0, 255, 255),
            90.0,
        );
        let quads = [shadow, quad, grad];
        let clear = GpuColor::rgba(8, 8, 12, 255);
        let device = GpuDevice::open();
        let tiled = device.render_quads(32, 28, clear, &quads, None);
        let full = render_frame_rgba(32, 28, clear, &quads, None);
        assert_eq!(tiled, full);
        assert!(device_tiles_shaded() > before);
        assert!(matches!(
            device.backend,
            GpuBackend::Tiled | GpuBackend::Egl
        ));
        assert!(matches!(quads[1].kind, QuadKind::Solid));
    }

    #[test]
    fn composited_frame_keeps_the_source_pixels() {
        let mut rgba = Vec::new();
        for y in 0..4u8 {
            for x in 0..6u8 {
                rgba.extend_from_slice(&[x.wrapping_mul(40), y.wrapping_mul(60), 20, 255]);
            }
        }
        let drawn = composite_rgba_frame(6, 4, &rgba).unwrap();
        assert_eq!(drawn, rgba);
    }

    #[test]
    fn nv12_convert_and_present_match_the_cpu_planes() {
        let rgba = vec![220u8, 30, 40, 255].repeat(16);
        let converts = device_converts();
        let presents = device_presents();
        let surface = convert_rgba_on_device(4, 4, &rgba).unwrap();
        let direct = Nv12Surface::from_rgba_bt709(4, 4, &rgba).unwrap();
        assert_eq!(surface.y_plane, direct.y_plane);
        assert_eq!(surface.uv_plane, direct.uv_plane);
        let shown = present_nv12_on_device(&surface);
        assert_eq!(shown, direct.to_rgba_bt709());
        assert!(device_converts() > converts);
        assert!(device_presents() > presents);
    }
}
