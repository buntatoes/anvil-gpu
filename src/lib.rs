// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 Buntato
//! CPU drawing of rounded boxes, box shadows, linear gradients, a shelf-packed RGBA atlas, and RGBA/NV12 conversion.
//!
//! CPU entry points:
//!
//! - [`render_frame_rgba`] rasterizes [`GpuQuad`] values into a straight RGBA8 framebuffer.
//! - [`sdf_rounded_box`], [`sdf_box_shadow_alpha`], and [`sample_linear_gradient`] are the signed-distance, shadow, and gradient helpers.
//! - [`TextureAtlas`] shelf-packs straight RGBA8 images and samples them.
//! - [`bt709_rgba_to_yuv`], [`bt709_yuv_to_rgba`], and [`Nv12Surface`] convert between straight RGBA8 and limited-range BT.709 NV12.
//!
//! `anvil-raster` calls the CPU SDF helpers for shadows, rounded boxes, and gradients.
//! `anvil-video` uses [`Nv12Surface`] for CPU color conversion.
//!
//! [`COMPOSITOR_HLSL`] and [`COMPOSITOR_GLSL`] are `include_str!` source strings.
//! [`device`] shades those quads in 16 by 16 tiles and probes `libEGL.so.1` on Linux.

mod device;
pub use device::{
    GpuBackend, GpuDevice, composite_rgba_frame, convert_rgba_on_device, device_converts,
    device_presents, device_tiles_shaded, present_nv12_on_device,
};

/// HLSL Shader Model 5.0 source text from `shaders/compositor.hlsl`.
///
/// Embedded with `include_str!`. This crate does not compile or run it.
pub const COMPOSITOR_HLSL: &str = include_str!("../shaders/compositor.hlsl");

/// GLSL 330 core source text from `shaders/compositor.glsl`.
///
/// Embedded with `include_str!`. This crate does not compile or run it.
pub const COMPOSITOR_GLSL: &str = include_str!("../shaders/compositor.glsl");

/// 8-bit straight RGBA. Channels are not premultiplied.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuColor {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl GpuColor {
    /// Stores the four straight channels as given.
    pub const fn rgba(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }
}

/// Axis-aligned rectangle. `x` and `y` are the minimum corner; size grows toward +x and +y.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuRect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl GpuRect {
    /// Builds a rectangle from its minimum corner and its size.
    pub const fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// Reports whether a point lies in this rectangle.
    ///
    /// The far edges are exclusive: `x + width` and `y + height` are outside.
    pub fn contains(self, px: f32, py: f32) -> bool {
        // Same pixel space as the fields. This compares the raw point, not a pixel center.
        px >= self.x && py >= self.y && px < self.x + self.width && py < self.y + self.height
    }
}

/// Converts 8-bit RGB to limited-range BT.709 YCbCr.
///
/// Luma uses Kr 0.2126, Kg 0.7152, and Kb 0.0722, scaled by 219/255 and offset by 16.
/// It is then rounded and clamped to 16..=235.
/// Cb and Cr use the matching BT.709 chroma axes, scaled by 224/255 and offset by 128,
/// then rounded and clamped to 16..=240. Neutral chroma is 128.
/// This is studio swing, not a full-range matrix with luma 0..=255.
pub fn bt709_rgba_to_yuv(r: u8, g: u8, b: u8) -> (u8, u8, u8) {
    let rf = r as f32;
    let gf = g as f32;
    let bf = b as f32;
    // Limited-range luma. Peak white lands on 235, not 255.
    let y = (16.0 + 0.182586 * rf + 0.614231 * gf + 0.062007 * bf)
        .round()
        .clamp(16.0, 235.0) as u8;
    // Limited-range Cb. The blue coefficient is (224/255) * 0.5.
    let u = (128.0 - 0.100644 * rf - 0.338572 * gf + 0.439216 * bf)
        .round()
        .clamp(16.0, 240.0) as u8;
    // Limited-range Cr, same 224/255 scale, red on the positive axis.
    let v = (128.0 + 0.439216 * rf - 0.398942 * gf - 0.040274 * bf)
        .round()
        .clamp(16.0, 240.0) as u8;
    (y, u, v)
}

/// Converts limited-range BT.709 YCbCr to straight RGBA8.
///
/// Luma below 16 is treated as 16. `(Y - 16)` is scaled by 255/219.
/// Cb and Cr are centered at 128 and expanded with the BT.709 inverse,
/// including the 255/224 studio-swing scale. Each channel is rounded and clamped to 0..=255.
/// Alpha is always 255. This is not a full-range (luma 0..=255) inverse.
pub fn bt709_yuv_to_rgba(y: u8, u: u8, v: u8) -> GpuColor {
    // Expand studio luma to full 8-bit swing. Values under 16 contribute nothing.
    let yf = ((y as f32) - 16.0).max(0.0) * 1.164384;
    let uf = (u as f32) - 128.0;
    let vf = (v as f32) - 128.0;
    let r = (yf + 1.792741 * vf).round().clamp(0.0, 255.0) as u8;
    let g = (yf - 0.213249 * uf - 0.532909 * vf)
        .round()
        .clamp(0.0, 255.0) as u8;
    let b = (yf + 2.112402 * uf).round().clamp(0.0, 255.0) as u8;
    GpuColor::rgba(r, g, b, 255)
}

/// Semi-planar 4:2:0 NV12 buffer in limited-range BT.709.
///
/// CPU memory only: this type does not allocate a GPU texture.
/// `y_plane` is one luma byte per pixel. `uv_plane` is interleaved Cb, Cr with one pair per 2x2 block.
/// Code 16 is black luma, 235 is peak white, and 128 is neutral chroma.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Nv12Surface {
    /// Width in pixels. [`Nv12Surface::new`] forces this to an even value of at least 2.
    pub width: u32,
    /// Height in pixels. [`Nv12Surface::new`] forces this to an even value of at least 2.
    pub height: u32,
    /// Limited-range luma, row-major, `width * height` bytes.
    pub y_plane: Vec<u8>,
    /// Interleaved Cb, Cr, row-major, `width * height / 2` bytes.
    pub uv_plane: Vec<u8>,
}

impl Nv12Surface {
    /// Allocates an even-sized surface filled with limited-range black.
    ///
    /// Dimensions below 2 become 2. Any odd size is rounded down to the next even number
    /// (`max(n, 2)` with the low bit cleared). Luma bytes are 16 and chroma bytes are 128.
    pub fn new(width: u32, height: u32) -> Self {
        let w = width.max(2) & !1;
        let h = height.max(2) & !1;
        let y_len = (w as usize) * (h as usize);
        let uv_len = ((w / 2) as usize) * ((h / 2) as usize) * 2;
        Self {
            width: w,
            height: h,
            y_plane: vec![16; y_len],
            uv_plane: vec![128; uv_len],
        }
    }

    /// Packs tightly packed straight RGBA8 into limited-range BT.709 NV12.
    ///
    /// Width and height must both be even and at least 2. There is no row stride.
    /// Source alpha is ignored. Each pixel keeps its own luma sample.
    /// Each 2x2 block stores the average of its four Cb samples and the average of its four Cr samples,
    /// rounded with `(sum + 2) / 4`.
    /// Returns `None` when the size is unusable or `rgba` is shorter than `width * height * 4`.
    pub fn from_rgba_bt709(width: u32, height: u32, rgba: &[u8]) -> Option<Self> {
        if width < 2 || height < 2 || (width % 2 != 0) || (height % 2 != 0) {
            return None;
        }
        let expected = (width as usize)
            .checked_mul(height as usize)?
            .checked_mul(4)?;
        if rgba.len() < expected {
            return None;
        }
        let mut surface = Self::new(width, height);
        let w = width as usize;
        let h = height as usize;
        let uv_w = w / 2;

        // 4:2:0. Chroma is one pair for the block, not a filtered sample between blocks.
        for by in (0..h).step_by(2) {
            for bx in (0..w).step_by(2) {
                let mut u_acc = 0u32;
                let mut v_acc = 0u32;
                for dy in 0..2 {
                    for dx in 0..2 {
                        let px = bx + dx;
                        let py = by + dy;
                        let idx = (py * w + px) * 4;
                        let (y, u, v) = bt709_rgba_to_yuv(rgba[idx], rgba[idx + 1], rgba[idx + 2]);
                        surface.y_plane[py * w + px] = y;
                        u_acc += u32::from(u);
                        v_acc += u32::from(v);
                    }
                }
                let uv_idx = ((by / 2) * uv_w + (bx / 2)) * 2;
                surface.uv_plane[uv_idx] = ((u_acc + 2) / 4) as u8;
                surface.uv_plane[uv_idx + 1] = ((v_acc + 2) / 4) as u8;
            }
        }
        Some(surface)
    }

    /// Expands this surface to tightly packed straight RGBA8.
    ///
    /// Each pixel reads its own luma byte and the CbCr pair of its 2x2 block.
    /// Chroma is not filtered across blocks. Alpha is always 255.
    /// The matrix is the limited-range inverse in [`bt709_yuv_to_rgba`].
    pub fn to_rgba_bt709(&self) -> Vec<u8> {
        let w = self.width as usize;
        let h = self.height as usize;
        let uv_w = w / 2;
        let mut rgba = vec![0u8; w * h * 4];
        for py in 0..h {
            for px in 0..w {
                let y = self.y_plane[py * w + px];
                // Point-sampled chroma: the whole 2x2 block shares one CbCr pair.
                let uv_idx = ((py / 2) * uv_w + (px / 2)) * 2;
                let u = self.uv_plane[uv_idx];
                let v = self.uv_plane[uv_idx + 1];
                let c = bt709_yuv_to_rgba(y, u, v);
                let dst = (py * w + px) * 4;
                rgba[dst] = c.r;
                rgba[dst + 1] = c.g;
                rgba[dst + 2] = c.b;
                rgba[dst + 3] = 255;
            }
        }
        rgba
    }
}

/// Integer texel rectangle inside a [`TextureAtlas`], origin at the atlas top-left.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AtlasRegion {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

/// Shelf-packed straight RGBA8 atlas.
///
/// Rows fill left to right. One texel is left empty between allocations and between rows.
/// Texels are copied from the caller and are not premultiplied.
/// This is CPU memory, not a GPU texture.
#[derive(Clone, Debug)]
pub struct TextureAtlas {
    pub width: u32,
    pub height: u32,
    /// Straight RGBA8, row-major, four bytes per texel.
    pub pixels: Vec<u8>,
    // Next free texel on the current shelf.
    cursor_x: u32,
    // Top of the current shelf, in texels. +y is downward.
    cursor_y: u32,
    // Tallest allocation placed on the current shelf.
    row_height: u32,
}

impl TextureAtlas {
    /// Allocates a zero-filled atlas. The shelf cursor starts at the top-left.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            pixels: vec![0; (width as usize) * (height as usize) * 4],
            cursor_x: 0,
            cursor_y: 0,
            row_height: 0,
        }
    }

    /// Shelf-packs `rgba` and copies it into the atlas.
    ///
    /// `rgba` must be tightly packed straight RGBA8, four bytes per texel, with no stride.
    /// A zero size, a short slice, or a rect larger than the atlas returns `None`.
    /// If the rect does not fit on the current row, the cursor moves to the next shelf
    /// (`row_height + 1` texels down). There is no search for holes.
    /// A one-texel gutter is reserved after the rect. That gutter may sit past the far edge,
    /// which only forces the following allocation onto a new shelf.
    pub fn allocate(&mut self, width: u32, height: u32, rgba: &[u8]) -> Option<AtlasRegion> {
        if width == 0 || height == 0 || rgba.len() < (width as usize) * (height as usize) * 4 {
            return None;
        }
        if width > self.width || height > self.height {
            return None;
        }
        // Equality still fits. The gutter is applied after the copy.
        if self.cursor_x + width > self.width {
            self.cursor_x = 0;
            self.cursor_y += self.row_height + 1;
            self.row_height = 0;
        }
        if self.cursor_y + height > self.height {
            return None;
        }

        let region = AtlasRegion {
            x: self.cursor_x,
            y: self.cursor_y,
            width,
            height,
        };
        // Straight copy into the shelf. No resample and no premultiply.
        for row in 0..height {
            let src_start = (row * width * 4) as usize;
            let dst_start = (((region.y + row) * self.width + region.x) * 4) as usize;
            self.pixels[dst_start..dst_start + (width as usize) * 4]
                .copy_from_slice(&rgba[src_start..src_start + (width as usize) * 4]);
        }

        self.cursor_x += width + 1;
        self.row_height = self.row_height.max(height);
        Some(region)
    }

    /// Nearest-texel sample of the atlas.
    ///
    /// `u` and `v` are clamped to 0..=1 and mapped with `round(t * (size - 1))`,
    /// so 0 and 1 land on the first and last texels, not on texel edges.
    /// An empty atlas returns opaque white. The stored channels are returned straight.
    pub fn sample_uv(&self, u: f32, v: f32) -> GpuColor {
        if self.width == 0 || self.height == 0 {
            return GpuColor::rgba(255, 255, 255, 255);
        }
        let px =
            ((u.clamp(0.0, 1.0) * (self.width as f32 - 1.0)).round() as u32).min(self.width - 1);
        let py =
            ((v.clamp(0.0, 1.0) * (self.height as f32 - 1.0)).round() as u32).min(self.height - 1);
        let idx = ((py * self.width + px) * 4) as usize;
        GpuColor::rgba(
            self.pixels[idx],
            self.pixels[idx + 1],
            self.pixels[idx + 2],
            self.pixels[idx + 3],
        )
    }

    /// Edge-normalized window of `region`: `[x/w, y/h, (x+width)/w, (y+height)/h]`.
    ///
    /// These are fractions of the atlas edges, not texel centers.
    /// A zero atlas size is treated as 1 so the divisions stay defined.
    pub fn uv_bounds(&self, region: AtlasRegion) -> [f32; 4] {
        let w = self.width.max(1) as f32;
        let h = self.height.max(1) as f32;
        [
            (region.x as f32) / w,
            (region.y as f32) / h,
            ((region.x + region.width) as f32) / w,
            ((region.y + region.height) as f32) / h,
        ]
    }
}

/// How a [`GpuQuad`] fills its rectangle.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum QuadKind {
    /// Flat [`GpuQuad::fill_color`].
    Solid,
    /// Two-stop gradient. [`GpuQuad::fill_color`] is the start color.
    /// `angle_deg` uses the same direction as [`sample_linear_gradient`].
    LinearGradient { end_color: GpuColor, angle_deg: f32 },
    /// Soft shadow. `blur_radius` is passed to [`sdf_box_shadow_alpha`].
    BoxShadow { blur_radius: f32 },
}

/// One quad rasterized by [`render_frame_rgba`] on the CPU.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuQuad {
    /// Fill bounds. The signed distance is measured from the center of this rectangle.
    pub rect: GpuRect,
    /// Clip rectangle in the same pixel space. Rasterization does not draw outside it.
    pub clip: GpuRect,
    /// Straight fill, gradient start, or shadow color.
    pub fill_color: GpuColor,
    /// Straight stroke color. Ignored while `border_width` is 0.
    pub border_color: GpuColor,
    /// Corner radius in pixels, before the SDF clamps it to the half-extents.
    pub corner_radius: f32,
    /// Stroke thickness in pixels, inset from the outer edge.
    pub border_width: f32,
    /// Optional atlas window `[u0, v0, u1, v1]` in the edge space of [`TextureAtlas::uv_bounds`].
    pub uv_rect: Option<[f32; 4]>,
    /// Solid fill, linear gradient, or box shadow.
    pub kind: QuadKind,
}

impl GpuQuad {
    /// Solid straight fill. The clip matches `rect`. Radius, stroke, and UV are clear.
    pub fn solid(rect: GpuRect, fill_color: GpuColor) -> Self {
        Self {
            rect,
            clip: rect,
            fill_color,
            border_color: GpuColor::rgba(0, 0, 0, 0),
            corner_radius: 0.0,
            border_width: 0.0,
            uv_rect: None,
            kind: QuadKind::Solid,
        }
    }

    /// Two-stop gradient across `rect`. `start_color` is stored as [`GpuQuad::fill_color`].
    ///
    /// `angle_deg` is documented on [`sample_linear_gradient`]. Corners stay square until
    /// [`GpuQuad::with_rounded_border`] is applied. The clip matches `rect`.
    pub fn linear_gradient(
        rect: GpuRect,
        start_color: GpuColor,
        end_color: GpuColor,
        angle_deg: f32,
    ) -> Self {
        Self {
            rect,
            clip: rect,
            fill_color: start_color,
            border_color: GpuColor::rgba(0, 0, 0, 0),
            corner_radius: 0.0,
            border_width: 0.0,
            uv_rect: None,
            kind: QuadKind::LinearGradient {
                end_color,
                angle_deg,
            },
        }
    }

    /// Shadow quad for a box, expanded by spread and offset, with blur stored on [`QuadKind`].
    ///
    /// `rect` is `box_rect` shifted by `(offset_x, offset_y)` and outset by `spread` on every side.
    /// Width and height are at least 1. `blur_radius` is clamped to >= 0 and the clip is `rect`
    /// outset by that blur. The corner radius is `(corner_radius + spread)` clamped to >= 0.
    /// Negative spread is left as a negative outset. The stroke is clear.
    pub fn box_shadow(
        box_rect: GpuRect,
        offset_x: f32,
        offset_y: f32,
        blur_radius: f32,
        spread_radius: f32,
        corner_radius: f32,
        color: GpuColor,
    ) -> Self {
        let blur = blur_radius.max(0.0);
        let spread = spread_radius;
        let rect = GpuRect::new(
            box_rect.x + offset_x - spread,
            box_rect.y + offset_y - spread,
            (box_rect.width + spread * 2.0).max(1.0),
            (box_rect.height + spread * 2.0).max(1.0),
        );
        let clip = GpuRect::new(
            rect.x - blur,
            rect.y - blur,
            rect.width + blur * 2.0,
            rect.height + blur * 2.0,
        );
        Self {
            rect,
            clip,
            fill_color: color,
            border_color: GpuColor::rgba(0, 0, 0, 0),
            corner_radius: (corner_radius + spread).max(0.0),
            border_width: 0.0,
            uv_rect: None,
            kind: QuadKind::BoxShadow { blur_radius: blur },
        }
    }

    /// Sets a rounded stroke. Radius and width are clamped to >= 0.
    pub fn with_rounded_border(
        mut self,
        corner_radius: f32,
        border_width: f32,
        border_color: GpuColor,
    ) -> Self {
        self.corner_radius = corner_radius.max(0.0);
        self.border_width = border_width.max(0.0);
        self.border_color = border_color;
        self
    }

    /// Replaces the clip. The new clip is not intersected with the previous one.
    pub fn with_clip(mut self, clip: GpuRect) -> Self {
        self.clip = clip;
        self
    }

    /// Attaches an atlas window. Sampling happens later, in [`render_frame_rgba`].
    pub fn with_uv(mut self, uv_rect: [f32; 4]) -> Self {
        self.uv_rect = Some(uv_rect);
        self
    }
}

/// Signed distance from a point to a rounded rectangle.
///
/// `rel_x` and `rel_y` are relative to the box center. `half_w` and `half_h` are the half-extents.
/// `radius` is clamped to `0..=min(half_w, half_h)`.
/// The result is negative inside, zero on the boundary, and positive outside.
pub fn sdf_rounded_box(rel_x: f32, rel_y: f32, half_w: f32, half_h: f32, radius: f32) -> f32 {
    let r = radius.min(half_w).min(half_h).max(0.0);
    // Box extent after the corner circles are removed, then a quarter-circle in the outside quadrant.
    let qx = rel_x.abs() - (half_w - r);
    let qy = rel_y.abs() - (half_h - r);
    let outside = (qx.max(0.0).powi(2) + qy.max(0.0).powi(2)).sqrt();
    let inside = qx.max(qy).min(0.0);
    outside + inside - r
}

/// Smoothstep coverage for a shadow at signed distance `dist`.
///
/// `sigma` is `max(blur_radius, 0.5)`. The input `t` is `clamp(0.5 - dist / sigma, 0, 1)`,
/// then the Hermite step `t*t*(3-2*t)`. Negative `dist` (inside) approaches 1.
/// Coverage reaches 0 once `dist` is at least `0.5 * sigma`.
pub fn sdf_box_shadow_alpha(dist: f32, blur_radius: f32) -> f32 {
    let sigma = blur_radius.max(0.5);
    let t = (0.5 - dist / sigma).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Two-stop straight-color gradient in box-center space.
///
/// The direction is `(sin(angle), -cos(angle))` after `angle_deg` is converted to radians.
/// Zero degrees points toward decreasing y, which is up when y grows downward.
/// Ninety degrees points toward increasing x.
/// The projection is divided by the box extent along that direction, with a floor of 0.5,
/// so one side of the box is the start color and the opposite side is the end color.
pub fn sample_linear_gradient(
    rel_x: f32,
    rel_y: f32,
    half_w: f32,
    half_h: f32,
    angle_deg: f32,
    start: GpuColor,
    end: GpuColor,
) -> GpuColor {
    let rad = angle_deg.to_radians();
    let dx = rad.sin();
    let dy = -rad.cos();
    let max_extent = (dx.abs() * half_w + dy.abs() * half_h).max(0.5);
    let proj = rel_x * dx + rel_y * dy;
    let t = ((proj / max_extent) * 0.5 + 0.5).clamp(0.0, 1.0);
    lerp_color(start, end, t)
}

/// One color stop on a multi-stop linear or radial gradient ramp (`offset` in `0.0..=1.0`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuGradientStop {
    pub offset: f32,
    pub color: GpuColor,
}

impl GpuGradientStop {
    /// Construct a gradient stop at `offset` (`0.0..=1.0`).
    pub const fn new(offset: f32, color: GpuColor) -> Self {
        Self { offset, color }
    }
}

/// Sample a multi-stop color ramp at normalized parameter `t` (`0.0..=1.0`).
pub fn sample_multi_stop_gradient(stops: &[GpuGradientStop], t: f32) -> GpuColor {
    if stops.is_empty() {
        return GpuColor::rgba(0, 0, 0, 0);
    }
    if stops.len() == 1 {
        return stops[0].color;
    }
    let clamped = t.clamp(0.0, 1.0);
    if clamped <= stops[0].offset {
        return stops[0].color;
    }
    for pair in stops.windows(2) {
        let a = pair[0];
        let b = pair[1];
        if clamped <= b.offset {
            let span = (b.offset - a.offset).max(1e-5);
            let local_t = ((clamped - a.offset) / span).clamp(0.0, 1.0);
            return lerp_color(a.color, b.color, local_t);
        }
    }
    stops[stops.len() - 1].color
}

/// Sample an elliptical radial gradient centered at `(cx, cy)` with radii `(rx, ry)` at point `(px, py)`.
pub fn sample_radial_gradient(
    cx: f32,
    cy: f32,
    rx: f32,
    ry: f32,
    px: f32,
    py: f32,
    stops: &[GpuGradientStop],
) -> GpuColor {
    let nx = (px - cx) / rx.max(0.5);
    let ny = (py - cy) / ry.max(0.5);
    let dist = (nx * nx + ny * ny).sqrt().clamp(0.0, 1.0);
    sample_multi_stop_gradient(stops, dist)
}

/// Status returned by [`probe_and_compile_d3d11_compositor`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct D3d11HardwareStatus {
    /// True when `D3D11CreateDevice` succeeded on Windows.
    pub device_created: bool,
    /// Negotiated D3D feature level (for example `0xb000` for `11_0`), or `0` on software fallback.
    pub feature_level: u32,
    /// Byte length of compiled `vs_main` DXBC bytecode.
    pub vs_bytecode_len: usize,
    /// Byte length of compiled `ps_main` DXBC bytecode.
    pub ps_bytecode_len: usize,
    /// Short backend label.
    pub backend_label: &'static str,
}

/// Compile an HLSL Shader Model 5.0 entry point (`vs_5_0` or `ps_5_0`) into `DXBC` bytecode
/// using `d3dcompiler_47.dll` on Windows.
#[cfg(target_os = "windows")]
pub fn compile_hlsl_entry(
    hlsl_source: &str,
    entry_point: &str,
    profile: &str,
) -> Result<Vec<u8>, String> {
    use std::ffi::{CString, c_void};
    use std::ptr::{null, null_mut};

    #[repr(C)]
    struct ID3DBlobVtbl {
        query_interface: usize,
        add_ref: usize,
        release: unsafe extern "system" fn(*mut ID3DBlob) -> u32,
        get_buffer_pointer: unsafe extern "system" fn(*mut ID3DBlob) -> *const u8,
        get_buffer_size: unsafe extern "system" fn(*mut ID3DBlob) -> usize,
    }

    #[repr(C)]
    struct ID3DBlob {
        vtbl: *const ID3DBlobVtbl,
    }

    type D3DCompileFn = unsafe extern "system" fn(
        src_data: *const u8,
        src_data_size: usize,
        source_name: *const i8,
        defines: *const c_void,
        include: *const c_void,
        entry_point: *const i8,
        target: *const i8,
        flags1: u32,
        flags2: u32,
        code: *mut *mut ID3DBlob,
        error_msgs: *mut *mut ID3DBlob,
    ) -> i32;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn LoadLibraryA(name: *const i8) -> *mut c_void;
        fn GetProcAddress(module: *mut c_void, name: *const i8) -> *mut c_void;
    }

    let c_entry = CString::new(entry_point).map_err(|_| "invalid entry point".to_string())?;
    let c_profile = CString::new(profile).map_err(|_| "invalid profile".to_string())?;
    unsafe {
        let dll = LoadLibraryA(c"d3dcompiler_47.dll".as_ptr());
        if dll.is_null() {
            return Err("d3dcompiler_47.dll not available".to_string());
        }
        let proc = GetProcAddress(dll, c"D3DCompile".as_ptr());
        if proc.is_null() {
            return Err("D3DCompile symbol not found".to_string());
        }
        let d3d_compile: D3DCompileFn = std::mem::transmute(proc);
        let mut code_blob: *mut ID3DBlob = null_mut();
        let mut err_blob: *mut ID3DBlob = null_mut();
        let hr = d3d_compile(
            hlsl_source.as_ptr(),
            hlsl_source.len(),
            c"compositor.hlsl".as_ptr(),
            null(),
            null(),
            c_entry.as_ptr(),
            c_profile.as_ptr(),
            (1 << 11) | (1 << 15),
            0,
            &mut code_blob,
            &mut err_blob,
        );
        if hr < 0 || code_blob.is_null() {
            let mut msg = format!("D3DCompile failed (HRESULT 0x{hr:08x})");
            if !err_blob.is_null() {
                let ptr = ((*(*err_blob).vtbl).get_buffer_pointer)(err_blob);
                let len = ((*(*err_blob).vtbl).get_buffer_size)(err_blob);
                if !ptr.is_null() && len > 0 {
                    let slice = std::slice::from_raw_parts(ptr, len);
                    msg = String::from_utf8_lossy(slice).trim().to_string();
                }
                ((*(*err_blob).vtbl).release)(err_blob);
            }
            return Err(msg);
        }
        if !err_blob.is_null() {
            ((*(*err_blob).vtbl).release)(err_blob);
        }
        let ptr = ((*(*code_blob).vtbl).get_buffer_pointer)(code_blob);
        let len = ((*(*code_blob).vtbl).get_buffer_size)(code_blob);
        let bytes = if !ptr.is_null() && len > 0 {
            std::slice::from_raw_parts(ptr, len).to_vec()
        } else {
            Vec::new()
        };
        ((*(*code_blob).vtbl).release)(code_blob);
        if bytes.len() < 4 || &bytes[0..4] != b"DXBC" {
            return Err("compiled blob missing DXBC signature".to_string());
        }
        Ok(bytes)
    }
}

/// Non-Windows stub for [`compile_hlsl_entry`].
#[cfg(not(target_os = "windows"))]
pub fn compile_hlsl_entry(
    _hlsl_source: &str,
    _entry_point: &str,
    _profile: &str,
) -> Result<Vec<u8>, String> {
    Err("Direct3D HLSL compilation is only supported on Windows".to_string())
}

/// Probe Direct3D 11 device creation (`d3d11.dll`) and compile [`COMPOSITOR_HLSL`]
/// (`vs_main` and `ps_main`) into `DXBC` bytecode on Windows.
pub fn probe_and_compile_d3d11_compositor() -> D3d11HardwareStatus {
    #[cfg(target_os = "windows")]
    {
        use std::ffi::c_void;
        use std::ptr::null_mut;

        #[repr(C)]
        struct IUnknownVtbl {
            query_interface: usize,
            add_ref: usize,
            release: unsafe extern "system" fn(*mut IUnknown) -> u32,
        }

        #[repr(C)]
        struct IUnknown {
            vtbl: *const IUnknownVtbl,
        }

        type D3D11CreateDeviceFn = unsafe extern "system" fn(
            adapter: *mut c_void,
            driver_type: u32,
            software: *mut c_void,
            flags: u32,
            feature_levels: *const u32,
            num_levels: u32,
            sdk_version: u32,
            device: *mut *mut IUnknown,
            out_level: *mut u32,
            context: *mut *mut IUnknown,
        ) -> i32;

        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn LoadLibraryA(name: *const i8) -> *mut c_void;
            fn GetProcAddress(module: *mut c_void, name: *const i8) -> *mut c_void;
        }

        let vs_len = compile_hlsl_entry(COMPOSITOR_HLSL, "VSMain", "vs_5_0")
            .map(|b| b.len())
            .unwrap_or(0);
        let ps_len = compile_hlsl_entry(COMPOSITOR_HLSL, "PSMain", "ps_5_0")
            .map(|b| b.len())
            .unwrap_or(0);

        unsafe {
            let dll = LoadLibraryA(c"d3d11.dll".as_ptr());
            if !dll.is_null() {
                let proc = GetProcAddress(dll, c"D3D11CreateDevice".as_ptr());
                if !proc.is_null() {
                    let create_dev: D3D11CreateDeviceFn = std::mem::transmute(proc);
                    let levels = [0xb000u32];
                    for &(driver_type, label) in &[
                        (1u32, "Direct3D 11 Hardware (SM 5.0 DXBC)"),
                        (5u32, "Direct3D 11 WARP (SM 5.0 DXBC)"),
                    ] {
                        let mut dev: *mut IUnknown = null_mut();
                        let mut ctx: *mut IUnknown = null_mut();
                        let mut level = 0u32;
                        let hr = create_dev(
                            null_mut(),
                            driver_type,
                            null_mut(),
                            0,
                            levels.as_ptr(),
                            1,
                            7,
                            &mut dev,
                            &mut level,
                            &mut ctx,
                        );
                        if hr >= 0 && !dev.is_null() {
                            if !ctx.is_null() {
                                ((*(*ctx).vtbl).release)(ctx);
                            }
                            ((*(*dev).vtbl).release)(dev);
                            return D3d11HardwareStatus {
                                device_created: true,
                                feature_level: level,
                                vs_bytecode_len: vs_len,
                                ps_bytecode_len: ps_len,
                                backend_label: label,
                            };
                        }
                    }
                }
            }
        }
        D3d11HardwareStatus {
            device_created: false,
            feature_level: 0,
            vs_bytecode_len: vs_len,
            ps_bytecode_len: ps_len,
            backend_label: "CPU Tile Compositor Fallback",
        }
    }
    #[cfg(not(target_os = "windows"))]
    {
        D3d11HardwareStatus {
            device_created: false,
            feature_level: 0,
            vs_bytecode_len: 0,
            ps_bytecode_len: 0,
            backend_label: "CPU Tile Compositor Fallback",
        }
    }
}

/// Rasterizes `quads` in painter's order into a straight RGBA8 buffer.
///
/// The buffer is `width` by `height`, row-major, four bytes per pixel, cleared to `clear` (alpha included).
/// +x is right and +y is down. Each visited pixel is sampled at its center (`index + 0.5`).
/// The visit range is the intersection of `clip` with `rect` expanded by the shadow blur,
/// then `floor` on the minimum and `ceil` on the maximum, clamped to the framebuffer.
/// Solid and gradient coverage is a one-pixel ramp of the rounded-box distance.
/// Shadows use [`sdf_box_shadow_alpha`] instead of that ramp.
/// An atlas sample is applied only when both `uv_rect` and `atlas` are present, as a straight multiply.
/// Later quads composite over earlier ones through [`blend_pixel`]. RGB stays straight, not premultiplied.
pub fn render_frame_rgba(
    width: u32,
    height: u32,
    clear: GpuColor,
    quads: &[GpuQuad],
    atlas: Option<&TextureAtlas>,
) -> Vec<u8> {
    let mut fb = vec![0u8; (width as usize) * (height as usize) * 4];
    for px in fb.chunks_exact_mut(4) {
        px[0] = clear.r;
        px[1] = clear.g;
        px[2] = clear.b;
        px[3] = clear.a;
    }

    for quad in quads {
        let pad = match quad.kind {
            QuadKind::BoxShadow { blur_radius } => blur_radius.max(0.0),
            _ => 0.0,
        };
        // Pixel indices. The far edge is exclusive after the `ceil` cast into the `..` range.
        let min_x = (quad.rect.x - pad).max(quad.clip.x).floor().max(0.0) as u32;
        let min_y = (quad.rect.y - pad).max(quad.clip.y).floor().max(0.0) as u32;
        let max_x = (quad.rect.x + quad.rect.width + pad)
            .min(quad.clip.x + quad.clip.width)
            .ceil()
            .min(width as f32) as u32;
        let max_y = (quad.rect.y + quad.rect.height + pad)
            .min(quad.clip.y + quad.clip.height)
            .ceil()
            .min(height as f32) as u32;

        let half_w = quad.rect.width * 0.5;
        let half_h = quad.rect.height * 0.5;
        let cx = quad.rect.x + half_w;
        let cy = quad.rect.y + half_h;

        for py in min_y..max_y {
            for px in min_x..max_x {
                // Pixel center. `px`/`py` are integer indices; y grows downward.
                let sample_x = px as f32 + 0.5;
                let sample_y = py as f32 + 0.5;
                let rel_x = sample_x - cx;
                let rel_y = sample_y - cy;
                let dist = sdf_rounded_box(rel_x, rel_y, half_w, half_h, quad.corner_radius);

                if let QuadKind::BoxShadow { blur_radius } = quad.kind {
                    let shadow_alpha = sdf_box_shadow_alpha(dist, blur_radius);
                    if shadow_alpha <= 0.0 {
                        continue;
                    }
                    let mut color = quad.fill_color;
                    color.a = ((color.a as f32) * shadow_alpha).round() as u8;
                    let dst_idx = ((py * width + px) * 4) as usize;
                    blend_pixel(&mut fb[dst_idx..dst_idx + 4], color);
                    continue;
                }

                // One-pixel coverage ramp around the zero isosurface. Not a premultiplied store.
                let outer_alpha = (0.5 - dist).clamp(0.0, 1.0);
                if outer_alpha <= 0.0 {
                    continue;
                }

                let mut color = match quad.kind {
                    QuadKind::LinearGradient {
                        end_color,
                        angle_deg,
                    } => sample_linear_gradient(
                        rel_x,
                        rel_y,
                        half_w,
                        half_h,
                        angle_deg,
                        quad.fill_color,
                        end_color,
                    ),
                    _ => quad.fill_color,
                };
                if let (Some(uv), Some(tex)) = (quad.uv_rect, atlas) {
                    // Map the pixel center across `rect` into the edge-normalized UV window.
                    let tx = ((sample_x - quad.rect.x) / quad.rect.width.max(1.0)).clamp(0.0, 1.0);
                    let ty = ((sample_y - quad.rect.y) / quad.rect.height.max(1.0)).clamp(0.0, 1.0);
                    let u = uv[0] + (uv[2] - uv[0]) * tx;
                    let v = uv[1] + (uv[3] - uv[1]) * ty;
                    let sampled = tex.sample_uv(u, v);
                    color = modulate_color(color, sampled);
                }

                if quad.border_width > 0.0 {
                    // `inner_dist` is zero one stroke-width inside the outer edge.
                    // The mix is a one-pixel ramp around that inner edge.
                    let inner_dist = dist + quad.border_width;
                    let border_mix = (0.5 + inner_dist).clamp(0.0, 1.0);
                    color = lerp_color(color, quad.border_color, border_mix);
                }

                color.a = ((color.a as f32) * outer_alpha).round() as u8;
                let dst_idx = ((py * width + px) * 4) as usize;
                blend_pixel(&mut fb[dst_idx..dst_idx + 4], color);
            }
        }
    }

    fb
}

/// Straight channel multiply with rounding `(a * b + 127) / 255`.
///
/// Used to tint an atlas sample by the quad color. Not a premultiplied composite.
fn modulate_color(a: GpuColor, b: GpuColor) -> GpuColor {
    GpuColor {
        r: ((u16::from(a.r) * u16::from(b.r) + 127) / 255) as u8,
        g: ((u16::from(a.g) * u16::from(b.g) + 127) / 255) as u8,
        b: ((u16::from(a.b) * u16::from(b.b) + 127) / 255) as u8,
        a: ((u16::from(a.a) * u16::from(b.a) + 127) / 255) as u8,
    }
}

/// Straight per-channel lerp. `t` is not clamped here; gradient callers already clamp it.
fn lerp_color(a: GpuColor, b: GpuColor, t: f32) -> GpuColor {
    let inv = 1.0 - t;
    GpuColor {
        r: ((a.r as f32) * inv + (b.r as f32) * t).round() as u8,
        g: ((a.g as f32) * inv + (b.g as f32) * t).round() as u8,
        b: ((a.b as f32) * inv + (b.b as f32) * t).round() as u8,
        a: ((a.a as f32) * inv + (b.a as f32) * t).round() as u8,
    }
}

/// Straight source-over onto one framebuffer pixel.
///
/// Source RGB is not premultiplied: it is weighted by `src.a` here.
/// Alpha 255 replaces RGB and writes alpha 255. Alpha 0 leaves the destination unchanged.
/// Any other alpha blends RGB with rounding `(sum + 127) / 255` and forces destination alpha to 255.
fn blend_pixel(dst: &mut [u8], src: GpuColor) {
    if src.a == 255 {
        dst[0] = src.r;
        dst[1] = src.g;
        dst[2] = src.b;
        dst[3] = 255;
        return;
    }
    if src.a == 0 {
        return;
    }
    let alpha = u16::from(src.a);
    let inv = 255 - alpha;
    dst[0] = ((u16::from(src.r) * alpha + u16::from(dst[0]) * inv + 127) / 255) as u8;
    dst[1] = ((u16::from(src.g) * alpha + u16::from(dst[1]) * inv + 127) / 255) as u8;
    dst[2] = ((u16::from(src.b) * alpha + u16::from(dst[2]) * inv + 127) / 255) as u8;
    dst[3] = 255;
}

/// Checks shader source markers, atlas shelf packing, NV12 conversion, rounded quads, gradients, and shadows.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embeds_hlsl_and_glsl_shaders_and_packs_atlas_regions() {
        assert!(COMPOSITOR_HLSL.contains("SdfRoundedBox"));
        assert!(COMPOSITOR_HLSL.contains("ConvertNv12Bt709ToRgba"));
        assert!(COMPOSITOR_GLSL.contains("SdfRoundedBox"));
        assert!(COMPOSITOR_GLSL.contains("ConvertNv12Bt709ToRgba"));
        assert!(COMPOSITOR_GLSL.contains("#version 330 core"));
        let mut atlas = TextureAtlas::new(32, 32);
        let red_2x2 = vec![255, 0, 0, 255].repeat(4);
        let reg = atlas.allocate(2, 2, &red_2x2).unwrap();
        assert_eq!(reg.x, 0);
        assert_eq!(reg.y, 0);
        let uvs = atlas.uv_bounds(reg);
        let sampled = atlas.sample_uv(uvs[0], uvs[1]);
        assert_eq!(sampled, GpuColor::rgba(255, 0, 0, 255));

        let nv12 = Nv12Surface::from_rgba_bt709(2, 2, &red_2x2).unwrap();
        let decoded = nv12.to_rgba_bt709();
        assert!(decoded[0] > 230 && decoded[1] < 25 && decoded[2] < 25);
    }

    #[test]
    fn renders_rounded_quad_with_clipped_corners_and_center_fill() {
        let quad = GpuQuad::solid(
            GpuRect::new(0.0, 0.0, 20.0, 20.0),
            GpuColor::rgba(20, 120, 220, 255),
        )
        .with_rounded_border(6.0, 2.0, GpuColor::rgba(255, 200, 50, 255));

        let fb = render_frame_rgba(20, 20, GpuColor::rgba(0, 0, 0, 255), &[quad], None);
        // Top-left corner pixel (0, 0) lies outside the 6px radius curve and stays black
        assert_eq!(&fb[0..4], &[0, 0, 0, 255]);
        // Center pixel (10, 10) has the fill color
        let center_idx = ((10 * 20 + 10) * 4) as usize;
        assert_eq!(&fb[center_idx..center_idx + 4], &[20, 120, 220, 255]);
    }

    #[test]
    fn renders_linear_gradient_and_soft_box_shadow_quads() {
        let shadow = GpuQuad::box_shadow(
            GpuRect::new(4.0, 4.0, 12.0, 12.0),
            0.0,
            2.0,
            4.0,
            0.0,
            3.0,
            GpuColor::rgba(0, 0, 0, 180),
        );
        let grad = GpuQuad::linear_gradient(
            GpuRect::new(0.0, 0.0, 20.0, 10.0),
            GpuColor::rgba(255, 0, 0, 255),
            GpuColor::rgba(0, 0, 255, 255),
            90.0,
        );
        let fb = render_frame_rgba(
            24,
            24,
            GpuColor::rgba(255, 255, 255, 255),
            &[shadow, grad],
            None,
        );
        let left_idx = ((5 * 24 + 1) * 4) as usize;
        let right_idx = ((5 * 24 + 18) * 4) as usize;
        assert!(fb[left_idx] > fb[right_idx]);
        assert!(fb[right_idx + 2] > fb[left_idx + 2]);

        // Below the gradient at (10, 17), the shadow darkens the white background
        let shadow_idx = ((17 * 24 + 10) * 4) as usize;
        assert!(fb[shadow_idx] < 255);
    }

    #[test]
    fn samples_multi_stop_and_radial_gradients_and_compiles_d3d11_hlsl() {
        let stops = [
            GpuGradientStop::new(0.0, GpuColor::rgba(255, 0, 0, 255)),
            GpuGradientStop::new(0.5, GpuColor::rgba(0, 255, 0, 255)),
            GpuGradientStop::new(1.0, GpuColor::rgba(0, 0, 255, 255)),
        ];
        assert_eq!(
            sample_multi_stop_gradient(&stops, 0.0),
            GpuColor::rgba(255, 0, 0, 255)
        );
        assert_eq!(
            sample_multi_stop_gradient(&stops, 0.5),
            GpuColor::rgba(0, 255, 0, 255)
        );
        assert_eq!(
            sample_radial_gradient(10.0, 10.0, 10.0, 10.0, 10.0, 10.0, &stops),
            GpuColor::rgba(255, 0, 0, 255)
        );
        assert_eq!(
            sample_radial_gradient(10.0, 10.0, 10.0, 10.0, 20.0, 10.0, &stops),
            GpuColor::rgba(0, 0, 255, 255)
        );

        let status = probe_and_compile_d3d11_compositor();
        #[cfg(target_os = "windows")]
        {
            assert!(status.device_created);
            assert_eq!(status.feature_level, 0xb000);
            assert!(status.vs_bytecode_len > 64);
            assert!(status.ps_bytecode_len > 64);
        }
        #[cfg(not(target_os = "windows"))]
        assert!(!status.device_created);
    }
}
