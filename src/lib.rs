//! 2D signed-distance-field quad compositor, shelf-packing RGBA texture atlas, and HLSL/GLSL SDF shader pipeline (Proof of Concept).
//!
//! Standalone `MPL-2.0` companion library for the Flux Browser Proof of Concept ecosystem.

/// Embedded HLSL Shader Model 5.0 source for the Direct3D compositor pipeline (Windows).
pub const COMPOSITOR_HLSL: &str = include_str!("../shaders/compositor.hlsl");

/// Embedded GLSL 330 Core source for the OpenGL/Vulkan compositor pipeline (Linux & cross-platform).
pub const COMPOSITOR_GLSL: &str = include_str!("../shaders/compositor.glsl");

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuColor {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl GpuColor {
    pub const fn rgba(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuRect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl GpuRect {
    pub const fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    pub fn contains(self, px: f32, py: f32) -> bool {
        px >= self.x && py >= self.y && px < self.x + self.width && py < self.y + self.height
    }
}

/// Convert an 8-bit `RGBA` color into BT.709 `YUV` (`(y, u, v)`).
pub fn bt709_rgba_to_yuv(r: u8, g: u8, b: u8) -> (u8, u8, u8) {
    let rf = r as f32;
    let gf = g as f32;
    let bf = b as f32;
    let y = (16.0 + 0.182586 * rf + 0.614231 * gf + 0.062007 * bf)
        .round()
        .clamp(16.0, 235.0) as u8;
    let u = (128.0 - 0.100644 * rf - 0.338572 * gf + 0.439216 * bf)
        .round()
        .clamp(16.0, 240.0) as u8;
    let v = (128.0 + 0.439216 * rf - 0.398942 * gf - 0.040274 * bf)
        .round()
        .clamp(16.0, 240.0) as u8;
    (y, u, v)
}

/// Convert an 8-bit BT.709 `(y, u, v)` sample into `GpuColor` (`RGBA8888`).
pub fn bt709_yuv_to_rgba(y: u8, u: u8, v: u8) -> GpuColor {
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

/// Semi-planar 4:2:0 `NV12` GPU video surface (`Y` luma plane + interleaved `UV` chroma plane).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Nv12Surface {
    pub width: u32,
    pub height: u32,
    pub y_plane: Vec<u8>,
    pub uv_plane: Vec<u8>,
}

impl Nv12Surface {
    /// Allocate a zeroed (black) `NV12` surface for `(width, height)`.
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

    /// Convert an `RGBA8888` buffer into a semi-planar BT.709 `NV12` surface for GPU video encoding.
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

    /// Convert this `NV12` hardware video surface into an `RGBA8888` pixel buffer using BT.709.
    pub fn to_rgba_bt709(&self) -> Vec<u8> {
        let w = self.width as usize;
        let h = self.height as usize;
        let uv_w = w / 2;
        let mut rgba = vec![0u8; w * h * 4];
        for py in 0..h {
            for px in 0..w {
                let y = self.y_plane[py * w + px];
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AtlasRegion {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

#[derive(Clone, Debug)]
pub struct TextureAtlas {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
    cursor_x: u32,
    cursor_y: u32,
    row_height: u32,
}

impl TextureAtlas {
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

    /// Allocate a sub-rectangle in the shelf atlas and upload `rgba` pixels into it.
    pub fn allocate(&mut self, width: u32, height: u32, rgba: &[u8]) -> Option<AtlasRegion> {
        if width == 0 || height == 0 || rgba.len() < (width as usize) * (height as usize) * 4 {
            return None;
        }
        if width > self.width || height > self.height {
            return None;
        }
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

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum QuadKind {
    Solid,
    LinearGradient { end_color: GpuColor, angle_deg: f32 },
    BoxShadow { blur_radius: f32 },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuQuad {
    pub rect: GpuRect,
    pub clip: GpuRect,
    pub fill_color: GpuColor,
    pub border_color: GpuColor,
    pub corner_radius: f32,
    pub border_width: f32,
    pub uv_rect: Option<[f32; 4]>,
    pub kind: QuadKind,
}

impl GpuQuad {
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

    pub fn with_clip(mut self, clip: GpuRect) -> Self {
        self.clip = clip;
        self
    }

    pub fn with_uv(mut self, uv_rect: [f32; 4]) -> Self {
        self.uv_rect = Some(uv_rect);
        self
    }
}

/// Evaluate the signed distance from `rel_pos` (relative to box center) to a rounded box.
pub fn sdf_rounded_box(rel_x: f32, rel_y: f32, half_w: f32, half_h: f32, radius: f32) -> f32 {
    let r = radius.min(half_w).min(half_h).max(0.0);
    let qx = rel_x.abs() - (half_w - r);
    let qy = rel_y.abs() - (half_h - r);
    let outside = (qx.max(0.0).powi(2) + qy.max(0.0).powi(2)).sqrt();
    let inside = qx.max(qy).min(0.0);
    outside + inside - r
}

/// Evaluate smoothstep SDF falloff alpha (`0.0..=1.0`) for a box shadow with `blur_radius`.
pub fn sdf_box_shadow_alpha(dist: f32, blur_radius: f32) -> f32 {
    let sigma = blur_radius.max(0.5);
    let t = (0.5 - dist / sigma).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Sample a two-stop linear gradient across a box of `(half_w, half_h)` at `angle_deg`.
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

/// Render a list of [`GpuQuad`] primitives into an RGBA framebuffer of `(width, height)`.
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
                    let tx = ((sample_x - quad.rect.x) / quad.rect.width.max(1.0)).clamp(0.0, 1.0);
                    let ty = ((sample_y - quad.rect.y) / quad.rect.height.max(1.0)).clamp(0.0, 1.0);
                    let u = uv[0] + (uv[2] - uv[0]) * tx;
                    let v = uv[1] + (uv[3] - uv[1]) * ty;
                    let sampled = tex.sample_uv(u, v);
                    color = modulate_color(color, sampled);
                }

                if quad.border_width > 0.0 {
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

fn modulate_color(a: GpuColor, b: GpuColor) -> GpuColor {
    GpuColor {
        r: ((u16::from(a.r) * u16::from(b.r) + 127) / 255) as u8,
        g: ((u16::from(a.g) * u16::from(b.g) + 127) / 255) as u8,
        b: ((u16::from(a.b) * u16::from(b.b) + 127) / 255) as u8,
        a: ((u16::from(a.a) * u16::from(b.a) + 127) / 255) as u8,
    }
}

fn lerp_color(a: GpuColor, b: GpuColor, t: f32) -> GpuColor {
    let inv = 1.0 - t;
    GpuColor {
        r: ((a.r as f32) * inv + (b.r as f32) * t).round() as u8,
        g: ((a.g as f32) * inv + (b.g as f32) * t).round() as u8,
        b: ((a.b as f32) * inv + (b.b as f32) * t).round() as u8,
        a: ((a.a as f32) * inv + (b.a as f32) * t).round() as u8,
    }
}

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
}
