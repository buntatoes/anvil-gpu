//! Proprietary 2D quad compositor, shelf-packing RGBA texture atlas, and HLSL SDF shader pipeline.

/// Embedded HLSL Shader Model 5.0 source for the Direct3D compositor pipeline.
pub const COMPOSITOR_HLSL: &str = include_str!("../shaders/compositor.hlsl");

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
pub struct GpuQuad {
    pub rect: GpuRect,
    pub clip: GpuRect,
    pub fill_color: GpuColor,
    pub border_color: GpuColor,
    pub corner_radius: f32,
    pub border_width: f32,
    pub uv_rect: Option<[f32; 4]>,
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
        let min_x = quad.rect.x.max(quad.clip.x).floor().max(0.0) as u32;
        let min_y = quad.rect.y.max(quad.clip.y).floor().max(0.0) as u32;
        let max_x = (quad.rect.x + quad.rect.width)
            .min(quad.clip.x + quad.clip.width)
            .ceil()
            .min(width as f32) as u32;
        let max_y = (quad.rect.y + quad.rect.height)
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
                let dist = sdf_rounded_box(
                    sample_x - cx,
                    sample_y - cy,
                    half_w,
                    half_h,
                    quad.corner_radius,
                );
                let outer_alpha = (0.5 - dist).clamp(0.0, 1.0);
                if outer_alpha <= 0.0 {
                    continue;
                }

                let mut color = quad.fill_color;
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
    fn embeds_hlsl_shader_and_packs_atlas_regions() {
        assert!(COMPOSITOR_HLSL.contains("SdfRoundedBox"));
        let mut atlas = TextureAtlas::new(32, 32);
        let red_2x2 = vec![255, 0, 0, 255].repeat(4);
        let reg = atlas.allocate(2, 2, &red_2x2).unwrap();
        assert_eq!(reg.x, 0);
        assert_eq!(reg.y, 0);
        let uvs = atlas.uv_bounds(reg);
        let sampled = atlas.sample_uv(uvs[0], uvs[1]);
        assert_eq!(sampled, GpuColor::rgba(255, 0, 0, 255));
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
}
