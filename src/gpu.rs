// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 Buntato
//! The compositor passes on a device, shared by every device backend.
//!
//! [`PassDevice`] is the little a device has to do: make textures, run one
//! pass from `shaders/passes.*` into a texture, and read RGBA back. The
//! functions here plan each call exactly as the CPU code does (same bounds,
//! same mask size, same stop list), run the passes, and write the result
//! into the target. [`DeviceBackend`] wraps a device as a
//! [`ComposeBackend`]. [`self_test`] draws a fixed scene both ways and
//! compares the bytes; a device that does not match is not used.
#![cfg_attr(not(any(target_os = "linux", target_os = "windows")), allow(dead_code))]

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::backend::{BackendKind, ComposeBackend};
use crate::compose::{
    self, BlitDesc, ComposeError, Filter, GradientDesc, GradientShape, RgbaTarget, ShadowDesc,
    VideoFrame,
};
use crate::{GpuColor, GpuGradientStop, GpuRect};

/// Largest texture a call may make on a device, in texels. 2^24 texels is
/// 64 MB for RGBA8 or R32F. A call that needs more is drawn on the CPU.
pub(crate) const MAX_DEVICE_TEXELS: u64 = 1 << 24;

static DEVICE_PASS_DRAWS: AtomicU64 = AtomicU64::new(0);

/// Pass draws a device finished since the process started.
///
/// Each shadow is up to eight draws (mask, six blur passes, paint), and each
/// gradient or blit one. The count only grows when a GPU backend draws, so
/// it shows whether calls really ran on a device.
pub fn device_pass_draws() -> u64 {
    DEVICE_PASS_DRAWS.load(Ordering::Relaxed)
}

/// Texel layouts the passes use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TexFormat {
    /// Straight RGBA, one byte per channel.
    Rgba8,
    /// One byte: a luma or chroma plane.
    R8,
    /// Two bytes: interleaved Cb, Cr.
    Rg8,
    /// One f32: the shadow mask.
    R32f,
}

impl TexFormat {
    pub(crate) fn texel_bytes(self) -> usize {
        match self {
            TexFormat::Rgba8 | TexFormat::R32f => 4,
            TexFormat::R8 => 1,
            TexFormat::Rg8 => 2,
        }
    }
}

/// A texture on a device, by index into that device's list for the call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TexId(pub(crate) usize);

/// One fragment pass from `shaders/passes.*`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Pass {
    ShadowMask,
    BoxBlur,
    ShadowPaint,
    Gradient,
    BlitRgba,
    BlitNv12,
    BlitI420,
}

impl Pass {
    pub(crate) const ALL: [Pass; 7] = [
        Pass::ShadowMask,
        Pass::BoxBlur,
        Pass::ShadowPaint,
        Pass::Gradient,
        Pass::BlitRgba,
        Pass::BlitNv12,
        Pass::BlitI420,
    ];

    pub(crate) fn index(self) -> usize {
        self as usize
    }

    /// The `PASS_*` symbol that selects this pass in `passes.glsl`.
    #[cfg(target_os = "linux")]
    pub(crate) fn glsl_symbol(self) -> &'static str {
        crate::GLSL_PASSES[self.index()]
    }

    /// The pixel shader entry point for this pass in `passes.hlsl`.
    #[cfg(any(target_os = "windows", test))]
    pub(crate) fn hlsl_entry(self) -> &'static str {
        // Entry 0 is the vertex shader; the pixel shaders follow in pass order.
        crate::PASS_ENTRY_POINTS[self.index() + 1].0
    }
}

/// Texture inputs of the passes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Slot {
    Source,
    Mask,
    Dest,
    Frame,
    Luma,
    ChromaUv,
    ChromaU,
    ChromaV,
}

impl Slot {
    /// Sampler name in `passes.glsl`.
    #[cfg(target_os = "linux")]
    pub(crate) fn glsl_name(self) -> &'static std::ffi::CStr {
        match self {
            Slot::Source => c"uSource",
            Slot::Mask => c"uMask",
            Slot::Dest => c"uDest",
            Slot::Frame => c"uFrame",
            Slot::Luma => c"uLuma",
            Slot::ChromaUv => c"uChroma",
            Slot::ChromaU => c"uChromaU",
            Slot::ChromaV => c"uChromaV",
        }
    }

    /// Texture register in `passes.hlsl`: `t0` for `Source` up to `t7`.
    #[cfg(any(target_os = "windows", test))]
    pub(crate) fn hlsl_register(self) -> usize {
        self as usize
    }
}

/// Everything a pass reads besides textures. Field for field the
/// `PassConstants` buffer in `passes.hlsl` and the uniforms in `passes.glsl`.
/// The region and viewport come from the render target.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct PassParams {
    pub origin: [i32; 2],
    pub shape: [f32; 4],
    pub shadow_color: [f32; 4],
    pub geometry: [f32; 4],
    pub dst_rect: [i32; 4],
    pub axis: [i32; 2],
    pub radius: i32,
    pub margin: i32,
    pub corner: f32,
    pub kind: i32,
    pub stop_count: i32,
    pub filter: i32,
    pub stop_offsets: [f32; 16],
    pub stop_colors: [[f32; 4]; 16],
}

/// Byte size of the `PassConstants` buffer in `passes.hlsl`.
#[cfg(any(target_os = "windows", test))]
pub(crate) const HLSL_CONSTANT_BYTES: usize = 448;

/// `params` laid out as the `PassConstants` buffer in `passes.hlsl`, for a
/// render target of `width` by `height`. Offsets follow the HLSL packing
/// rules: members fill 16-byte registers in order, and arrays start a new
/// register with one register per element.
#[cfg(any(target_os = "windows", test))]
pub(crate) fn hlsl_constants(p: &PassParams, width: u32, height: u32) -> [u8; HLSL_CONSTANT_BYTES] {
    fn f(buf: &mut [u8], at: usize, values: &[f32]) {
        for (i, v) in values.iter().enumerate() {
            buf[at + i * 4..at + i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
    }
    fn i(buf: &mut [u8], at: usize, values: &[i32]) {
        for (n, v) in values.iter().enumerate() {
            buf[at + n * 4..at + n * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
    }
    let mut b = [0u8; HLSL_CONSTANT_BYTES];
    let (w, h) = (width as f32, height as f32);
    f(&mut b, 0, &[0.0, 0.0, w, h]); // Region
    f(&mut b, 16, &[w, h]); // ViewportSize
    i(&mut b, 24, &p.origin); // Origin
    f(&mut b, 32, &p.shape); // Shape
    f(&mut b, 48, &p.shadow_color); // ShadowColor
    f(&mut b, 64, &p.geometry); // Geometry
    i(&mut b, 80, &p.dst_rect); // DstRect
    i(&mut b, 96, &p.axis); // Axis
    i(&mut b, 104, &[p.radius, p.margin]); // Radius, Margin
    f(&mut b, 112, &[p.corner]); // Corner
    i(&mut b, 116, &[p.kind, p.stop_count, p.filter]); // Kind, StopCount, FilterMode
    f(&mut b, 128, &p.stop_offsets); // StopOffsets: four float4
    for (n, c) in p.stop_colors.iter().enumerate() {
        f(&mut b, 192 + n * 16, c); // StopColors
    }
    b
}

/// What a device must do to run the passes.
///
/// Textures made during one backend call stay alive until that call ends.
pub(crate) trait PassDevice {
    /// Largest texture or render target side.
    fn max_side(&self) -> u32;
    /// A texture filled from `data` (rows top first, no padding).
    fn upload(
        &mut self,
        format: TexFormat,
        width: u32,
        height: u32,
        data: &[u8],
    ) -> Result<TexId, ComposeError>;
    /// An empty texture that a pass can draw into.
    fn render_target(
        &mut self,
        format: TexFormat,
        width: u32,
        height: u32,
    ) -> Result<TexId, ComposeError>;
    /// Run `pass` over all of `out`.
    fn draw(
        &mut self,
        pass: Pass,
        out: TexId,
        inputs: &[(Slot, TexId)],
        params: &PassParams,
    ) -> Result<(), ComposeError>;
    /// Copy an RGBA8 texture into `out`, rows top first, no padding.
    fn read_rgba(&mut self, tex: TexId, out: &mut [u8]) -> Result<(), ComposeError>;
}

/// Whether a device call drew, or left the call to the CPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Outcome {
    Drawn,
    /// Too big for the device limits, or values so large that f32 math could
    /// overflow, where devices are not required to agree with the CPU.
    /// Nothing was changed.
    Cpu,
}

/// Largest gradient coordinate or radius sent to a device. Squares and
/// products of values this size stay far below the f32 limit, so the device
/// does the same IEEE operations as the CPU without overflow.
const MAX_DEVICE_COORD: f32 = 1e15;

fn fits(dev: &dyn PassDevice, width: usize, height: usize) -> bool {
    let side = dev.max_side() as usize;
    width <= side && height <= side && (width as u64) * (height as u64) <= MAX_DEVICE_TEXELS
}

fn draw(
    dev: &mut dyn PassDevice,
    pass: Pass,
    out: TexId,
    inputs: &[(Slot, TexId)],
    params: &PassParams,
) -> Result<(), ComposeError> {
    dev.draw(pass, out, inputs, params)?;
    DEVICE_PASS_DRAWS.fetch_add(1, Ordering::Relaxed);
    Ok(())
}

fn color(c: GpuColor) -> [f32; 4] {
    [c.r as f32, c.g as f32, c.b as f32, c.a as f32]
}

/// Upload the target pixels in `bounds`, run `pass` over them, and write the
/// result back.
fn draw_over_target(
    dev: &mut dyn PassDevice,
    target: &mut RgbaTarget<'_>,
    bounds: compose::Bounds,
    pass: Pass,
    inputs: &[(Slot, TexId)],
    params: &PassParams,
    with_dest: bool,
) -> Result<(), ComposeError> {
    let (w, h) = (bounds.width(), bounds.height());
    let mut all = inputs.to_vec();
    if with_dest {
        let dest = dev.upload(TexFormat::Rgba8, w, h, &target.save(bounds))?;
        all.push((Slot::Dest, dest));
    }
    let out = dev.render_target(TexFormat::Rgba8, w, h)?;
    draw(dev, pass, out, &all, params)?;
    let mut pixels = vec![0u8; (w as usize) * (h as usize) * 4];
    dev.read_rgba(out, &mut pixels)?;
    target.restore(bounds, &pixels);
    Ok(())
}

/// [`compose::draw_box_shadow`] on a device.
pub(crate) fn shadow(
    dev: &mut dyn PassDevice,
    target: &mut RgbaTarget<'_>,
    desc: &ShadowDesc,
) -> Result<Outcome, ComposeError> {
    let Some(plan) = compose::plan_shadow(target, desc)? else {
        return Ok(Outcome::Drawn);
    };
    if !fits(dev, plan.mask_w, plan.mask_h) {
        return Ok(Outcome::Cpu);
    }
    let (mw, mh) = (plan.mask_w as u32, plan.mask_h as u32);
    let mut a = dev.render_target(TexFormat::R32f, mw, mh)?;
    let mask = PassParams {
        // The mask starts at most 3 * 128 pixels before the target, so these fit.
        origin: [plan.mask_x as i32, plan.mask_y as i32],
        shape: [plan.cx, plan.cy, plan.half_w, plan.half_h],
        corner: plan.corner,
        ..PassParams::default()
    };
    draw(dev, Pass::ShadowMask, a, &[], &mask)?;
    if plan.radius > 0 {
        let mut b = dev.render_target(TexFormat::R32f, mw, mh)?;
        for axis in [[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1]] {
            let blur = PassParams {
                axis,
                radius: plan.radius as i32,
                ..PassParams::default()
            };
            draw(dev, Pass::BoxBlur, b, &[(Slot::Source, a)], &blur)?;
            std::mem::swap(&mut a, &mut b);
        }
    }
    let paint = PassParams {
        origin: [plan.out.x0 as i32, plan.out.y0 as i32],
        shadow_color: color(desc.color),
        margin: 3 * plan.radius as i32,
        ..PassParams::default()
    };
    draw_over_target(
        dev,
        target,
        plan.out,
        Pass::ShadowPaint,
        &[(Slot::Mask, a)],
        &paint,
        true,
    )?;
    Ok(Outcome::Drawn)
}

/// [`compose::draw_gradient`] on a device.
pub(crate) fn gradient(
    dev: &mut dyn PassDevice,
    target: &mut RgbaTarget<'_>,
    desc: &GradientDesc<'_>,
) -> Result<Outcome, ComposeError> {
    let Some((bounds, stops)) = compose::plan_gradient(target, desc)? else {
        return Ok(Outcome::Drawn);
    };
    let geometry = match desc.shape {
        GradientShape::Linear {
            start_x,
            start_y,
            end_x,
            end_y,
        } => [start_x, start_y, end_x, end_y],
        GradientShape::Radial {
            center_x,
            center_y,
            radius_x,
            radius_y,
        } => [center_x, center_y, radius_x, radius_y],
    };
    if !fits(dev, bounds.width() as usize, bounds.height() as usize)
        || geometry.iter().any(|v| v.abs() > MAX_DEVICE_COORD)
    {
        return Ok(Outcome::Cpu);
    }
    let stops = stops.as_slice();
    let mut params = PassParams {
        origin: [bounds.x0 as i32, bounds.y0 as i32],
        stop_count: stops.len() as i32,
        ..PassParams::default()
    };
    for (i, s) in stops.iter().enumerate() {
        params.stop_offsets[i] = s.offset;
        params.stop_colors[i] = color(s.color);
    }
    params.geometry = geometry;
    params.kind = match desc.shape {
        GradientShape::Linear { .. } => 0,
        GradientShape::Radial { .. } => 1,
    };
    draw_over_target(dev, target, bounds, Pass::Gradient, &[], &params, true)?;
    Ok(Outcome::Drawn)
}

/// Rows of a plane without their padding.
fn packed_plane(plane: &[u8], stride: usize, row: usize, rows: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(row * rows);
    for y in 0..rows {
        out.extend_from_slice(&plane[y * stride..y * stride + row]);
    }
    out
}

/// [`compose::draw_blit`] on a device.
pub(crate) fn blit(
    dev: &mut dyn PassDevice,
    target: &mut RgbaTarget<'_>,
    frame: &VideoFrame<'_>,
    desc: &BlitDesc,
) -> Result<Outcome, ComposeError> {
    let Some(bounds) = compose::blit_bounds(target, frame, desc)? else {
        return Ok(Outcome::Drawn);
    };
    let (fw, fh) = (frame.width(), frame.height());
    if !fits(dev, bounds.width() as usize, bounds.height() as usize)
        || !fits(dev, fw as usize, fh as usize)
    {
        return Ok(Outcome::Cpu);
    }
    let (w, h) = (fw as usize, fh as usize);
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    let (cwu, chu) = (cw as u32, ch as u32);
    // `blit_bounds` validated every plane, so the slices below are in range.
    let (pass, inputs) = match *frame {
        VideoFrame::Rgba { pixels, stride, .. } => {
            let f = dev.upload(
                TexFormat::Rgba8,
                fw,
                fh,
                &packed_plane(pixels, stride, w * 4, h),
            )?;
            (Pass::BlitRgba, vec![(Slot::Frame, f)])
        }
        VideoFrame::Nv12 {
            y,
            y_stride,
            uv,
            uv_stride,
            ..
        } => {
            let l = dev.upload(TexFormat::R8, fw, fh, &packed_plane(y, y_stride, w, h))?;
            let c = dev.upload(
                TexFormat::Rg8,
                cwu,
                chu,
                &packed_plane(uv, uv_stride, cw * 2, ch),
            )?;
            (Pass::BlitNv12, vec![(Slot::Luma, l), (Slot::ChromaUv, c)])
        }
        VideoFrame::I420 {
            y,
            y_stride,
            u,
            u_stride,
            v,
            v_stride,
            ..
        } => {
            let l = dev.upload(TexFormat::R8, fw, fh, &packed_plane(y, y_stride, w, h))?;
            let cu = dev.upload(TexFormat::R8, cwu, chu, &packed_plane(u, u_stride, cw, ch))?;
            let cv = dev.upload(TexFormat::R8, cwu, chu, &packed_plane(v, v_stride, cw, ch))?;
            (
                Pass::BlitI420,
                vec![(Slot::Luma, l), (Slot::ChromaU, cu), (Slot::ChromaV, cv)],
            )
        }
    };
    let params = PassParams {
        origin: [bounds.x0 as i32, bounds.y0 as i32],
        // A non-empty blit starts less than MAX_SIDE pixels left of or above
        // the target, and its size is at most MAX_SIDE, so these fit.
        dst_rect: [desc.x, desc.y, desc.width as i32, desc.height as i32],
        filter: match desc.filter {
            Filter::Nearest => 0,
            Filter::Bilinear => 1,
        },
        ..PassParams::default()
    };
    draw_over_target(dev, target, bounds, pass, &inputs, &params, false)?;
    Ok(Outcome::Drawn)
}

/// Draw a fixed scene with `draw_device` and in software and compare.
///
/// The scene covers every pass: blurred, sharp, and clipped shadows, linear
/// and radial gradients with several stops, and RGBA, NV12, and I420 blits
/// with both filters, on a target with row padding.
pub(crate) fn self_test(dev: &mut dyn PassDevice) -> bool {
    const W: u32 = 45;
    const H: u32 = 33;
    const STRIDE: usize = W as usize * 4 + 12;
    let mut base = vec![0u8; STRIDE * H as usize];
    for (i, b) in base.iter_mut().enumerate() {
        *b = (i * 7 + i / 5) as u8;
    }
    let mut want = base.clone();
    let mut got = base;
    let ok = (|| -> Result<bool, ComposeError> {
        let mut soft = RgbaTarget::new(&mut want, W, H, STRIDE)?;
        let mut hard = RgbaTarget::new(&mut got, W, H, STRIDE)?;
        for desc in self_test_shadows() {
            compose::draw_box_shadow(&mut soft, &desc)?;
            if shadow(dev, &mut hard, &desc)? != Outcome::Drawn {
                return Ok(false);
            }
        }
        let stops = [
            GpuGradientStop::new(0.0, GpuColor::rgba(250, 10, 30, 255)),
            GpuGradientStop::new(0.3, GpuColor::rgba(20, 200, 40, 180)),
            GpuGradientStop::new(0.3, GpuColor::rgba(0, 0, 250, 90)),
            GpuGradientStop::new(1.2, GpuColor::rgba(255, 255, 255, 255)),
        ];
        for (rect, shape) in [
            (
                GpuRect::new(1.5, 20.0, 30.0, 12.3),
                GradientShape::Linear {
                    start_x: 2.0,
                    start_y: 21.0,
                    end_x: 29.0,
                    end_y: 30.0,
                },
            ),
            (
                GpuRect::new(20.2, 2.0, 24.0, 20.0),
                GradientShape::Radial {
                    center_x: 31.0,
                    center_y: 11.0,
                    radius_x: 12.0,
                    radius_y: 7.5,
                },
            ),
        ] {
            let desc = GradientDesc {
                rect,
                shape,
                stops: &stops,
                clip: Some(GpuRect::new(0.0, 0.0, 44.0, 33.0)),
            };
            compose::draw_gradient(&mut soft, &desc)?;
            if gradient(dev, &mut hard, &desc)? != Outcome::Drawn {
                return Ok(false);
            }
        }
        let rgba: Vec<u8> = (0..5 * 3 * 4).map(|i| (i * 37 + 11) as u8).collect();
        let luma: Vec<u8> = (0..5 * 3).map(|i| (i * 53 + 16) as u8).collect();
        let uv: Vec<u8> = (0..3 * 2 * 2).map(|i| (i * 41 + 60) as u8).collect();
        let u: Vec<u8> = (0..3 * 2).map(|i| (i * 29 + 70) as u8).collect();
        let v: Vec<u8> = (0..3 * 2).map(|i| (i * 31 + 90) as u8).collect();
        let frames = [
            VideoFrame::Rgba {
                pixels: &rgba,
                width: 5,
                height: 3,
                stride: 20,
            },
            VideoFrame::Nv12 {
                y: &luma,
                y_stride: 5,
                uv: &uv,
                uv_stride: 6,
                width: 5,
                height: 3,
            },
            VideoFrame::I420 {
                y: &luma,
                y_stride: 5,
                u: &u,
                u_stride: 3,
                v: &v,
                v_stride: 3,
                width: 5,
                height: 3,
            },
        ];
        for (n, frame) in frames.iter().enumerate() {
            for filter in [Filter::Nearest, Filter::Bilinear] {
                let mut desc = BlitDesc::new(
                    -3 + 15 * n as i32,
                    3 + (filter == Filter::Bilinear) as i32 * 14,
                    17,
                    11,
                );
                desc.filter = filter;
                compose::draw_blit(&mut soft, frame, &desc)?;
                if blit(dev, &mut hard, frame, &desc)? != Outcome::Drawn {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    })();
    matches!(ok, Ok(true)) && want == got
}

fn self_test_shadows() -> [ShadowDesc; 3] {
    let mut soft = ShadowDesc::new(
        GpuRect::new(6.0, 5.5, 20.0, 12.0),
        GpuColor::rgba(10, 20, 30, 170),
    );
    soft.blur_radius = 7.0;
    soft.corner_radius = 4.0;
    soft.offset_x = 1.5;
    soft.offset_y = 2.25;
    soft.spread = 1.0;
    let mut sharp = ShadowDesc::new(
        GpuRect::new(30.3, 3.0, 9.0, 7.0),
        GpuColor::rgba(200, 40, 10, 255),
    );
    sharp.corner_radius = 2.5;
    let mut clipped = ShadowDesc::new(
        GpuRect::new(-4.0, 22.0, 18.0, 30.0),
        GpuColor::rgba(0, 90, 255, 128),
    );
    clipped.blur_radius = 3.0;
    clipped.spread = -1.5;
    clipped.clip = Some(GpuRect::new(0.0, 24.0, 10.0, 6.0));
    [soft, sharp, clipped]
}

/// A device that can run the passes for the duration of one call.
pub(crate) trait SharedDevice: Send + 'static {
    /// Run `f` with the device ready (for GL: its context current).
    fn with_passes<R>(
        &mut self,
        f: impl FnOnce(&mut dyn PassDevice) -> R,
    ) -> Result<R, ComposeError>;
}

/// Faults to put into a device on purpose, for the fallback tests.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Faults {
    /// Make every pass fail to compile.
    pub break_shaders: bool,
    /// Ask for a context version that does not exist.
    pub bad_context: bool,
    /// Fail the pass draw with this number (counting from 1) in a call.
    pub fail_draw: Option<u32>,
}

/// A [`PassDevice`] that fails one chosen draw.
struct FaultyDraws<'d> {
    inner: &'d mut dyn PassDevice,
    fail_draw: Option<u32>,
    draws: u32,
}

impl PassDevice for FaultyDraws<'_> {
    fn max_side(&self) -> u32 {
        self.inner.max_side()
    }
    fn upload(
        &mut self,
        format: TexFormat,
        width: u32,
        height: u32,
        data: &[u8],
    ) -> Result<TexId, ComposeError> {
        self.inner.upload(format, width, height, data)
    }
    fn render_target(
        &mut self,
        format: TexFormat,
        width: u32,
        height: u32,
    ) -> Result<TexId, ComposeError> {
        self.inner.render_target(format, width, height)
    }
    fn draw(
        &mut self,
        pass: Pass,
        out: TexId,
        inputs: &[(Slot, TexId)],
        params: &PassParams,
    ) -> Result<(), ComposeError> {
        self.draws += 1;
        if self.fail_draw == Some(self.draws) {
            return Err(ComposeError::Device);
        }
        self.inner.draw(pass, out, inputs, params)
    }
    fn read_rgba(&mut self, tex: TexId, out: &mut [u8]) -> Result<(), ComposeError> {
        self.inner.read_rgba(tex, out)
    }
}

/// A device as a [`ComposeBackend`]. Calls too big for the device limits are
/// drawn with the CPU passes, which give the same bytes.
pub(crate) struct DeviceBackend<D: SharedDevice> {
    device: Arc<Mutex<D>>,
    fail_draw: Option<u32>,
}

impl<D: SharedDevice> DeviceBackend<D> {
    pub(crate) fn new(device: Arc<Mutex<D>>, faults: Faults) -> Self {
        Self {
            device,
            fail_draw: faults.fail_draw,
        }
    }

    fn call(
        &mut self,
        target: &mut RgbaTarget<'_>,
        on_device: impl FnOnce(
            &mut dyn PassDevice,
            &mut RgbaTarget<'_>,
        ) -> Result<Outcome, ComposeError>,
        on_cpu: impl FnOnce(&mut RgbaTarget<'_>) -> Result<(), ComposeError>,
    ) -> Result<(), ComposeError> {
        let fail_draw = self.fail_draw;
        let outcome = {
            let mut device = self.device.lock().unwrap_or_else(|e| e.into_inner());
            device.with_passes(|dev| {
                let mut faulty = FaultyDraws {
                    inner: dev,
                    fail_draw,
                    draws: 0,
                };
                on_device(&mut faulty, target)
            })??
        };
        match outcome {
            Outcome::Drawn => Ok(()),
            Outcome::Cpu => on_cpu(target),
        }
    }
}

impl<D: SharedDevice> ComposeBackend for DeviceBackend<D> {
    fn kind(&self) -> BackendKind {
        BackendKind::Hardware
    }

    fn box_shadow(
        &mut self,
        target: &mut RgbaTarget<'_>,
        desc: &ShadowDesc,
    ) -> Result<(), ComposeError> {
        self.call(
            target,
            |d, t| shadow(d, t, desc),
            |t| compose::draw_box_shadow(t, desc),
        )
    }

    fn fill_gradient(
        &mut self,
        target: &mut RgbaTarget<'_>,
        desc: &GradientDesc<'_>,
    ) -> Result<(), ComposeError> {
        self.call(
            target,
            |d, t| gradient(d, t, desc),
            |t| compose::draw_gradient(t, desc),
        )
    }

    fn blit_frame(
        &mut self,
        target: &mut RgbaTarget<'_>,
        frame: &VideoFrame<'_>,
        desc: &BlitDesc,
    ) -> Result<(), ComposeError> {
        self.call(
            target,
            |d, t| blit(d, t, frame, desc),
            |t| compose::draw_blit(t, frame, desc),
        )
    }
}

/// Helpers shared by the device backend tests.
#[cfg(test)]
pub(crate) mod testing {
    use std::sync::mpsc;
    use std::time::Duration;

    use crate::{
        BlitDesc, Compositor, Filter, GpuColor, GpuGradientStop, GpuRect, GradientDesc,
        GradientShape, Nv12Surface, RgbaTarget, ShadowDesc, VideoFrame,
    };

    /// Longest one of these tests may take before it counts as hung.
    pub(crate) const TEST_LIMIT: Duration = Duration::from_secs(60);

    /// Run `f` on its own thread and fail if it panics or runs too long.
    pub(crate) fn within_limit(f: impl FnOnce() + Send + 'static) {
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            f();
            let _ = tx.send(());
        });
        match rx.recv_timeout(TEST_LIMIT) {
            Ok(()) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => panic!("the test body panicked"),
            Err(mpsc::RecvTimeoutError::Timeout) => panic!("the test body ran past {TEST_LIMIT:?}"),
        }
    }

    /// A shadow, two gradients, and three blits on a padded 50 by 40 target.
    pub(crate) fn draw_scene(c: &mut Compositor) -> Vec<u8> {
        const STRIDE: usize = 50 * 4 + 8;
        let mut buf: Vec<u8> = (0..STRIDE * 40).map(|i| (i * 13 + 7) as u8).collect();
        let mut t = RgbaTarget::new(&mut buf, 50, 40, STRIDE).unwrap();
        let mut shadow = ShadowDesc::new(
            GpuRect::new(8.0, 6.0, 22.0, 14.0),
            GpuColor::rgba(5, 10, 40, 180),
        );
        shadow.blur_radius = 9.0;
        shadow.corner_radius = 5.0;
        shadow.offset_y = 3.0;
        shadow.spread = 1.5;
        c.box_shadow(&mut t, &shadow).unwrap();
        let stops = [
            GpuGradientStop::new(0.0, GpuColor::rgba(250, 120, 20, 255)),
            GpuGradientStop::new(0.4, GpuColor::rgba(20, 220, 90, 128)),
            GpuGradientStop::new(1.0, GpuColor::rgba(20, 80, 250, 200)),
        ];
        let linear = GradientDesc {
            rect: GpuRect::new(2.0, 24.0, 30.0, 14.0),
            shape: GradientShape::Linear {
                start_x: 2.0,
                start_y: 24.0,
                end_x: 30.0,
                end_y: 36.0,
            },
            stops: &stops,
            clip: None,
        };
        c.fill_gradient(&mut t, &linear).unwrap();
        let radial = GradientDesc {
            rect: GpuRect::new(30.0, 2.0, 18.0, 20.0),
            shape: GradientShape::Radial {
                center_x: 39.0,
                center_y: 12.0,
                radius_x: 9.0,
                radius_y: 6.0,
            },
            stops: &stops,
            clip: Some(GpuRect::new(31.0, 3.0, 15.0, 15.0)),
        };
        c.fill_gradient(&mut t, &radial).unwrap();
        let rgba: Vec<u8> = (0..7 * 5 * 4).map(|i| (i * 29 + 3) as u8).collect();
        let surface = Nv12Surface::from_rgba_bt709(6, 4, &rgba[..6 * 4 * 4]).unwrap();
        let u = [100u8, 120, 140, 160, 180, 200];
        let v = [90u8, 70, 50, 150, 170, 190];
        let i420 = VideoFrame::I420 {
            y: &surface.y_plane,
            y_stride: 6,
            u: &u,
            u_stride: 3,
            v: &v,
            v_stride: 3,
            width: 6,
            height: 4,
        };
        let frames = [
            VideoFrame::Rgba {
                pixels: &rgba,
                width: 7,
                height: 5,
                stride: 28,
            },
            VideoFrame::from_nv12(&surface),
            i420,
        ];
        for (n, frame) in frames.iter().enumerate() {
            let mut blit = BlitDesc::new(34 + n as i32 * 3, 22 + n as i32 * 5, 13, 9);
            blit.filter = if n == 1 {
                Filter::Nearest
            } else {
                Filter::Bilinear
            };
            c.blit_frame(&mut t, frame, &blit).unwrap();
        }
        buf
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn word(b: &[u8], at: usize) -> [u8; 4] {
        b[at..at + 4].try_into().unwrap()
    }

    #[test]
    fn hlsl_constants_follow_the_cbuffer_in_passes_hlsl() {
        // The members, in order, as declared. The offsets used by
        // hlsl_constants follow from this order and the HLSL packing rules.
        let body = crate::PASSES_HLSL
            .split_once("cbuffer PassConstants : register(b0)")
            .and_then(|(_, rest)| rest.split_once("};"))
            .map(|(body, _)| body)
            .unwrap();
        let members: Vec<&str> = body
            .lines()
            .map(|l| l.split("//").next().unwrap().trim())
            .filter(|l| !l.is_empty() && *l != "{")
            .collect();
        assert_eq!(
            members,
            [
                "float4 Region;",
                "float2 ViewportSize;",
                "int2 Origin;",
                "float4 Shape;",
                "float4 ShadowColor;",
                "float4 Geometry;",
                "int4 DstRect;",
                "int2 Axis;",
                "int Radius;",
                "int Margin;",
                "float Corner;",
                "int Kind;",
                "int StopCount;",
                "int FilterMode;",
                "float4 StopOffsets[MAX_STOPS / 4];",
                "float4 StopColors[MAX_STOPS];",
            ]
        );
        let mut p = PassParams {
            origin: [-3, 4],
            shape: [1.0, 2.0, 3.0, 4.0],
            shadow_color: [5.0, 6.0, 7.0, 8.0],
            geometry: [9.0, 10.0, 11.0, 12.0],
            dst_rect: [13, 14, 15, 16],
            axis: [0, 1],
            radius: 17,
            margin: 51,
            corner: 2.5,
            kind: 1,
            stop_count: 16,
            filter: 1,
            ..PassParams::default()
        };
        for n in 0..16 {
            p.stop_offsets[n] = n as f32 / 15.0;
            p.stop_colors[n] = [n as f32, 1.0, 2.0, 255.0];
        }
        let b = hlsl_constants(&p, 37, 21);
        let f = |v: f32| v.to_le_bytes();
        let i = |v: i32| v.to_le_bytes();
        assert_eq!(word(&b, 8), f(37.0));
        assert_eq!(word(&b, 12), f(21.0));
        assert_eq!(word(&b, 16), f(37.0));
        assert_eq!(word(&b, 20), f(21.0));
        assert_eq!(word(&b, 24), i(-3));
        assert_eq!(word(&b, 28), i(4));
        assert_eq!(word(&b, 32), f(1.0));
        assert_eq!(word(&b, 60), f(8.0));
        assert_eq!(word(&b, 64), f(9.0));
        assert_eq!(word(&b, 92), i(16));
        assert_eq!(word(&b, 100), i(1));
        assert_eq!(word(&b, 104), i(17));
        assert_eq!(word(&b, 108), i(51));
        assert_eq!(word(&b, 112), f(2.5));
        assert_eq!(word(&b, 116), i(1));
        assert_eq!(word(&b, 120), i(16));
        assert_eq!(word(&b, 124), i(1));
        // StopOffsets[i / 4][i % 4], as StopOffset in the shader reads it.
        assert_eq!(word(&b, 128 + 7 * 4), f(7.0 / 15.0));
        assert_eq!(word(&b, 128 + 15 * 4), f(1.0));
        assert_eq!(word(&b, 192 + 9 * 16), f(9.0));
        assert_eq!(word(&b, 192 + 15 * 16 + 12), f(255.0));
        assert_eq!(b.len(), 192 + 16 * 16);
    }

    #[test]
    fn passes_map_to_their_hlsl_entries_and_registers() {
        let entries: Vec<_> = Pass::ALL.iter().map(|p| p.hlsl_entry()).collect();
        assert_eq!(
            entries,
            [
                "PSShadowMask",
                "PSBoxBlur",
                "PSShadowPaint",
                "PSGradient",
                "PSBlitRgba",
                "PSBlitNv12",
                "PSBlitI420",
            ]
        );
        for (slot, name) in [
            (Slot::Source, "Source"),
            (Slot::Mask, "Mask"),
            (Slot::Dest, "Dest"),
            (Slot::Frame, "Frame"),
            (Slot::Luma, "Luma"),
            (Slot::ChromaUv, "ChromaUV"),
            (Slot::ChromaU, "ChromaU"),
            (Slot::ChromaV, "ChromaV"),
        ] {
            let decl = format!(" {name} : register(t{});", slot.hlsl_register());
            assert!(crate::PASSES_HLSL.contains(&decl), "{decl}");
        }
    }
}
