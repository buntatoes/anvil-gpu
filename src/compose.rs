// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 Buntato
//! Compositor drawing on the CPU: blurred box shadows, linear and radial
//! gradients, and video frame blits.
//!
//! These are the reference results. A device backend has to match them within
//! one step per channel (see [`crate::Compositor`]). `shaders/passes.glsl` and
//! `shaders/passes.hlsl` spell out the same math for a device.
//!
//! Pixels are straight (not premultiplied) RGBA8. A pixel is drawn when its
//! center lies inside the shape and the clip. Shadows and gradients are
//! blended with straight-alpha source-over ([`over_straight`]). Blits replace
//! the destination pixels.

use crate::{GpuColor, GpuGradientStop, GpuRect, sample_multi_stop_gradient, sdf_rounded_box};

/// Largest accepted target, frame, or blit side, in pixels.
pub const MAX_SIDE: u32 = 16384;
/// Blur radii above this are treated as this value.
pub const MAX_BLUR_RADIUS: f32 = 256.0;
/// Most gradient stops accepted. The shader passes hold this many.
pub const MAX_GRADIENT_STOPS: usize = 16;

/// Why a draw was refused. Nothing is drawn when one of these comes back.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ComposeError {
    /// Target size, stride, or buffer length does not fit together.
    BadTarget,
    /// Frame size, a stride, or a plane length does not fit together.
    BadFrame,
    /// A coordinate, radius, size, or stop is not finite or is out of range.
    BadInput,
    /// A device backend failed. [`crate::Compositor`] never returns this; it
    /// draws the same thing in software instead.
    Device,
}

impl std::fmt::Display for ComposeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            ComposeError::BadTarget => "target size, stride, or length is unusable",
            ComposeError::BadFrame => "frame size, stride, or plane length is unusable",
            ComposeError::BadInput => "draw input is not finite or out of range",
            ComposeError::Device => "device backend failed",
        })
    }
}

impl std::error::Error for ComposeError {}

/// A straight RGBA8 destination with a row stride in bytes.
#[derive(Debug)]
pub struct RgbaTarget<'a> {
    pixels: &'a mut [u8],
    width: u32,
    height: u32,
    stride: usize,
}

impl<'a> RgbaTarget<'a> {
    /// Wraps `pixels` as `width` by `height`, `stride` bytes per row.
    ///
    /// Each side must be at most [`MAX_SIDE`], `stride` at least `width * 4`,
    /// and `pixels` long enough for the last row. The last row does not need
    /// padding. A zero width or height is allowed and draws nothing.
    pub fn new(
        pixels: &'a mut [u8],
        width: u32,
        height: u32,
        stride: usize,
    ) -> Result<Self, ComposeError> {
        if width > MAX_SIDE || height > MAX_SIDE {
            return Err(ComposeError::BadTarget);
        }
        let row = (width as usize) * 4;
        if stride < row {
            return Err(ComposeError::BadTarget);
        }
        if width > 0 && height > 0 {
            let needed = stride
                .checked_mul(height as usize - 1)
                .and_then(|n| n.checked_add(row))
                .ok_or(ComposeError::BadTarget)?;
            if pixels.len() < needed {
                return Err(ComposeError::BadTarget);
            }
        }
        Ok(Self {
            pixels,
            width,
            height,
            stride,
        })
    }

    /// Wraps tightly packed pixels (`stride == width * 4`).
    pub fn packed(pixels: &'a mut [u8], width: u32, height: u32) -> Result<Self, ComposeError> {
        Self::new(pixels, width, height, (width as usize) * 4)
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    /// Bytes from the start of one row to the next.
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// The four bytes at `(x, y)`, or `None` outside the target.
    pub fn pixel(&self, x: u32, y: u32) -> Option<[u8; 4]> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let i = self.offset(x, y);
        Some([
            self.pixels[i],
            self.pixels[i + 1],
            self.pixels[i + 2],
            self.pixels[i + 3],
        ])
    }

    fn offset(&self, x: u32, y: u32) -> usize {
        (y as usize) * self.stride + (x as usize) * 4
    }

    fn px_mut(&mut self, x: u32, y: u32) -> &mut [u8] {
        let i = self.offset(x, y);
        &mut self.pixels[i..i + 4]
    }

    /// Copy of the pixels inside `bounds`, row by row.
    pub(crate) fn save(&self, bounds: Bounds) -> Vec<u8> {
        let row = (bounds.width() as usize) * 4;
        let mut out = Vec::with_capacity(row * bounds.height() as usize);
        for y in bounds.y0..bounds.y1 {
            let i = self.offset(bounds.x0, y);
            out.extend_from_slice(&self.pixels[i..i + row]);
        }
        out
    }

    /// Puts back pixels taken by [`RgbaTarget::save`] with the same bounds.
    pub(crate) fn restore(&mut self, bounds: Bounds, saved: &[u8]) {
        let row = (bounds.width() as usize) * 4;
        for (n, y) in (bounds.y0..bounds.y1).enumerate() {
            let i = self.offset(bounds.x0, y);
            self.pixels[i..i + row].copy_from_slice(&saved[n * row..(n + 1) * row]);
        }
    }
}

/// Pixel rectangle inside a target. `x1` and `y1` are exclusive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bounds {
    pub x0: u32,
    pub y0: u32,
    pub x1: u32,
    pub y1: u32,
}

impl Bounds {
    pub fn width(&self) -> u32 {
        self.x1 - self.x0
    }

    pub fn height(&self) -> u32 {
        self.y1 - self.y0
    }

    fn intersect(self, other: Bounds) -> Option<Bounds> {
        let b = Bounds {
            x0: self.x0.max(other.x0),
            y0: self.y0.max(other.y0),
            x1: self.x1.min(other.x1),
            y1: self.y1.min(other.y1),
        };
        (b.x0 < b.x1 && b.y0 < b.y1).then_some(b)
    }
}

/// Pixels whose centers fall in `lo..hi`, clamped to `0..=limit`.
///
/// A center `i + 0.5` is inside when `lo <= i + 0.5 < hi`, which is
/// `ceil(lo - 0.5) <= i < ceil(hi - 0.5)`.
fn center_span(lo: f64, hi: f64, limit: u32) -> (u32, u32) {
    let clamp = |v: f64| v.clamp(0.0, limit as f64) as u32;
    (clamp((lo - 0.5).ceil()), clamp((hi - 0.5).ceil()))
}

fn rect_bounds(rect: GpuRect, width: u32, height: u32) -> Option<Bounds> {
    let (x0, x1) = center_span(rect.x as f64, rect.x as f64 + rect.width as f64, width);
    let (y0, y1) = center_span(rect.y as f64, rect.y as f64 + rect.height as f64, height);
    (x0 < x1 && y0 < y1).then_some(Bounds { x0, y0, x1, y1 })
}

fn rect_is_valid(rect: GpuRect) -> bool {
    rect.x.is_finite()
        && rect.y.is_finite()
        && rect.width.is_finite()
        && rect.height.is_finite()
        && rect.width >= 0.0
        && rect.height >= 0.0
}

fn target_bounds(
    target: &RgbaTarget<'_>,
    clip: Option<GpuRect>,
) -> Result<Option<Bounds>, ComposeError> {
    let all = Bounds {
        x0: 0,
        y0: 0,
        x1: target.width,
        y1: target.height,
    };
    if all.x1 == 0 || all.y1 == 0 {
        return Ok(None);
    }
    match clip {
        None => Ok(Some(all)),
        Some(c) if rect_is_valid(c) => Ok(rect_bounds(c, target.width, target.height)),
        Some(_) => Err(ComposeError::BadInput),
    }
}

/// Straight-alpha source-over of `src` with coverage `alpha` (0..=1).
///
/// `out_a = sa + da * (1 - sa)` and `out_rgb = (s * sa + d * da * (1 - sa)) / out_a`,
/// where `sa = src.a / 255 * alpha`. Each channel is rounded half up as
/// `floor(x + 0.5)`. This is the
/// `OverStraight` function in the shader passes.
pub fn over_straight(dst: &mut [u8], src: GpuColor, alpha: f32) {
    let sa = (src.a as f32 / 255.0) * alpha.clamp(0.0, 1.0);
    if sa <= 0.0 {
        return;
    }
    let da = dst[3] as f32 / 255.0;
    let keep = da * (1.0 - sa);
    let oa = sa + keep;
    // floor(x + 0.5), as in the shaders. f32::round differs for one value:
    // 0.5 - 2^-25 rounds to 0, but adding 0.5 to it rounds up to 1.0.
    let mix = |s: u8, d: u8| {
        ((s as f32 * sa + d as f32 * keep) / oa + 0.5)
            .floor()
            .clamp(0.0, 255.0) as u8
    };
    dst[0] = mix(src.r, dst[0]);
    dst[1] = mix(src.g, dst[1]);
    dst[2] = mix(src.b, dst[2]);
    dst[3] = (oa * 255.0 + 0.5).floor().clamp(0.0, 255.0) as u8;
}

/// A CSS-style box shadow: a rounded box, moved, grown, blurred, then painted.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShadowDesc {
    /// The box that casts the shadow, in target pixels.
    pub rect: GpuRect,
    /// Corner radius of that box.
    pub corner_radius: f32,
    pub offset_x: f32,
    pub offset_y: f32,
    /// Blur radius in pixels. The blur is close to a gaussian with sigma `blur_radius / 2`.
    pub blur_radius: f32,
    /// Grows (or, when negative, shrinks) the box on every side. A nonzero
    /// corner radius grows or shrinks with it. A zero radius stays zero.
    pub spread: f32,
    pub color: GpuColor,
    /// Optional clip in target pixels.
    pub clip: Option<GpuRect>,
}

impl ShadowDesc {
    /// A sharp shadow exactly under `rect`. Set the other fields as needed.
    pub fn new(rect: GpuRect, color: GpuColor) -> Self {
        Self {
            rect,
            corner_radius: 0.0,
            offset_x: 0.0,
            offset_y: 0.0,
            blur_radius: 0.0,
            spread: 0.0,
            color,
            clip: None,
        }
    }

    fn is_valid(&self) -> bool {
        rect_is_valid(self.rect)
            && [
                self.corner_radius,
                self.offset_x,
                self.offset_y,
                self.blur_radius,
                self.spread,
            ]
            .iter()
            .all(|v| v.is_finite())
            && self.corner_radius >= 0.0
            && self.blur_radius >= 0.0
    }
}

/// Box blur radius for each of the three passes.
///
/// Three box passes of width `w = 2r + 1` have variance `3 (w² - 1) / 12`.
/// Setting that to `sigma²` with `sigma = blur / 2` gives `w = sqrt(blur² + 1)`.
/// `r` is rounded to whole pixels. A blur of 2 gives 1, 4 gives 2, 10 gives 5.
pub fn shadow_pass_radius(blur_radius: f32) -> u32 {
    let blur = blur_radius.clamp(0.0, MAX_BLUR_RADIUS);
    (((blur * blur + 1.0).sqrt() - 1.0) * 0.5).round() as u32
}

/// Where a shadow draws and the mask it needs. The device passes use the same plan.
pub(crate) struct ShadowPlan {
    pub(crate) out: Bounds,
    // Mask origin in target pixels. It may lie outside the target.
    pub(crate) mask_x: i64,
    pub(crate) mask_y: i64,
    pub(crate) mask_w: usize,
    pub(crate) mask_h: usize,
    pub(crate) radius: usize,
    pub(crate) cx: f32,
    pub(crate) cy: f32,
    pub(crate) half_w: f32,
    pub(crate) half_h: f32,
    pub(crate) corner: f32,
}

pub(crate) fn plan_shadow(
    target: &RgbaTarget<'_>,
    desc: &ShadowDesc,
) -> Result<Option<ShadowPlan>, ComposeError> {
    if !desc.is_valid() {
        return Err(ComposeError::BadInput);
    }
    let Some(area) = target_bounds(target, desc.clip)? else {
        return Ok(None);
    };
    if desc.color.a == 0 {
        return Ok(None);
    }
    let x = desc.rect.x as f64 + desc.offset_x as f64 - desc.spread as f64;
    let y = desc.rect.y as f64 + desc.offset_y as f64 - desc.spread as f64;
    let w = desc.rect.width as f64 + 2.0 * desc.spread as f64;
    let h = desc.rect.height as f64 + 2.0 * desc.spread as f64;
    if w <= 0.0 || h <= 0.0 {
        return Ok(None);
    }
    let radius = shadow_pass_radius(desc.blur_radius) as usize;
    // Coverage reaches half a pixel past the edge, and the blur 3r more.
    let reach = 1.0 + 3.0 * radius as f64;
    let (x0, x1) = center_span(x - reach, x + w + reach, target.width);
    let (y0, y1) = center_span(y - reach, y + h + reach, target.height);
    if x0 >= x1 || y0 >= y1 {
        return Ok(None);
    }
    let Some(out) = (Bounds { x0, y0, x1, y1 }).intersect(area) else {
        return Ok(None);
    };
    let margin = 3 * radius;
    Ok(Some(ShadowPlan {
        out,
        mask_x: out.x0 as i64 - margin as i64,
        mask_y: out.y0 as i64 - margin as i64,
        mask_w: out.width() as usize + 2 * margin,
        mask_h: out.height() as usize + 2 * margin,
        radius,
        cx: (x + w * 0.5) as f32,
        cy: (y + h * 0.5) as f32,
        half_w: (w * 0.5) as f32,
        half_h: (h * 0.5) as f32,
        // A sharp box keeps sharp corners when it grows. A rounded one grows
        // its radius with the spread.
        corner: if desc.corner_radius > 0.0 {
            (desc.corner_radius + desc.spread).max(0.0)
        } else {
            0.0
        },
    }))
}

/// Pixels a shadow may change, or `None` when it draws nothing.
pub(crate) fn shadow_bounds(
    target: &RgbaTarget<'_>,
    desc: &ShadowDesc,
) -> Result<Option<Bounds>, ComposeError> {
    Ok(plan_shadow(target, desc)?.map(|p| p.out))
}

/// Fixed-point one for the blurred shadow mask: 15 fraction bits.
///
/// Before the box passes, coverage is rounded (ties to even) to a multiple of
/// `1 / MASK_ONE`, and every pass works on those integers. Sums of up to 257
/// such values stay below 2^24, so they are exact everywhere, including in
/// the f32 math of the shader passes, and a device gives the same mask as
/// this code bit for bit. Each pass rounds its mean half up:
/// `(2 * sum + w) / (2 * w)` with `w = 2r + 1`.
const MASK_ONE: u32 = 1 << 15;

/// Coverage in 0..=1 as a mask value.
fn mask_value(coverage: f32) -> u32 {
    (coverage * MASK_ONE as f32).round_ties_even() as u32
}

/// One box pass over a line of `n` mask values, fed one value at a time.
///
/// Output `i` is the rounded mean of inputs `i - r..=i + r`, with zeros
/// outside the line. The sums are exact integers, so it makes no difference
/// whether a whole line is blurred at once or values arrive one row at a time.
#[derive(Clone, Debug)]
struct BoxRun {
    n: usize,
    r: usize,
    width: u32,
    acc: u32,
    // Last 2r + 2 inputs, by index modulo the length.
    ring: Vec<u32>,
    pushed: usize,
    emitted: usize,
}

impl BoxRun {
    fn new(n: usize, r: usize) -> Self {
        Self {
            n,
            r,
            width: (2 * r + 1) as u32,
            acc: 0,
            ring: vec![0; 2 * r + 2],
            pushed: 0,
            emitted: 0,
        }
    }

    fn input(&self, k: usize) -> u32 {
        self.ring[k % self.ring.len()]
    }

    fn emit(&mut self) -> u32 {
        self.emitted += 1;
        (2 * self.acc + self.width) / (2 * self.width)
    }

    /// Feed input `pushed`. Returns the next output once it is known.
    fn push(&mut self, v: u32) -> Option<u32> {
        debug_assert!(self.pushed < self.n);
        let k = self.pushed;
        let slot = k % self.ring.len();
        self.pushed += 1;
        let first = self.r.min(self.n - 1);
        if k <= first {
            self.acc += v;
            self.ring[slot] = v;
            return (k == first).then(|| self.emit());
        }
        // `v` is input i + r + 1 for output i = k - r - 1.
        let i = k - self.r - 1;
        self.acc += v;
        if i >= self.r {
            self.acc -= self.input(i - self.r);
        }
        self.ring[slot] = v;
        Some(self.emit())
    }

    /// After all `n` inputs: the remaining outputs, one per call.
    fn finish(&mut self) -> Option<u32> {
        debug_assert_eq!(self.pushed, self.n);
        if self.emitted >= self.n {
            return None;
        }
        let i = self.emitted - 1;
        if i >= self.r {
            self.acc -= self.input(i - self.r);
        }
        Some(self.emit())
    }
}

/// One box pass over a whole line.
fn box_blur_line(src: &[u32], dst: &mut [u32], r: usize) {
    if src.is_empty() {
        return;
    }
    let mut run = BoxRun::new(src.len(), r);
    let mut out = 0;
    for &v in src {
        if let Some(o) = run.push(v) {
            dst[out] = o;
            out += 1;
        }
    }
    while let Some(o) = run.finish() {
        dst[out] = o;
        out += 1;
    }
}

/// Feed `v` through a chain of passes. Returns the last pass's output, if any.
fn through(passes: &mut [BoxRun], v: u32) -> Option<u32> {
    let mut v = v;
    for pass in passes {
        v = pass.push(v)?;
    }
    Some(v)
}

/// Next output of a chain whose first pass has had all its inputs.
fn drain(passes: &mut [BoxRun]) -> Option<u32> {
    for k in 0..passes.len() {
        while let Some(v) = passes[k].finish() {
            if let Some(o) = through(&mut passes[k + 1..], v) {
                return Some(o);
            }
        }
    }
    None
}

/// Rows of the shadow mask: coverage, then three box passes along the row.
struct MaskRows<'p> {
    plan: &'p ShadowPlan,
    coverage: Vec<f32>,
    a: Vec<u32>,
    b: Vec<u32>,
}

impl<'p> MaskRows<'p> {
    fn new(plan: &'p ShadowPlan) -> Self {
        Self {
            plan,
            coverage: vec![0.0; plan.mask_w],
            a: vec![0; plan.mask_w],
            b: vec![0; plan.mask_w],
        }
    }

    /// Coverage of mask row `j`: the rounded box, with the same one-pixel
    /// ramp quads use.
    fn coverage(&mut self, j: usize) -> &[f32] {
        let plan = self.plan;
        let sy = (plan.mask_y + j as i64) as f32 + 0.5 - plan.cy;
        for (i, c) in self.coverage.iter_mut().enumerate() {
            let sx = (plan.mask_x + i as i64) as f32 + 0.5 - plan.cx;
            let d = sdf_rounded_box(sx, sy, plan.half_w, plan.half_h, plan.corner);
            *c = (0.5 - d).clamp(0.0, 1.0);
        }
        &self.coverage
    }

    /// Mask row `j` as mask values, blurred along x.
    fn blurred(&mut self, j: usize) -> &[u32] {
        self.coverage(j);
        for (q, &c) in self.a.iter_mut().zip(&self.coverage) {
            *q = mask_value(c);
        }
        let r = self.plan.radius;
        box_blur_line(&self.a, &mut self.b, r);
        box_blur_line(&self.b, &mut self.a, r);
        box_blur_line(&self.a, &mut self.b, r);
        &self.b
    }
}

/// Draw a blurred box shadow in software.
///
/// Without blur, the coverage of each pixel is painted as is. With blur, the
/// mask is never held whole. Rows are made one at a time, blurred along x,
/// and streamed through three box passes per output column, so memory is
/// about `3 * (2r + 2)` integers per output column plus a few mask rows,
/// where `r` is [`shadow_pass_radius`]. At the largest blur and a 16384 pixel
/// wide target that is under 60 MB. The result is the same, bit for bit, as
/// blurring the whole mask at once. See `MASK_ONE` for the fixed-point mask.
pub fn draw_box_shadow(target: &mut RgbaTarget<'_>, desc: &ShadowDesc) -> Result<(), ComposeError> {
    let Some(plan) = plan_shadow(target, desc)? else {
        return Ok(());
    };
    let margin = 3 * plan.radius;
    let out_w = plan.out.width() as usize;
    let mut rows = MaskRows::new(&plan);
    let paint = |target: &mut RgbaTarget<'_>, j: usize, line: &[f32]| {
        // Mask row j lands on target row out.y0 + j - margin when inside the region.
        if j < margin || j >= margin + plan.out.height() as usize {
            return;
        }
        let y = plan.out.y0 + (j - margin) as u32;
        for (n, &cov) in line.iter().enumerate() {
            if cov > 0.0 {
                over_straight(target.px_mut(plan.out.x0 + n as u32, y), desc.color, cov);
            }
        }
    };
    if plan.radius == 0 {
        for j in 0..plan.mask_h {
            let line = rows.coverage(j)[margin..margin + out_w].to_vec();
            paint(target, j, &line);
        }
        return Ok(());
    }
    // Three box passes down each output column, fed one mask row at a time.
    let mut columns: Vec<[BoxRun; 3]> = (0..out_w)
        .map(|_| std::array::from_fn(|_| BoxRun::new(plan.mask_h, plan.radius)))
        .collect();
    let mut line = vec![0f32; out_w];
    let to_coverage = |q: u32| q as f32 / MASK_ONE as f32;
    let mut next = 0;
    for j in 0..plan.mask_h {
        let row = rows.blurred(j);
        let mut ready = false;
        for (n, passes) in columns.iter_mut().enumerate() {
            if let Some(v) = through(passes, row[margin + n]) {
                line[n] = to_coverage(v);
                ready = true;
            }
        }
        if ready {
            paint(target, next, &line);
            next += 1;
        }
    }
    while next < plan.mask_h {
        for (n, passes) in columns.iter_mut().enumerate() {
            line[n] = to_coverage(drain(passes).unwrap_or(0));
        }
        paint(target, next, &line);
        next += 1;
    }
    Ok(())
}

/// Where a gradient's colors run.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GradientShape {
    /// Stop offset 0 sits at the start point and 1 at the end point, along
    /// the line between them. Past either end the end color holds. A line
    /// shorter than a millionth of a pixel paints the last stop.
    Linear {
        start_x: f32,
        start_y: f32,
        end_x: f32,
        end_y: f32,
    },
    /// Offset is the distance from the center, measured in radii, so 1 is the
    /// ellipse edge. Radii below 0.5 are treated as 0.5, as in
    /// [`crate::sample_radial_gradient`].
    Radial {
        center_x: f32,
        center_y: f32,
        radius_x: f32,
        radius_y: f32,
    },
}

/// A gradient filling `rect`. Coordinates are target pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GradientDesc<'s> {
    pub rect: GpuRect,
    pub shape: GradientShape,
    /// 1 to [`MAX_GRADIENT_STOPS`] stops. Offsets are clamped to 0..=1 and an
    /// offset below an earlier one is raised to it.
    pub stops: &'s [GpuGradientStop],
    pub clip: Option<GpuRect>,
}

/// Stops after clamping and ordering, in a fixed array.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Stops {
    items: [GpuGradientStop; MAX_GRADIENT_STOPS],
    len: usize,
}

impl Stops {
    fn new(stops: &[GpuGradientStop]) -> Option<Self> {
        if stops.is_empty() || stops.len() > MAX_GRADIENT_STOPS {
            return None;
        }
        let mut items = [GpuGradientStop::new(0.0, GpuColor::rgba(0, 0, 0, 0)); MAX_GRADIENT_STOPS];
        let mut floor = 0.0f32;
        for (slot, stop) in items.iter_mut().zip(stops) {
            if !stop.offset.is_finite() {
                return None;
            }
            floor = stop.offset.clamp(0.0, 1.0).max(floor);
            *slot = GpuGradientStop::new(floor, stop.color);
        }
        Some(Self {
            items,
            len: stops.len(),
        })
    }

    pub(crate) fn as_slice(&self) -> &[GpuGradientStop] {
        &self.items[..self.len]
    }
}

pub(crate) fn plan_gradient(
    target: &RgbaTarget<'_>,
    desc: &GradientDesc<'_>,
) -> Result<Option<(Bounds, Stops)>, ComposeError> {
    if !rect_is_valid(desc.rect) {
        return Err(ComposeError::BadInput);
    }
    let shape_ok = match desc.shape {
        GradientShape::Linear {
            start_x,
            start_y,
            end_x,
            end_y,
        } => [start_x, start_y, end_x, end_y]
            .iter()
            .all(|v| v.is_finite()),
        GradientShape::Radial {
            center_x,
            center_y,
            radius_x,
            radius_y,
        } => {
            [center_x, center_y, radius_x, radius_y]
                .iter()
                .all(|v| v.is_finite())
                && radius_x >= 0.0
                && radius_y >= 0.0
        }
    };
    if !shape_ok {
        return Err(ComposeError::BadInput);
    }
    let stops = Stops::new(desc.stops).ok_or(ComposeError::BadInput)?;
    let Some(area) = target_bounds(target, desc.clip)? else {
        return Ok(None);
    };
    let Some(fill) = rect_bounds(desc.rect, target.width, target.height) else {
        return Ok(None);
    };
    Ok(fill.intersect(area).map(|b| (b, stops)))
}

/// Pixels a gradient may change, or `None` when it draws nothing.
pub(crate) fn gradient_bounds(
    target: &RgbaTarget<'_>,
    desc: &GradientDesc<'_>,
) -> Result<Option<Bounds>, ComposeError> {
    Ok(plan_gradient(target, desc)?.map(|(b, _)| b))
}

/// Stop parameter of a linear gradient at a point.
pub fn linear_gradient_t(
    start_x: f32,
    start_y: f32,
    end_x: f32,
    end_y: f32,
    px: f32,
    py: f32,
) -> f32 {
    let dx = end_x - start_x;
    let dy = end_y - start_y;
    let len2 = dx * dx + dy * dy;
    if len2 < 1e-12 {
        return 1.0;
    }
    (((px - start_x) * dx + (py - start_y) * dy) / len2).clamp(0.0, 1.0)
}

/// Fill a gradient in software.
pub fn draw_gradient(
    target: &mut RgbaTarget<'_>,
    desc: &GradientDesc<'_>,
) -> Result<(), ComposeError> {
    let Some((bounds, stops)) = plan_gradient(target, desc)? else {
        return Ok(());
    };
    let stops = stops.as_slice();
    for y in bounds.y0..bounds.y1 {
        let py = y as f32 + 0.5;
        for x in bounds.x0..bounds.x1 {
            let px = x as f32 + 0.5;
            let color = match desc.shape {
                GradientShape::Linear {
                    start_x,
                    start_y,
                    end_x,
                    end_y,
                } => sample_multi_stop_gradient(
                    stops,
                    linear_gradient_t(start_x, start_y, end_x, end_y, px, py),
                ),
                GradientShape::Radial {
                    center_x,
                    center_y,
                    radius_x,
                    radius_y,
                } => crate::sample_radial_gradient(
                    center_x, center_y, radius_x, radius_y, px, py, stops,
                ),
            };
            over_straight(target.px_mut(x, y), color, 1.0);
        }
    }
    Ok(())
}

/// A decoded video frame. Planes are read, never written.
///
/// YUV frames are limited-range BT.709, the same as [`crate::Nv12Surface`].
/// Chroma planes are `ceil(width / 2)` by `ceil(height / 2)`, and each chroma
/// sample covers a 2 by 2 block of luma.
///
/// Chroma siting: [`Filter::Bilinear`] treats each chroma sample as sitting at
/// the center of its 2 by 2 luma block, horizontally and vertically. Most
/// BT.709 video (MPEG-2, H.264, and HEVC with default settings) sites chroma
/// on the left edge of the block instead, horizontally in line with the
/// even luma column. For that video the bilinear chroma is offset by half a
/// luma pixel to the right, which shows only as slight color fringes on
/// sharp color edges. [`Filter::Nearest`] uses the block's chroma as is, so
/// siting does not change it. The GLSL and HLSL blit passes use the same
/// center siting and the same integer bilinear math.
#[derive(Clone, Copy, Debug)]
pub enum VideoFrame<'f> {
    /// Straight RGBA8, `stride` bytes per row.
    Rgba {
        pixels: &'f [u8],
        width: u32,
        height: u32,
        stride: usize,
    },
    /// Luma plane, then one plane of interleaved Cb, Cr pairs.
    Nv12 {
        y: &'f [u8],
        y_stride: usize,
        uv: &'f [u8],
        uv_stride: usize,
        width: u32,
        height: u32,
    },
    /// Luma, Cb, and Cr in three planes.
    I420 {
        y: &'f [u8],
        y_stride: usize,
        u: &'f [u8],
        u_stride: usize,
        v: &'f [u8],
        v_stride: usize,
        width: u32,
        height: u32,
    },
}

fn plane_fits(len: usize, stride: usize, row: usize, rows: usize) -> bool {
    stride >= row
        && stride
            .checked_mul(rows - 1)
            .and_then(|n| n.checked_add(row))
            .is_some_and(|n| len >= n)
}

impl VideoFrame<'_> {
    pub fn width(&self) -> u32 {
        match *self {
            VideoFrame::Rgba { width, .. }
            | VideoFrame::Nv12 { width, .. }
            | VideoFrame::I420 { width, .. } => width,
        }
    }

    pub fn height(&self) -> u32 {
        match *self {
            VideoFrame::Rgba { height, .. }
            | VideoFrame::Nv12 { height, .. }
            | VideoFrame::I420 { height, .. } => height,
        }
    }

    /// Wraps an [`crate::Nv12Surface`] (tightly packed planes).
    pub fn from_nv12(surface: &crate::Nv12Surface) -> VideoFrame<'_> {
        VideoFrame::Nv12 {
            y: &surface.y_plane,
            y_stride: surface.width as usize,
            uv: &surface.uv_plane,
            uv_stride: surface.width as usize,
            width: surface.width,
            height: surface.height,
        }
    }

    /// Checks sizes, strides, and plane lengths.
    pub fn validate(&self) -> Result<(), ComposeError> {
        let (w, h) = (self.width(), self.height());
        if w == 0 || h == 0 || w > MAX_SIDE || h > MAX_SIDE {
            return Err(ComposeError::BadFrame);
        }
        let (w, h) = (w as usize, h as usize);
        let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
        let ok = match *self {
            VideoFrame::Rgba { pixels, stride, .. } => plane_fits(pixels.len(), stride, w * 4, h),
            VideoFrame::Nv12 {
                y,
                y_stride,
                uv,
                uv_stride,
                ..
            } => plane_fits(y.len(), y_stride, w, h) && plane_fits(uv.len(), uv_stride, cw * 2, ch),
            VideoFrame::I420 {
                y,
                y_stride,
                u,
                u_stride,
                v,
                v_stride,
                ..
            } => {
                plane_fits(y.len(), y_stride, w, h)
                    && plane_fits(u.len(), u_stride, cw, ch)
                    && plane_fits(v.len(), v_stride, cw, ch)
            }
        };
        if ok {
            Ok(())
        } else {
            Err(ComposeError::BadFrame)
        }
    }

    // Luma (or the whole pixel for RGBA) at integer texel `(x, y)`.
    fn luma(&self, x: usize, y: usize) -> f32 {
        match *self {
            VideoFrame::Nv12 { y: p, y_stride, .. } | VideoFrame::I420 { y: p, y_stride, .. } => {
                p[y * y_stride + x] as f32
            }
            VideoFrame::Rgba { .. } => 0.0,
        }
    }

    // Cb, Cr at integer chroma texel `(x, y)`.
    fn chroma(&self, x: usize, y: usize) -> (f32, f32) {
        match *self {
            VideoFrame::Nv12 { uv, uv_stride, .. } => {
                let i = y * uv_stride + x * 2;
                (uv[i] as f32, uv[i + 1] as f32)
            }
            VideoFrame::I420 {
                u,
                u_stride,
                v,
                v_stride,
                ..
            } => (u[y * u_stride + x] as f32, v[y * v_stride + x] as f32),
            VideoFrame::Rgba { .. } => (128.0, 128.0),
        }
    }

    fn rgba(&self, x: usize, y: usize) -> [u8; 4] {
        match *self {
            VideoFrame::Rgba { pixels, stride, .. } => {
                let i = y * stride + x * 4;
                [pixels[i], pixels[i + 1], pixels[i + 2], pixels[i + 3]]
            }
            _ => [0, 0, 0, 0],
        }
    }
}

/// How a blit samples the frame when the sizes differ.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Filter {
    /// The texel under each destination pixel center. Exact at 1:1.
    Nearest,
    /// Two by two texels, clamped at the frame edges. RGBA is weighted by
    /// alpha so clear texels do not darken their neighbors. YUV chroma is
    /// treated as center sited; see [`VideoFrame`] for what that means for
    /// left-sited video.
    Bilinear,
}

/// Where a frame lands: scaled to `width` by `height` at `(x, y)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlitDesc {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
    pub filter: Filter,
    pub clip: Option<GpuRect>,
}

impl BlitDesc {
    /// Nearest-texel blit of the whole frame into the given rectangle.
    pub fn new(x: i32, y: i32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
            filter: Filter::Nearest,
            clip: None,
        }
    }
}

/// Pixels a blit may change, or `None` when it draws nothing.
pub(crate) fn blit_bounds(
    target: &RgbaTarget<'_>,
    frame: &VideoFrame<'_>,
    desc: &BlitDesc,
) -> Result<Option<Bounds>, ComposeError> {
    frame.validate()?;
    if desc.width > MAX_SIDE || desc.height > MAX_SIDE {
        return Err(ComposeError::BadInput);
    }
    let Some(area) = target_bounds(target, desc.clip)? else {
        return Ok(None);
    };
    if desc.width == 0 || desc.height == 0 {
        return Ok(None);
    }
    let clamp_x = |v: i64| v.clamp(0, target.width as i64) as u32;
    let clamp_y = |v: i64| v.clamp(0, target.height as i64) as u32;
    let b = Bounds {
        x0: clamp_x(desc.x as i64),
        y0: clamp_y(desc.y as i64),
        x1: clamp_x(desc.x as i64 + desc.width as i64),
        y1: clamp_y(desc.y as i64 + desc.height as i64),
    };
    if b.x0 >= b.x1 || b.y0 >= b.y1 {
        return Ok(None);
    }
    Ok(b.intersect(area))
}

/// Limited-range BT.709 to RGB from 8-bit code values, in f32.
///
/// Same coefficients and rounding as [`crate::bt709_yuv_to_rgba`], so whole
/// code values give the same bytes. Nearest blits use it; bilinear blits use
/// an integer version so that devices match exactly.
pub fn bt709_yuv_f32_to_rgb(y: f32, u: f32, v: f32) -> [u8; 3] {
    let yf = (y - 16.0).max(0.0) * 1.164384;
    let uf = u - 128.0;
    let vf = v - 128.0;
    [
        (yf + 1.792741 * vf).round().clamp(0.0, 255.0) as u8,
        (yf - 0.213249 * uf - 0.532909 * vf)
            .round()
            .clamp(0.0, 255.0) as u8,
        (yf + 2.112402 * uf).round().clamp(0.0, 255.0) as u8,
    ]
}

/// Texel index under destination pixel `d` of `dst` when scaling from `src`.
/// Integer math: `floor((d + 0.5) * src / dst)`.
fn nearest_index(d: u64, src: u64, dst: u64) -> usize {
    (((2 * d + 1) * src) / (2 * dst)).min(src - 1) as usize
}

/// Bilinear weights along one axis are whole multiples of `1 / BILINEAR_ONE`.
///
/// Positions, weights, and sums are integers, on the CPU and in the shader
/// passes alike, and every sum fits in 32 bits as the shaders need. So a
/// device that runs the passes gives the same bytes: there is no float sum
/// that could land on a rounding boundary one way here and the other way on
/// the device.
const BILINEAR_ONE: u64 = 4096;

/// Bilinear taps along one axis for destination pixel `d`: two clamped
/// texel indices and the weight of the second, in `0..BILINEAR_ONE`.
///
/// The sample position is `(d + 0.5) * src / dst - 0.5` texels, rounded to
/// the nearest multiple of `1 / BILINEAR_ONE` (halves up). `dst` is the
/// destination size, times two for chroma, whose texels are twice as wide;
/// that puts chroma centers in the middle of each 2 by 2 luma block.
fn fixed_taps(d: u64, src: u64, dst: u64, size: usize) -> (usize, usize, u64) {
    // The position plus one half texel, which is never below zero:
    // round((2d + 1) * src * ONE / (2 dst)). The shader splits this
    // division to stay inside 32 bits; the result is the same.
    let den = 2 * dst;
    let p = (2 * (2 * d + 1) * src * BILINEAR_ONE + den) / (2 * den);
    // Shift by the half texel the other way, staying at or above zero.
    let p = p + BILINEAR_ONE / 2;
    let w = p % BILINEAR_ONE;
    // `p / BILINEAR_ONE - 1` is the texel at or left of the sample.
    let i = (p / BILINEAR_ONE) as i64 - 1;
    let last = size as i64 - 1;
    (
        i.clamp(0, last) as usize,
        (i + 1).clamp(0, last) as usize,
        w,
    )
}

/// Limited-range BT.709 to RGB from code values in 1/256 steps.
///
/// Integer coefficients in 1/8192 steps, so the CPU and the shader passes
/// give the same bytes. Whole code values can come out one step away from
/// [`bt709_yuv_f32_to_rgb`], which nearest blits use.
fn bt709_fixed_to_rgb(y: i64, u: i64, v: i64) -> [u8; 3] {
    let yf = (y - 16 * 256).max(0) * 9539;
    let uf = u - 128 * 256;
    let vf = v - 128 * 256;
    // Rounded half up from 1/(256 * 8192) steps. Below zero is clamped first.
    let out = |s: i64| ((s.max(0) + (1 << 20)) >> 21).min(255) as u8;
    [
        out(yf + 14686 * vf),
        out(yf - 1747 * uf - 4366 * vf),
        out(yf + 17305 * uf),
    ]
}

/// One bilinear RGBA sample, alpha-weighted, from the four texels
/// `[top left, top right, bottom left, bottom right]`.
///
/// Rows first: each row is mixed along x in premultiplied color, rounded to
/// 1/16 steps, then the two rows are mixed along y. Alpha is mixed exactly.
/// Every intermediate value stays below 2^32.
fn bilinear_rgba(texels: [[u8; 4]; 4], wx: u64, wy: u64) -> [u8; 4] {
    let one = BILINEAR_ONE;
    let row = |l: [u8; 4], r: [u8; 4]| {
        let (al, ar) = (l[3] as u64, r[3] as u64);
        let mut pre = [0u64; 3];
        for (c, p) in pre.iter_mut().enumerate() {
            let v = l[c] as u64 * al * (one - wx) + r[c] as u64 * ar * wx;
            *p = (v + 128) >> 8;
        }
        (pre, al * (one - wx) + ar * wx)
    };
    let (top, top_a) = row(texels[0], texels[1]);
    let (bottom, bottom_a) = row(texels[2], texels[3]);
    let alpha = top_a * (one - wy) + bottom_a * wy;
    // Alpha in 1/65536 steps, to match the color sums.
    let alpha16 = (alpha + 128) >> 8;
    if alpha16 == 0 {
        return [0, 0, 0, 0];
    }
    let mut out = [0u8; 4];
    for c in 0..3 {
        let v = top[c] * (one - wy) + bottom[c] * wy;
        out[c] = ((v + alpha16 / 2) / alpha16).min(255) as u8;
    }
    out[3] = ((alpha + (1 << 23)) >> 24).min(255) as u8;
    out
}

/// One bilinear plane sample from four code values, in 1/256 steps.
fn bilinear_plane(texels: [u8; 4], wx: u64, wy: u64) -> i64 {
    let one = BILINEAR_ONE;
    let k = [
        (one - wx) * (one - wy),
        wx * (one - wy),
        (one - wx) * wy,
        wx * wy,
    ];
    // At most 255 * 2^24, which fits in 32 bits.
    let sum: u64 = texels.iter().zip(&k).map(|(&t, &k)| t as u64 * k).sum();
    ((sum + (1 << 15)) >> 16) as i64
}

/// Copy (and scale and convert) a frame into the target in software.
pub fn draw_blit(
    target: &mut RgbaTarget<'_>,
    frame: &VideoFrame<'_>,
    desc: &BlitDesc,
) -> Result<(), ComposeError> {
    let Some(bounds) = blit_bounds(target, frame, desc)? else {
        return Ok(());
    };
    let (sw, sh) = (frame.width() as usize, frame.height() as usize);
    let (cw, ch) = (sw.div_ceil(2), sh.div_ceil(2));
    for y in bounds.y0..bounds.y1 {
        let ly = (y as i64 - desc.y as i64) as u64;
        for x in bounds.x0..bounds.x1 {
            let lx = (x as i64 - desc.x as i64) as u64;
            let out = match desc.filter {
                Filter::Nearest => {
                    let tx = nearest_index(lx, sw as u64, desc.width as u64);
                    let ty = nearest_index(ly, sh as u64, desc.height as u64);
                    match frame {
                        VideoFrame::Rgba { .. } => frame.rgba(tx, ty),
                        _ => {
                            let (u, v) = frame.chroma(tx / 2, ty / 2);
                            let [r, g, b] = bt709_yuv_f32_to_rgb(frame.luma(tx, ty), u, v);
                            [r, g, b, 255]
                        }
                    }
                }
                Filter::Bilinear => {
                    let (dw, dh) = (desc.width as u64, desc.height as u64);
                    let (x0, x1, wx) = fixed_taps(lx, sw as u64, dw, sw);
                    let (y0, y1, wy) = fixed_taps(ly, sh as u64, dh, sh);
                    match frame {
                        VideoFrame::Rgba { .. } => bilinear_rgba(
                            [
                                frame.rgba(x0, y0),
                                frame.rgba(x1, y0),
                                frame.rgba(x0, y1),
                                frame.rgba(x1, y1),
                            ],
                            wx,
                            wy,
                        ),
                        _ => {
                            let code = |v: f32| v as u8;
                            let luma = bilinear_plane(
                                [
                                    code(frame.luma(x0, y0)),
                                    code(frame.luma(x1, y0)),
                                    code(frame.luma(x0, y1)),
                                    code(frame.luma(x1, y1)),
                                ],
                                wx,
                                wy,
                            );
                            let (c0, c1, cwx) = fixed_taps(lx, sw as u64, 2 * dw, cw);
                            let (r0, r1, cwy) = fixed_taps(ly, sh as u64, 2 * dh, ch);
                            let at = [(c0, r0), (c1, r0), (c0, r1), (c1, r1)];
                            let chroma = at.map(|(tx, ty)| frame.chroma(tx, ty));
                            let u = bilinear_plane(chroma.map(|c| code(c.0)), cwx, cwy);
                            let v = bilinear_plane(chroma.map(|c| code(c.1)), cwx, cwy);
                            let [r, g, b] = bt709_fixed_to_rgb(luma, u, v);
                            [r, g, b, 255]
                        }
                    }
                }
            };
            target.px_mut(x, y).copy_from_slice(&out);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Nv12Surface;

    const WHITE: GpuColor = GpuColor::rgba(255, 255, 255, 255);
    const BLACK: GpuColor = GpuColor::rgba(0, 0, 0, 255);

    fn filled(w: u32, h: u32, c: GpuColor) -> Vec<u8> {
        [c.r, c.g, c.b, c.a].repeat((w * h) as usize)
    }

    fn at(buf: &[u8], w: u32, x: u32, y: u32) -> [u8; 4] {
        let i = ((y * w + x) * 4) as usize;
        [buf[i], buf[i + 1], buf[i + 2], buf[i + 3]]
    }

    #[test]
    fn over_straight_matches_hand_math() {
        // 200 * 128/255 + 10 * 127/255 = 105.37, and so on.
        let mut px = [10, 20, 30, 255];
        over_straight(&mut px, GpuColor::rgba(200, 100, 50, 128), 1.0);
        assert_eq!(px, [105, 60, 40, 255]);
        // Over a clear pixel the straight color comes through unchanged.
        let mut px = [0, 0, 0, 0];
        over_straight(&mut px, GpuColor::rgba(200, 100, 50, 128), 1.0);
        assert_eq!(px, [200, 100, 50, 128]);
        // Half coverage of opaque red over opaque blue: 127.5 rounds up.
        let mut px = [0, 0, 255, 255];
        over_straight(&mut px, GpuColor::rgba(255, 0, 0, 255), 0.5);
        assert_eq!(px, [128, 0, 128, 255]);
        // No coverage leaves the pixel alone.
        let mut px = [1, 2, 3, 4];
        over_straight(&mut px, WHITE, 0.0);
        assert_eq!(px, [1, 2, 3, 4]);
    }

    #[test]
    fn pass_radius_follows_three_box_variance() {
        assert_eq!(shadow_pass_radius(0.0), 0);
        assert_eq!(shadow_pass_radius(1.0), 0);
        assert_eq!(shadow_pass_radius(2.0), 1);
        assert_eq!(shadow_pass_radius(4.0), 2);
        assert_eq!(shadow_pass_radius(10.0), 5);
        assert_eq!(shadow_pass_radius(1e9), shadow_pass_radius(MAX_BLUR_RADIUS));
    }

    #[test]
    fn sharp_shadow_covers_exactly_the_box() {
        let mut buf = filled(8, 8, WHITE);
        let mut t = RgbaTarget::packed(&mut buf, 8, 8).unwrap();
        draw_box_shadow(
            &mut t,
            &ShadowDesc::new(GpuRect::new(2.0, 2.0, 4.0, 3.0), BLACK),
        )
        .unwrap();
        for y in 0..8 {
            for x in 0..8 {
                let inside = (2..6).contains(&x) && (2..5).contains(&y);
                let want = if inside { [0, 0, 0, 255] } else { [255; 4] };
                assert_eq!(at(&buf, 8, x, y), want, "pixel {x},{y}");
            }
        }
    }

    #[test]
    fn blurred_shadow_of_one_pixel_is_the_three_box_kernel() {
        // Blur 2 gives r = 1. Three passes of [1 1 1] / 3 on a single pixel give
        // [1 3 6 7 6 3 1] / 27 on each axis. The 2D weight is the product.
        let mut buf = filled(16, 16, WHITE);
        let mut t = RgbaTarget::packed(&mut buf, 16, 16).unwrap();
        let mut desc = ShadowDesc::new(GpuRect::new(5.0, 5.0, 1.0, 1.0), BLACK);
        desc.blur_radius = 2.0;
        desc.offset_x = 1.0;
        desc.offset_y = 2.0;
        draw_box_shadow(&mut t, &desc).unwrap();
        // The impulse lands on (6, 7).
        // Center: 255 * (1 - 49/729) = 237.86.
        assert_eq!(at(&buf, 16, 6, 7), [238, 238, 238, 255]);
        // One step right: 255 * (1 - 42/729) = 240.31.
        assert_eq!(at(&buf, 16, 7, 7), [240, 240, 240, 255]);
        // Three steps right: 255 * (1 - 7/729) = 252.55.
        assert_eq!(at(&buf, 16, 9, 7), [253, 253, 253, 255]);
        // Corner of the reach: 255 * (1 - 1/729) = 254.65.
        assert_eq!(at(&buf, 16, 9, 10), [255, 255, 255, 255]);
        // Past the reach nothing changes.
        assert_eq!(at(&buf, 16, 10, 7), [255; 4]);
        assert_eq!(at(&buf, 16, 2, 7), [255; 4]);
    }

    #[test]
    fn blur_keeps_the_shadow_mass() {
        // On a clear target the alpha sum is the box area times 255, give or
        // take rounding, as long as nothing is clipped.
        let mut buf = vec![0u8; 64 * 64 * 4];
        let mut t = RgbaTarget::packed(&mut buf, 64, 64).unwrap();
        let mut desc = ShadowDesc::new(GpuRect::new(20.0, 22.0, 18.0, 12.0), BLACK);
        desc.blur_radius = 9.0;
        desc.corner_radius = 4.0;
        draw_box_shadow(&mut t, &desc).unwrap();
        let sum: u64 = buf.iter().skip(3).step_by(4).map(|&a| a as u64).sum();
        // Rounded corners take away 4 * (r^2 - pi r^2 / 4) of the 216 px box.
        let area = 18.0 * 12.0 - (4.0 - std::f64::consts::PI) * 16.0;
        let got = sum as f64 / 255.0;
        assert!((got - area).abs() < 3.0, "mass {got} vs {area}");
    }

    // A plain box pass from prefix sums: the reference for the running sums
    // in BoxRun.
    fn whole_line(src: &[u32], dst: &mut [u32], r: usize) {
        let n = src.len();
        let w = (2 * r + 1) as u32;
        let mut prefix = vec![0u32; n + 1];
        for (i, &v) in src.iter().enumerate() {
            prefix[i + 1] = prefix[i] + v;
        }
        for (i, out) in dst.iter_mut().enumerate().take(n) {
            let sum = prefix[(i + r + 1).min(n)] - prefix[i.saturating_sub(r)];
            *out = (2 * sum + w) / (2 * w);
        }
    }

    // The whole mask in memory, blurred row by row and then column by
    // column: the reference for the streamed version.
    fn whole_mask_shadow(target: &mut RgbaTarget<'_>, desc: &ShadowDesc) {
        let Some(plan) = plan_shadow(target, desc).unwrap() else {
            return;
        };
        let (w, h, r) = (plan.mask_w, plan.mask_h, plan.radius);
        let mut cover = vec![0f32; w * h];
        for j in 0..h {
            let sy = (plan.mask_y + j as i64) as f32 + 0.5 - plan.cy;
            for i in 0..w {
                let sx = (plan.mask_x + i as i64) as f32 + 0.5 - plan.cx;
                let d = sdf_rounded_box(sx, sy, plan.half_w, plan.half_h, plan.corner);
                cover[j * w + i] = (0.5 - d).clamp(0.0, 1.0);
            }
        }
        if r > 0 {
            let mut mask: Vec<u32> = cover.iter().map(|&c| mask_value(c)).collect();
            let mut a = vec![0u32; w.max(h)];
            let mut b = vec![0u32; w.max(h)];
            for row in mask.chunks_exact_mut(w) {
                whole_line(row, &mut a[..w], r);
                whole_line(&a[..w], &mut b[..w], r);
                whole_line(&b[..w], row, r);
            }
            for x in 0..w {
                for y in 0..h {
                    a[y] = mask[y * w + x];
                }
                whole_line(&a[..h], &mut b[..h], r);
                whole_line(&b[..h], &mut a[..h], r);
                whole_line(&a[..h], &mut b[..h], r);
                for y in 0..h {
                    mask[y * w + x] = b[y];
                }
            }
            for (c, &q) in cover.iter_mut().zip(&mask) {
                *c = q as f32 / MASK_ONE as f32;
            }
        }
        let margin = 3 * r;
        for y in plan.out.y0..plan.out.y1 {
            let j = (y - plan.out.y0) as usize + margin;
            for x in plan.out.x0..plan.out.x1 {
                let i = (x - plan.out.x0) as usize + margin;
                let cov = cover[j * w + i];
                if cov > 0.0 {
                    over_straight(target.px_mut(x, y), desc.color, cov);
                }
            }
        }
    }

    #[test]
    fn running_box_pass_matches_the_plain_sum() {
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for _ in 0..400 {
            let n = 1 + (next() % 90) as usize;
            let r = (next() % 40) as usize;
            let src: Vec<u32> = (0..n)
                .map(|_| (next() % (MASK_ONE as u64 + 1)) as u32)
                .collect();
            let mut want = vec![0u32; n];
            let mut got = vec![0u32; n];
            whole_line(&src, &mut want, r);
            box_blur_line(&src, &mut got, r);
            assert_eq!(got, want, "n {n} r {r}");
        }
    }

    #[test]
    fn row_by_row_shadow_matches_the_whole_mask() {
        let mut seed = 0x0123_4567_89AB_CDEFu64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let mut f = move |lo: f32, hi: f32| lo + (next() % 10_000) as f32 / 10_000.0 * (hi - lo);
        for case in 0..300 {
            let (w, h) = (1 + f(0.0, 60.0) as u32, 1 + f(0.0, 60.0) as u32);
            let base: Vec<u8> = (0..w * h * 4).map(|i| (i * 37 + case) as u8).collect();
            let mut desc = ShadowDesc::new(
                GpuRect::new(f(-20.0, 60.0), f(-20.0, 60.0), f(0.0, 50.0), f(0.0, 50.0)),
                GpuColor::rgba(f(0.0, 255.0) as u8, 40, 200, f(1.0, 255.0) as u8),
            );
            desc.corner_radius = f(0.0, 15.0);
            desc.offset_x = f(-8.0, 8.0);
            desc.offset_y = f(-8.0, 8.0);
            desc.spread = f(-5.0, 8.0);
            // Mostly small blurs, now and then a huge one.
            desc.blur_radius = if case % 10 == 3 {
                f(100.0, 600.0)
            } else {
                f(0.0, 30.0)
            };
            if case % 4 == 1 {
                desc.clip = Some(GpuRect::new(
                    f(-5.0, 30.0),
                    f(-5.0, 30.0),
                    f(0.0, 40.0),
                    f(0.0, 40.0),
                ));
            }
            let mut rows = base.clone();
            let mut whole = base.clone();
            draw_box_shadow(&mut RgbaTarget::packed(&mut rows, w, h).unwrap(), &desc).unwrap();
            whole_mask_shadow(&mut RgbaTarget::packed(&mut whole, w, h).unwrap(), &desc);
            assert_eq!(rows, whole, "case {case}: {desc:?}");
        }
    }

    #[test]
    fn shadow_clip_spread_and_stride_are_respected() {
        // 6 by 4 target with 8 bytes of padding per row.
        let stride = 6 * 4 + 8;
        let mut buf = vec![7u8; stride * 4];
        for y in 0..4 {
            for x in 0..6 {
                buf[y * stride + x * 4..y * stride + x * 4 + 4].copy_from_slice(&[255; 4]);
            }
        }
        let mut t = RgbaTarget::new(&mut buf, 6, 4, stride).unwrap();
        let mut desc = ShadowDesc::new(GpuRect::new(2.0, 1.0, 2.0, 2.0), BLACK);
        desc.spread = 1.0;
        desc.clip = Some(GpuRect::new(0.0, 0.0, 3.0, 4.0));
        draw_box_shadow(&mut t, &desc).unwrap();
        // Spread 1 grows the box to x 1..5, y 0..4. The clip keeps x 0..3.
        assert_eq!(t.pixel(1, 0), Some([0, 0, 0, 255]));
        assert_eq!(t.pixel(2, 3), Some([0, 0, 0, 255]));
        assert_eq!(t.pixel(0, 0), Some([255; 4]));
        assert_eq!(t.pixel(3, 1), Some([255; 4]));
        for y in 0..4 {
            assert!(
                buf[y * stride + 24..y * stride + 32]
                    .iter()
                    .all(|&b| b == 7)
            );
        }
    }

    #[test]
    fn linear_gradient_matches_hand_values() {
        let stops = [
            GpuGradientStop::new(0.0, GpuColor::rgba(255, 0, 0, 255)),
            GpuGradientStop::new(1.0, GpuColor::rgba(0, 0, 255, 255)),
        ];
        let mut buf = filled(4, 1, BLACK);
        let mut t = RgbaTarget::packed(&mut buf, 4, 1).unwrap();
        let desc = GradientDesc {
            rect: GpuRect::new(0.0, 0.0, 4.0, 1.0),
            shape: GradientShape::Linear {
                start_x: 0.0,
                start_y: 0.5,
                end_x: 4.0,
                end_y: 0.5,
            },
            stops: &stops,
            clip: None,
        };
        draw_gradient(&mut t, &desc).unwrap();
        // Centers at t = 1/8, 3/8, 5/8, 7/8.
        assert_eq!(at(&buf, 4, 0, 0), [223, 0, 32, 255]);
        assert_eq!(at(&buf, 4, 1, 0), [159, 0, 96, 255]);
        assert_eq!(at(&buf, 4, 2, 0), [96, 0, 159, 255]);
        assert_eq!(at(&buf, 4, 3, 0), [32, 0, 223, 255]);
    }

    #[test]
    fn radial_gradient_matches_hand_values() {
        let stops = [
            GpuGradientStop::new(0.0, GpuColor::rgba(0, 0, 0, 255)),
            GpuGradientStop::new(1.0, GpuColor::rgba(255, 255, 255, 255)),
        ];
        let mut buf = filled(4, 4, BLACK);
        let mut t = RgbaTarget::packed(&mut buf, 4, 4).unwrap();
        let desc = GradientDesc {
            rect: GpuRect::new(0.0, 0.0, 4.0, 4.0),
            shape: GradientShape::Radial {
                center_x: 2.0,
                center_y: 2.0,
                radius_x: 2.0,
                radius_y: 2.0,
            },
            stops: &stops,
            clip: None,
        };
        draw_gradient(&mut t, &desc).unwrap();
        // (2.5, 2.5): sqrt(0.5) / 2 = 0.3536, 255 * 0.3536 = 90.2.
        assert_eq!(at(&buf, 4, 2, 2), [90, 90, 90, 255]);
        assert_eq!(at(&buf, 4, 1, 2), [90, 90, 90, 255]);
        // (0.5, 2.5): sqrt(2.5) / 2 = 0.7906, 201.6.
        assert_eq!(at(&buf, 4, 0, 2), [202, 202, 202, 255]);
        // (0.5, 0.5) is past the edge and holds the last stop.
        assert_eq!(at(&buf, 4, 0, 0), [255; 4]);
    }

    #[test]
    fn gradient_stops_are_ordered_and_checked() {
        let red = GpuColor::rgba(255, 0, 0, 255);
        let green = GpuColor::rgba(0, 255, 0, 255);
        let stops = [
            GpuGradientStop::new(0.6, red),
            GpuGradientStop::new(0.2, green),
        ];
        let line = GradientShape::Linear {
            start_x: 0.0,
            start_y: 0.0,
            end_x: 10.0,
            end_y: 0.0,
        };
        let mut buf = filled(10, 1, BLACK);
        let mut t = RgbaTarget::packed(&mut buf, 10, 1).unwrap();
        let desc = GradientDesc {
            rect: GpuRect::new(0.0, 0.0, 10.0, 1.0),
            shape: line,
            stops: &stops,
            clip: None,
        };
        draw_gradient(&mut t, &desc).unwrap();
        // The second stop is raised to 0.6, so the color switches there.
        assert_eq!(at(&buf, 10, 5, 0), [255, 0, 0, 255]);
        assert_eq!(at(&buf, 10, 6, 0), [0, 255, 0, 255]);

        let mut t = RgbaTarget::packed(&mut buf, 10, 1).unwrap();
        for bad in [
            &[][..],
            &[GpuGradientStop::new(f32::NAN, red)][..],
            &[GpuGradientStop::new(0.0, red); MAX_GRADIENT_STOPS + 1][..],
        ] {
            let desc = GradientDesc { stops: bad, ..desc };
            assert_eq!(draw_gradient(&mut t, &desc), Err(ComposeError::BadInput));
        }
    }

    fn test_surface(w: u32, h: u32) -> Nv12Surface {
        let mut rgba = Vec::new();
        for y in 0..h {
            for x in 0..w {
                rgba.extend_from_slice(&[
                    (x * 37 + y * 11) as u8,
                    (x * 5 + y * 53) as u8,
                    (x * y * 7) as u8,
                    255,
                ]);
            }
        }
        Nv12Surface::from_rgba_bt709(w, h, &rgba).unwrap()
    }

    #[test]
    fn nv12_and_i420_blits_match_the_cpu_planes() {
        let surface = test_surface(8, 6);
        let expected = surface.to_rgba_bt709();
        let mut buf = vec![0u8; 8 * 6 * 4];
        let mut t = RgbaTarget::packed(&mut buf, 8, 6).unwrap();
        draw_blit(
            &mut t,
            &VideoFrame::from_nv12(&surface),
            &BlitDesc::new(0, 0, 8, 6),
        )
        .unwrap();
        assert_eq!(buf, expected);

        let u: Vec<u8> = surface.uv_plane.iter().step_by(2).copied().collect();
        let v: Vec<u8> = surface
            .uv_plane
            .iter()
            .skip(1)
            .step_by(2)
            .copied()
            .collect();
        let i420 = VideoFrame::I420 {
            y: &surface.y_plane,
            y_stride: 8,
            u: &u,
            u_stride: 4,
            v: &v,
            v_stride: 4,
            width: 8,
            height: 6,
        };
        let mut buf2 = vec![0u8; 8 * 6 * 4];
        let mut t = RgbaTarget::packed(&mut buf2, 8, 6).unwrap();
        draw_blit(&mut t, &i420, &BlitDesc::new(0, 0, 8, 6)).unwrap();
        assert_eq!(buf2, expected);
    }

    #[test]
    fn float_conversion_matches_the_byte_conversion() {
        for y in (0..=255u16).step_by(3) {
            for u in (0..=255u16).step_by(5) {
                for v in (0..=255u16).step_by(5) {
                    let c = crate::bt709_yuv_to_rgba(y as u8, u as u8, v as u8);
                    assert_eq!(
                        bt709_yuv_f32_to_rgb(y as f32, u as f32, v as f32),
                        [c.r, c.g, c.b]
                    );
                }
            }
        }
    }

    #[test]
    fn rgba_blit_scales_with_both_filters() {
        let src = [10u8, 20, 30, 255, 200, 200, 200, 255];
        let frame = VideoFrame::Rgba {
            pixels: &src,
            width: 2,
            height: 1,
            stride: 8,
        };
        let mut buf = vec![0u8; 4 * 4];
        let mut t = RgbaTarget::packed(&mut buf, 4, 1).unwrap();
        draw_blit(&mut t, &frame, &BlitDesc::new(0, 0, 4, 1)).unwrap();
        assert_eq!(buf, [src[..4].repeat(2), src[4..].repeat(2)].concat());

        // Centers land at texel -0.25, 0.25, 0.75, 1.25.
        let src = [0u8, 0, 0, 255, 200, 100, 40, 255];
        let frame = VideoFrame::Rgba {
            pixels: &src,
            width: 2,
            height: 1,
            stride: 8,
        };
        let mut t = RgbaTarget::packed(&mut buf, 4, 1).unwrap();
        let mut desc = BlitDesc::new(0, 0, 4, 1);
        desc.filter = Filter::Bilinear;
        draw_blit(&mut t, &frame, &desc).unwrap();
        assert_eq!(
            buf,
            [
                0, 0, 0, 255, 50, 25, 10, 255, 150, 75, 30, 255, 200, 100, 40, 255
            ]
        );

        // A clear texel gives no color to its neighbor, only less alpha.
        let src = [255u8, 0, 0, 0, 0, 0, 255, 255];
        let frame = VideoFrame::Rgba {
            pixels: &src,
            width: 2,
            height: 1,
            stride: 8,
        };
        let mut t = RgbaTarget::packed(&mut buf, 4, 1).unwrap();
        draw_blit(&mut t, &frame, &desc).unwrap();
        assert_eq!(&buf[4..8], &[0, 0, 255, 64]);
    }

    #[test]
    fn bilinear_yuv_of_a_flat_frame_is_flat() {
        let rgba = [90u8, 140, 200, 255].repeat(16);
        let surface = Nv12Surface::from_rgba_bt709(4, 4, &rgba).unwrap();
        let want = surface.to_rgba_bt709();
        let mut buf = vec![0u8; 7 * 5 * 4];
        let mut t = RgbaTarget::packed(&mut buf, 7, 5).unwrap();
        let mut desc = BlitDesc::new(0, 0, 7, 5);
        desc.filter = Filter::Bilinear;
        draw_blit(&mut t, &VideoFrame::from_nv12(&surface), &desc).unwrap();
        for px in (0..buf.len()).step_by(4) {
            assert_eq!(&buf[px..px + 4], &want[..4]);
        }
    }

    #[test]
    fn blit_is_clipped_to_the_target() {
        let src: Vec<u8> = (0..4u8).flat_map(|i| [i * 50, 0, 0, 255]).collect();
        let frame = VideoFrame::Rgba {
            pixels: &src,
            width: 4,
            height: 1,
            stride: 16,
        };
        let mut buf = vec![9u8; 3 * 4];
        let mut t = RgbaTarget::packed(&mut buf, 3, 1).unwrap();
        draw_blit(&mut t, &frame, &BlitDesc::new(-2, 0, 4, 1)).unwrap();
        assert_eq!(buf, [100, 0, 0, 255, 150, 0, 0, 255, 9, 9, 9, 9]);
    }

    #[test]
    fn bad_targets_and_frames_are_refused() {
        let mut buf = vec![0u8; 15];
        assert_eq!(
            RgbaTarget::packed(&mut buf, 2, 2).unwrap_err(),
            ComposeError::BadTarget
        );
        assert_eq!(
            RgbaTarget::new(&mut buf, 2, 1, 7).unwrap_err(),
            ComposeError::BadTarget
        );
        assert_eq!(
            RgbaTarget::packed(&mut buf, MAX_SIDE + 1, 0).unwrap_err(),
            ComposeError::BadTarget
        );
        assert!(RgbaTarget::packed(&mut buf, 0, 0).is_ok());
        // The last row needs no padding.
        assert!(RgbaTarget::new(&mut buf, 1, 2, 11).is_ok());

        let short = VideoFrame::Nv12 {
            y: &[16; 4],
            y_stride: 2,
            uv: &[128; 1],
            uv_stride: 2,
            width: 2,
            height: 2,
        };
        assert_eq!(short.validate(), Err(ComposeError::BadFrame));
        let narrow = VideoFrame::Rgba {
            pixels: &[0; 64],
            width: 4,
            height: 2,
            stride: 12,
        };
        assert_eq!(narrow.validate(), Err(ComposeError::BadFrame));
        // Odd sizes round the chroma planes up.
        let odd = VideoFrame::I420 {
            y: &[16; 9],
            y_stride: 3,
            u: &[128; 4],
            u_stride: 2,
            v: &[128; 4],
            v_stride: 2,
            width: 3,
            height: 3,
        };
        assert_eq!(odd.validate(), Ok(()));
        let mut buf = vec![0u8; 3 * 3 * 4];
        let mut t = RgbaTarget::packed(&mut buf, 3, 3).unwrap();
        draw_blit(&mut t, &odd, &BlitDesc::new(0, 0, 3, 3)).unwrap();
        for (x, y) in [(0, 0), (2, 0), (1, 1), (2, 2)] {
            assert_eq!(t.pixel(x, y), Some([0, 0, 0, 255]));
        }

        let mut shadow = ShadowDesc::new(GpuRect::new(0.0, 0.0, 1.0, 1.0), BLACK);
        shadow.blur_radius = f32::INFINITY;
        assert_eq!(
            draw_box_shadow(&mut t, &shadow),
            Err(ComposeError::BadInput)
        );
        shadow.blur_radius = -1.0;
        assert_eq!(
            draw_box_shadow(&mut t, &shadow),
            Err(ComposeError::BadInput)
        );
    }
}
