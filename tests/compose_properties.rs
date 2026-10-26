// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 Buntato
//! Property checks for the compositor passes.
//!
//! `tests/robustness.rs` checks that bad input never panics and that a failed
//! device leaves the same pixels as software. These tests check what the
//! passes draw: which pixels a call may change, exact colors where the math
//! has one answer, mirror symmetry of a centered shadow, and agreement
//! between the NV12, I420, padded, and packed forms of the same frame.
//!
//! Each case is an 8-byte seed for its own generator, so a failure names a
//! case that reproduces on every run. Each test stops after a short time
//! budget.

use anvil_gpu::{
    BlitDesc, Compositor, Filter, GpuColor, GpuGradientStop, GpuRect, GradientDesc, GradientShape,
    MAX_SIDE, Nv12Surface, RgbaTarget, ShadowDesc, VideoFrame, draw_blit, draw_box_shadow,
    draw_gradient, shadow_pass_radius,
};

// ---- shared harness (same text in each repo's robustness test) ----

use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

/// Wall-clock budget for one robustness test. Cases run until it is spent.
/// Set `ANVIL_ROBUSTNESS_MS` for a longer local soak run.
fn test_budget() -> Duration {
    let millis = std::env::var("ANVIL_ROBUSTNESS_MS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(1500);
    Duration::from_millis(millis)
}

/// One input that runs longer than this counts as a hang.
const CASE_LIMIT: Duration = Duration::from_secs(10);

/// Upper bound on cases per test, so a fast machine does not loop forever.
const MAX_CASES: usize = 200_000;

/// xorshift64 generator. A fixed seed gives the same inputs on every run.
struct Rng(u64);

#[allow(dead_code)] // Not every test file uses every helper.
impl Rng {
    fn new(seed: u64) -> Self {
        Rng((seed ^ 0x9E37_79B9_7F4A_7C15) | 1)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Value in `0..n`. Returns 0 when `n` is 0.
    fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next_u64() % n as u64) as usize
        }
    }

    fn byte(&mut self) -> u8 {
        self.next_u64() as u8
    }

    fn one_in(&mut self, n: usize) -> bool {
        self.below(n) == 0
    }

    fn pick<T: Copy>(&mut self, items: &[T]) -> T {
        items[self.below(items.len())]
    }

    fn bytes(&mut self, max_len: usize) -> Vec<u8> {
        let len = self.below(max_len + 1);
        (0..len).map(|_| self.byte()).collect()
    }
}

/// Apply 1 to 8 random edits: bit flips, byte overwrites, boundary values,
/// inserts, deletes, duplicated runs, and truncation.
#[allow(dead_code)] // Only the seed generator is used here.
fn mutate(rng: &mut Rng, seed: &[u8]) -> Vec<u8> {
    let mut out = seed.to_vec();
    let edits = 1 + rng.below(8);
    for _ in 0..edits {
        let len = out.len();
        match rng.below(8) {
            0 if len > 0 => {
                let i = rng.below(len);
                out[i] ^= 1 << rng.below(8);
            }
            1 if len > 0 => {
                let i = rng.below(len);
                out[i] = rng.byte();
            }
            2 if len > 0 => {
                let i = rng.below(len);
                out[i] = rng.pick(&[0x00, 0x01, 0x7F, 0x80, 0xFE, 0xFF]);
            }
            3 if len >= 4 => {
                let i = rng.below(len - 3);
                let v = rng.pick(&[0u32, 1, 0x7FFF_FFFF, 0x8000_0000, 0xFFFF_FFFF, 0xFFFF]);
                out[i..i + 4].copy_from_slice(&v.to_be_bytes());
            }
            4 => {
                let i = rng.below(len + 1);
                let extra = rng.bytes(16);
                out.splice(i..i, extra);
            }
            5 if len > 0 => {
                let i = rng.below(len);
                let n = 1 + rng.below((len - i).min(32));
                out.drain(i..i + n);
            }
            6 if len > 0 => {
                let i = rng.below(len);
                let n = 1 + rng.below((len - i).min(64));
                let run = out[i..i + n].to_vec();
                let at = rng.below(len + 1);
                out.splice(at..at, run);
            }
            _ => {
                let keep = rng.below(len + 1);
                out.truncate(keep);
            }
        }
    }
    out
}

fn preview(input: &[u8]) -> String {
    let shown: Vec<String> = input.iter().take(96).map(|b| format!("{b:02x}")).collect();
    let more = if input.len() > 96 { " ..." } else { "" };
    format!("{} bytes: {}{}", input.len(), shown.join(" "), more)
}

/// Run `target` on `input` in its own thread. Fails the test on a panic or
/// when the case runs past [`CASE_LIMIT`]. A hung thread is left behind and
/// ends with the test process.
fn check_case(label: &str, case: usize, input: Vec<u8>, target: fn(&[u8])) {
    let (done_tx, done_rx) = mpsc::channel();
    let shared = std::sync::Arc::new(input);
    let worker_input = shared.clone();
    let handle = thread::Builder::new()
        .name(format!("{label}-{case}"))
        .stack_size(16 * 1024 * 1024)
        .spawn(move || {
            target(&worker_input);
            let _ = done_tx.send(());
        })
        .expect("spawn case thread");
    match done_rx.recv_timeout(CASE_LIMIT) {
        Ok(()) => {
            let _ = handle.join();
        }
        Err(mpsc::RecvTimeoutError::Disconnected) => {
            let _ = handle.join();
            panic!(
                "{label}: case {case} panicked on input {}",
                preview(&shared)
            );
        }
        Err(mpsc::RecvTimeoutError::Timeout) => {
            panic!(
                "{label}: case {case} did not finish within {:?} on input {}",
                CASE_LIMIT,
                preview(&shared)
            );
        }
    }
}

/// Generate cases from `seed` until the budget is spent and check each one.
fn run_cases(
    label: &str,
    seed: u64,
    mut generate: impl FnMut(&mut Rng) -> Vec<u8>,
    target: fn(&[u8]),
) {
    let mut rng = Rng::new(seed);
    let budget = test_budget();
    let started = Instant::now();
    let mut case = 0;
    while case < MAX_CASES && started.elapsed() < budget {
        let input = generate(&mut rng);
        check_case(label, case, input, target);
        case += 1;
    }
    eprintln!("{label}: {case} cases in {:?}", started.elapsed());
}

/// A case input is the seed of its own generator.
fn seed_input(rng: &mut Rng) -> Vec<u8> {
    rng.next_u64().to_le_bytes().to_vec()
}

fn case_rng(input: &[u8]) -> Rng {
    let mut seed = [0u8; 8];
    seed.copy_from_slice(&input[..8]);
    Rng::new(u64::from_le_bytes(seed))
}

/// A target buffer filled with random bytes, row padding included.
struct Canvas {
    buf: Vec<u8>,
    width: u32,
    height: u32,
    stride: usize,
}

impl Canvas {
    fn random(rng: &mut Rng) -> Self {
        let side = |rng: &mut Rng| -> u32 {
            match rng.below(6) {
                0 => rng.below(2) as u32,
                1 => 1 + 2 * rng.below(20) as u32,
                _ => 2 + rng.below(70) as u32,
            }
        };
        let width = side(rng);
        let height = side(rng);
        let stride = width as usize * 4 + rng.pick(&[0, 0, 1, 3, 4, 13]);
        let len = if width == 0 || height == 0 {
            rng.below(16)
        } else {
            stride * (height as usize - 1) + width as usize * 4 + rng.below(3)
        };
        let buf = (0..len).map(|_| rng.byte()).collect();
        Canvas {
            buf,
            width,
            height,
            stride,
        }
    }

    fn target(&mut self) -> RgbaTarget<'_> {
        RgbaTarget::new(&mut self.buf, self.width, self.height, self.stride)
            .expect("canvas sizes are valid")
    }

    fn at(&self, x: u32, y: u32) -> [u8; 4] {
        let i = y as usize * self.stride + x as usize * 4;
        [
            self.buf[i],
            self.buf[i + 1],
            self.buf[i + 2],
            self.buf[i + 3],
        ]
    }

    /// True when byte `i` is row padding or past the last row.
    fn is_padding(&self, i: usize) -> bool {
        if self.width == 0 || self.height == 0 {
            return true;
        }
        let row = i / self.stride;
        row >= self.height as usize || i % self.stride >= self.width as usize * 4
    }
}

/// Pixel centers of `x..x + len` (half open), as in the crate docs.
fn center_inside(i: u32, lo: f64, len: f64) -> bool {
    let c = i as f64 + 0.5;
    lo <= c && c < lo + len
}

fn rect_contains(r: GpuRect, x: u32, y: u32) -> bool {
    center_inside(x, r.x as f64, r.width as f64) && center_inside(y, r.y as f64, r.height as f64)
}

fn coord(rng: &mut Rng, size: u32) -> f32 {
    let s = size as f32;
    match rng.below(5) {
        0 => rng.below(size as usize + 1) as f32,
        1 => (rng.below(4 * size as usize + 8) as f32) * 0.25 - 2.0,
        2 => -(rng.below(40) as f32) - 0.5,
        _ => s * (rng.below(1000) as f32 / 1000.0),
    }
}

fn color(rng: &mut Rng) -> GpuColor {
    GpuColor::rgba(rng.byte(), rng.byte(), rng.byte(), rng.byte())
}

fn clip(rng: &mut Rng, w: u32, h: u32) -> Option<GpuRect> {
    rng.one_in(2).then(|| {
        GpuRect::new(
            coord(rng, w),
            coord(rng, h),
            coord(rng, w).abs() + rng.below(3) as f32,
            coord(rng, h).abs() + rng.below(3) as f32,
        )
    })
}

fn in_clip(c: Option<GpuRect>, x: u32, y: u32) -> bool {
    c.is_none_or(|c| rect_contains(c, x, y))
}

fn gradient_case(input: &[u8]) {
    let mut rng = case_rng(input);
    let mut canvas = Canvas::random(&mut rng);
    let before = canvas.buf.clone();
    let (w, h) = (canvas.width, canvas.height);
    let rect = GpuRect::new(
        coord(&mut rng, w),
        coord(&mut rng, h),
        coord(&mut rng, w).abs(),
        coord(&mut rng, h).abs(),
    );
    let c = GpuColor::rgba(rng.byte(), rng.byte(), rng.byte(), 255);
    let count = 1 + rng.below(16);
    let mut stops: Vec<GpuGradientStop> = (0..count)
        .map(|_| GpuGradientStop::new(rng.below(1200) as f32 / 1000.0 - 0.1, c))
        .collect();
    if rng.one_in(4) {
        stops.reverse();
    }
    let shape = if rng.one_in(2) {
        GradientShape::Linear {
            start_x: coord(&mut rng, w),
            start_y: coord(&mut rng, h),
            end_x: coord(&mut rng, w),
            end_y: coord(&mut rng, h),
        }
    } else {
        GradientShape::Radial {
            center_x: coord(&mut rng, w),
            center_y: coord(&mut rng, h),
            radius_x: coord(&mut rng, w).abs(),
            radius_y: coord(&mut rng, h).abs(),
        }
    };
    let clip = clip(&mut rng, w, h);
    let desc = GradientDesc {
        rect,
        shape,
        stops: &stops,
        clip,
    };
    let result = if rng.one_in(2) {
        draw_gradient(&mut canvas.target(), &desc)
    } else {
        Compositor::software().fill_gradient(&mut canvas.target(), &desc)
    };
    assert_eq!(result, Ok(()), "finite gradient refused");
    for y in 0..h {
        for x in 0..w {
            let got = canvas.at(x, y);
            if rect_contains(rect, x, y) && in_clip(clip, x, y) {
                assert_eq!(
                    got,
                    [c.r, c.g, c.b, 255],
                    "flat opaque gradient at ({x}, {y})"
                );
            } else {
                let i = y as usize * canvas.stride + x as usize * 4;
                assert_eq!(
                    &got[..],
                    &before[i..i + 4],
                    "gradient changed ({x}, {y}) outside its rectangle"
                );
            }
        }
    }
    for (i, (a, b)) in canvas.buf.iter().zip(&before).enumerate() {
        if canvas.is_padding(i) {
            assert_eq!(a, b, "gradient wrote padding byte {i}");
        }
    }
}

fn shadow_desc(rng: &mut Rng, w: u32, h: u32) -> ShadowDesc {
    let rect = GpuRect::new(
        coord(rng, w),
        coord(rng, h),
        coord(rng, w).abs(),
        coord(rng, h).abs(),
    );
    let mut desc = ShadowDesc::new(rect, color(rng));
    desc.corner_radius = rng.below(30) as f32 * 0.5;
    desc.offset_x = rng.below(21) as f32 - 10.0;
    desc.offset_y = rng.below(21) as f32 - 10.0;
    desc.blur_radius = match rng.below(4) {
        0 => 0.0,
        1 => rng.below(8) as f32 * 0.5,
        2 => rng.below(40) as f32,
        _ => rng.pick(&[255.0, 256.0, 300.0, 1.0e6]),
    };
    desc.spread = rng.below(21) as f32 - 10.0;
    desc.clip = clip(rng, w, h);
    desc
}

fn shadow_reach_case(input: &[u8]) {
    let mut rng = case_rng(input);
    let mut canvas = Canvas::random(&mut rng);
    let before = canvas.buf.clone();
    let (w, h) = (canvas.width, canvas.height);
    let desc = shadow_desc(&mut rng, w, h);
    let result = if rng.one_in(2) {
        draw_box_shadow(&mut canvas.target(), &desc)
    } else {
        Compositor::software().box_shadow(&mut canvas.target(), &desc)
    };
    assert_eq!(result, Ok(()), "finite shadow refused");
    // The grown box, moved by the offset, plus half a pixel of soft ramp and
    // 3r of blur on each side. Nothing outside it or the clip may change.
    let reach = 1.0 + 3.0 * shadow_pass_radius(desc.blur_radius) as f64;
    let x0 = desc.rect.x as f64 + desc.offset_x as f64 - desc.spread as f64 - reach;
    let y0 = desc.rect.y as f64 + desc.offset_y as f64 - desc.spread as f64 - reach;
    let bw = desc.rect.width as f64 + 2.0 * desc.spread as f64 + 2.0 * reach;
    let bh = desc.rect.height as f64 + 2.0 * desc.spread as f64 + 2.0 * reach;
    for y in 0..h {
        for x in 0..w {
            let inside =
                center_inside(x, x0, bw) && center_inside(y, y0, bh) && in_clip(desc.clip, x, y);
            if !inside {
                let i = y as usize * canvas.stride + x as usize * 4;
                assert_eq!(
                    &canvas.at(x, y)[..],
                    &before[i..i + 4],
                    "shadow changed ({x}, {y}) out of reach"
                );
            }
        }
    }
    for (i, (a, b)) in canvas.buf.iter().zip(&before).enumerate() {
        if canvas.is_padding(i) {
            assert_eq!(a, b, "shadow wrote padding byte {i}");
        }
    }
}

/// A sharp opaque shadow whose box covers the whole target paints every
/// pixel exactly, at any scale of coordinates the type accepts.
fn covering_shadow_case(input: &[u8]) {
    let mut rng = case_rng(input);
    let mut canvas = Canvas::random(&mut rng);
    let (w, h) = (canvas.width, canvas.height);
    let scale = rng.pick(&[
        2.0f32,
        100.0,
        1.0e6,
        1.0e20,
        1.0e30,
        1.0e37,
        1.0e38,
        f32::MAX / 4.0,
    ]);
    let c = GpuColor::rgba(rng.byte(), rng.byte(), rng.byte(), 255);
    let rect = GpuRect::new(
        -scale,
        -scale,
        2.0 * scale + w as f32 + h as f32,
        2.0 * scale + w as f32 + h as f32,
    );
    let mut desc = ShadowDesc::new(rect, c);
    desc.corner_radius = rng.pick(&[0.0, 1.0, scale * 0.25]);
    desc.spread = rng.pick(&[0.0, 1.0, scale]);
    let result = draw_box_shadow(&mut canvas.target(), &desc);
    assert_eq!(result, Ok(()), "finite shadow refused");
    for y in 0..h {
        for x in 0..w {
            assert_eq!(
                canvas.at(x, y),
                [c.r, c.g, c.b, 255],
                "covering shadow missed ({x}, {y}) at scale {scale}, corner {}, spread {}",
                desc.corner_radius,
                desc.spread
            );
        }
    }
}

/// A shadow centered in a cleared target is the same after a left-right or
/// top-bottom mirror, give or take one step of rounding.
fn symmetric_shadow_case(input: &[u8]) {
    let mut rng = case_rng(input);
    let w = 1 + rng.below(48) as u32;
    let h = 1 + rng.below(48) as u32;
    let mut buf = vec![0u8; (w * h * 4) as usize];
    let bw = rng.below(w as usize * 2 + 1) as f32 * 0.5;
    let bh = rng.below(h as usize * 2 + 1) as f32 * 0.5;
    let rect = GpuRect::new((w as f32 - bw) * 0.5, (h as f32 - bh) * 0.5, bw, bh);
    let mut desc = ShadowDesc::new(
        rect,
        GpuColor::rgba(rng.byte(), rng.byte(), rng.byte(), 1 + rng.byte() / 2),
    );
    desc.corner_radius = rng.below(20) as f32 * 0.5;
    desc.blur_radius = rng.below(30) as f32;
    desc.spread = rng.below(9) as f32 - 4.0;
    let mut target = RgbaTarget::packed(&mut buf, w, h).expect("packed target");
    draw_box_shadow(&mut target, &desc).expect("finite shadow");
    let at = |x: u32, y: u32| {
        let i = ((y * w + x) * 4) as usize;
        [buf[i], buf[i + 1], buf[i + 2], buf[i + 3]]
    };
    for y in 0..h {
        for x in 0..w {
            let p = at(x, y);
            for q in [at(w - 1 - x, y), at(x, h - 1 - y)] {
                for k in 0..4 {
                    assert!(
                        p[k].abs_diff(q[k]) <= 1,
                        "centered shadow not symmetric at ({x}, {y}): {p:?} vs {q:?}, {desc:?}"
                    );
                }
            }
        }
    }
}

/// Planes with random padding at the end of each row.
fn padded(rng: &mut Rng, plane: &[u8], row: usize, rows: usize) -> (Vec<u8>, usize) {
    let stride = row + rng.pick(&[0, 0, 1, 2, 7, 64]);
    let mut out = Vec::with_capacity(stride * rows);
    for r in 0..rows {
        out.extend_from_slice(&plane[r * row..(r + 1) * row]);
        if r + 1 < rows || rng.one_in(2) {
            out.extend((0..stride - row).map(|_| rng.byte()));
        }
    }
    (out, stride)
}

fn blit_into(frame: &VideoFrame<'_>, w: u32, h: u32, filter: Filter) -> Vec<u8> {
    let mut buf = vec![0x5Au8; (w * h * 4) as usize];
    let mut target = RgbaTarget::packed(&mut buf, w, h).expect("packed target");
    let mut desc = BlitDesc::new(0, 0, w, h);
    desc.filter = filter;
    draw_blit(&mut target, frame, &desc).expect("valid frame");
    buf
}

/// NV12 and I420 with the same samples, packed or padded, give the same
/// pixels. At 1:1 with even sides they match `Nv12Surface::to_rgba_bt709`.
fn yuv_agreement_case(input: &[u8]) {
    let mut rng = case_rng(input);
    let w = 1 + rng.below(40) as usize;
    let h = 1 + rng.below(40) as usize;
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    let y: Vec<u8> = (0..w * h).map(|_| rng.byte()).collect();
    let u: Vec<u8> = (0..cw * ch).map(|_| rng.byte()).collect();
    let v: Vec<u8> = (0..cw * ch).map(|_| rng.byte()).collect();
    let uv: Vec<u8> = u.iter().zip(&v).flat_map(|(&a, &b)| [a, b]).collect();
    let (y_pad, y_stride) = padded(&mut rng, &y, w, h);
    let (uv_pad, uv_stride) = padded(&mut rng, &uv, cw * 2, ch);
    let (u_pad, u_stride) = padded(&mut rng, &u, cw, ch);
    let (v_pad, v_stride) = padded(&mut rng, &v, cw, ch);
    let nv12 = VideoFrame::Nv12 {
        y: &y,
        y_stride: w,
        uv: &uv,
        uv_stride: cw * 2,
        width: w as u32,
        height: h as u32,
    };
    let nv12_padded = VideoFrame::Nv12 {
        y: &y_pad,
        y_stride,
        uv: &uv_pad,
        uv_stride,
        width: w as u32,
        height: h as u32,
    };
    let i420 = VideoFrame::I420 {
        y: &y_pad,
        y_stride,
        u: &u_pad,
        u_stride,
        v: &v_pad,
        v_stride,
        width: w as u32,
        height: h as u32,
    };
    let filter = if rng.one_in(2) {
        Filter::Nearest
    } else {
        Filter::Bilinear
    };
    let (dw, dh) = if rng.one_in(2) {
        (w as u32, h as u32)
    } else {
        (1 + rng.below(50) as u32, 1 + rng.below(50) as u32)
    };
    let a = blit_into(&nv12, dw, dh, filter);
    assert_eq!(
        a,
        blit_into(&nv12_padded, dw, dh, filter),
        "padded NV12 differs ({w}x{h} -> {dw}x{dh})"
    );
    assert_eq!(
        a,
        blit_into(&i420, dw, dh, filter),
        "I420 differs from NV12 ({w}x{h} -> {dw}x{dh})"
    );
    assert!(
        a.chunks_exact(4).all(|p| p[3] == 255),
        "YUV blit left a pixel that is not opaque"
    );
    if w % 2 == 0 && h % 2 == 0 {
        let surface = Nv12Surface {
            width: w as u32,
            height: h as u32,
            y_plane: y.clone(),
            uv_plane: uv.clone(),
        };
        let exact = blit_into(&nv12, w as u32, h as u32, Filter::Nearest);
        assert_eq!(
            exact,
            surface.to_rgba_bt709(),
            "1:1 NV12 blit differs from to_rgba_bt709 ({w}x{h})"
        );
    }
}

/// A flat RGBA frame scales to the same flat color with either filter, and
/// the blit changes only its destination rectangle.
fn flat_rgba_blit_case(input: &[u8]) {
    let mut rng = case_rng(input);
    let fw = 1 + rng.below(30);
    let fh = 1 + rng.below(30);
    let any_alpha = rng.byte();
    let px = [
        rng.byte(),
        rng.byte(),
        rng.byte(),
        rng.pick(&[0, 1, 128, 255, any_alpha]),
    ];
    let stride = fw * 4 + rng.pick(&[0, 4, 5]);
    let mut frame_buf = vec![0u8; stride * (fh - 1) + fw * 4];
    for r in 0..fh {
        for c in 0..fw {
            frame_buf[r * stride + c * 4..r * stride + c * 4 + 4].copy_from_slice(&px);
        }
    }
    let frame = VideoFrame::Rgba {
        pixels: &frame_buf,
        width: fw as u32,
        height: fh as u32,
        stride,
    };
    let mut canvas = Canvas::random(&mut rng);
    let before = canvas.buf.clone();
    let (w, h) = (canvas.width, canvas.height);
    let mut desc = BlitDesc::new(
        rng.below(w as usize + 20) as i32 - 10,
        rng.below(h as usize + 20) as i32 - 10,
        rng.below(80) as u32,
        rng.below(80) as u32,
    );
    desc.filter = if rng.one_in(2) {
        Filter::Nearest
    } else {
        Filter::Bilinear
    };
    desc.clip = clip(&mut rng, w, h);
    let result = if rng.one_in(2) {
        draw_blit(&mut canvas.target(), &frame, &desc)
    } else {
        Compositor::software().blit_frame(&mut canvas.target(), &frame, &desc)
    };
    assert_eq!(result, Ok(()), "valid blit refused");
    let want = if px[3] == 0 && desc.filter == Filter::Bilinear {
        [0, 0, 0, 0]
    } else {
        px
    };
    for y in 0..h {
        for x in 0..w {
            let inside = (x as i64) >= desc.x as i64
                && (x as i64) < desc.x as i64 + desc.width as i64
                && (y as i64) >= desc.y as i64
                && (y as i64) < desc.y as i64 + desc.height as i64
                && in_clip(desc.clip, x, y);
            let got = canvas.at(x, y);
            if inside {
                assert_eq!(got, want, "flat blit at ({x}, {y}), {desc:?}");
            } else {
                let i = y as usize * canvas.stride + x as usize * 4;
                assert_eq!(
                    &got[..],
                    &before[i..i + 4],
                    "blit changed ({x}, {y}) outside its rectangle"
                );
            }
        }
    }
    for (i, (a, b)) in canvas.buf.iter().zip(&before).enumerate() {
        if canvas.is_padding(i) {
            assert_eq!(a, b, "blit wrote padding byte {i}");
        }
    }
}

#[test]
fn flat_gradients_paint_exactly_their_rectangle() {
    run_cases("gradient-area", 0x6721, seed_input, gradient_case);
}

#[test]
fn shadows_stay_within_their_reach_and_clip() {
    run_cases("shadow-reach", 0x5AD0, seed_input, shadow_reach_case);
}

#[test]
fn covering_sharp_shadows_paint_every_pixel() {
    run_cases("shadow-cover", 0xC0BE, seed_input, covering_shadow_case);
}

#[test]
fn centered_shadows_are_mirror_symmetric() {
    run_cases("shadow-mirror", 0x5E77, seed_input, symmetric_shadow_case);
}

#[test]
fn nv12_i420_padded_and_packed_frames_agree() {
    run_cases("yuv-agree", 0x0420, seed_input, yuv_agreement_case);
}

#[test]
fn flat_rgba_frames_blit_flat_and_stay_in_place() {
    run_cases("rgba-flat", 0xF1A7, seed_input, flat_rgba_blit_case);
}

/// The largest sides the crate accepts, one pixel thick, with the largest
/// blur and a frame scaled to the full cap.
#[test]
fn largest_accepted_sizes_finish() {
    for (w, h) in [(MAX_SIDE, 1), (1, MAX_SIDE)] {
        let mut buf = vec![0u8; (w * h * 4) as usize];
        let mut target = RgbaTarget::packed(&mut buf, w, h).expect("capped target");
        let mut shadow = ShadowDesc::new(
            GpuRect::new(-5.0, -5.0, w as f32 + 10.0, h as f32 + 10.0),
            GpuColor::rgba(9, 8, 7, 255),
        );
        shadow.blur_radius = f32::MAX;
        shadow.corner_radius = 1.0e30;
        draw_box_shadow(&mut target, &shadow).expect("finite shadow");
        let frame_px = [1u8, 2, 3, 255];
        let frame = VideoFrame::Rgba {
            pixels: &frame_px,
            width: 1,
            height: 1,
            stride: 4,
        };
        let mut desc = BlitDesc::new(-(MAX_SIDE as i32) / 2, 0, MAX_SIDE, MAX_SIDE);
        desc.filter = Filter::Bilinear;
        draw_blit(&mut target, &frame, &desc).expect("capped blit");
        let stops = [
            GpuGradientStop::new(0.0, GpuColor::rgba(0, 0, 0, 255)),
            GpuGradientStop::new(1.0, GpuColor::rgba(255, 255, 255, 255)),
        ];
        let gradient = GradientDesc {
            rect: GpuRect::new(0.0, 0.0, w as f32, h as f32),
            shape: GradientShape::Linear {
                start_x: -f32::MAX,
                start_y: 0.0,
                end_x: f32::MAX,
                end_y: 0.0,
            },
            stops: &stops,
            clip: None,
        };
        draw_gradient(&mut target, &gradient).expect("finite gradient");
    }
}

/// A two-stop linear gradient between far-apart endpoints still gives the
/// middle color halfway between them.
#[test]
fn far_linear_endpoints_keep_the_middle_color() {
    let stops = [
        GpuGradientStop::new(0.0, GpuColor::rgba(0, 0, 0, 255)),
        GpuGradientStop::new(1.0, GpuColor::rgba(200, 200, 200, 255)),
    ];
    for reach in [1.0e3f32, 1.0e15, 1.0e19, 1.0e20, 1.0e30, f32::MAX] {
        let mut buf = vec![0u8; 4];
        let mut target = RgbaTarget::packed(&mut buf, 1, 1).expect("one pixel");
        let desc = GradientDesc {
            rect: GpuRect::new(0.0, 0.0, 1.0, 1.0),
            shape: GradientShape::Linear {
                start_x: 0.5 - reach,
                start_y: 0.5,
                end_x: 0.5 + reach,
                end_y: 0.5,
            },
            stops: &stops,
            clip: None,
        };
        draw_gradient(&mut target, &desc).expect("finite gradient");
        assert_eq!(
            buf,
            [100, 100, 100, 255],
            "middle of a gradient {reach} pixels long"
        );
    }
}
