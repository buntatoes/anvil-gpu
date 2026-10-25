// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 Buntato
//! Robustness checks for the compositor passes.
//!
//! Inputs come from a seeded generator, so a failure names a case number
//! that reproduces on every run. Each test stops after a short time budget.
//! Every case is a byte string that is read back into a target, a frame, and
//! a draw call, including bad sizes, short buffers, odd strides, and values
//! that are not finite. A call may return an error. It must not panic or
//! hang, it must not touch row padding, and a device that fails must leave
//! the same pixels as software alone. Every case also runs through
//! `Compositor::open`, which on a machine with a usable device draws on it;
//! that must give the same result and pixels as software, with no fallback.

use anvil_gpu::{
    BackendKind, BlitDesc, ComposeBackend, ComposeError, Compositor, Filter, GpuColor,
    GpuGradientStop, GpuRect, GradientDesc, GradientShape, RgbaTarget, ShadowDesc, VideoFrame,
    draw_blit, draw_box_shadow, draw_gradient,
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

// ---- reading a case back out of its bytes ----

/// Reads values from a case. Past the end it keeps going with a stream
/// seeded from the case length, so short cases still read varied values and
/// the same bytes always read the same way.
struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
    tail: u32,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Reader {
            bytes,
            at: 0,
            tail: bytes.len() as u32 ^ 0x2545_F491,
        }
    }

    fn u8(&mut self) -> u8 {
        let b = match self.bytes.get(self.at) {
            Some(&b) => b,
            None => {
                self.tail = self
                    .tail
                    .wrapping_mul(1_664_525)
                    .wrapping_add(1_013_904_223);
                (self.tail >> 24) as u8
            }
        };
        self.at += 1;
        b
    }

    /// True about once in `n` reads.
    fn one_in(&mut self, n: u8) -> bool {
        self.u8() % n < 1
    }

    fn u16(&mut self) -> u16 {
        u16::from(self.u8()) | (u16::from(self.u8()) << 8)
    }

    /// Small sizes most of the time, sometimes far too large.
    fn size(&mut self, small: u32) -> u32 {
        match self.u8() % 64 {
            0 => 0,
            1 => anvil_gpu::MAX_SIDE + u32::from(self.u8()),
            2 => u32::MAX,
            _ => u32::from(self.u8()) % (small + 1),
        }
    }

    /// Pixel values in -8..56 most of the time, plus edge cases.
    fn f32(&mut self) -> f32 {
        const ODD: [f32; 10] = [
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            -0.0,
            1e30,
            -1e30,
            1e-30,
            16384.0,
            -16384.5,
            f32::MAX,
        ];
        let tag = self.u8();
        // Zeros past the end of a case read as plain values.
        if tag % 37 == 1 {
            ODD[usize::from(tag / 37) % ODD.len()]
        } else {
            (f32::from(self.u16()) / 1024.0) - 8.0
        }
    }

    fn color(&mut self) -> GpuColor {
        GpuColor::rgba(self.u8(), self.u8(), self.u8(), self.u8())
    }

    fn rect(&mut self) -> GpuRect {
        GpuRect::new(self.f32(), self.f32(), self.f32().abs(), self.f32().abs())
    }

    fn clip(&mut self) -> Option<GpuRect> {
        if self.one_in(3) {
            let (x, y, w, h) = (self.f32(), self.f32(), self.f32(), self.f32());
            // A negative clip size is refused. Keep it rare.
            let keep = self.one_in(8);
            let fix = |v: f32| if keep { v } else { v.abs() };
            Some(GpuRect::new(x, y, fix(w), fix(h)))
        } else {
            None
        }
    }

    /// Bytes for a plane of `rows` rows, `row` bytes each, at `stride`, with
    /// a length that is sometimes short.
    fn plane(&mut self, row: usize, rows: usize) -> (Vec<u8>, usize) {
        let stride = match self.u8() % 24 {
            0 => row.saturating_sub(1),
            1 => row + usize::from(self.u8() % 9),
            _ => row,
        };
        let full = stride
            .saturating_mul(rows.saturating_sub(1))
            .saturating_add(row);
        let len = match self.u8() % 24 {
            0 => full.saturating_sub(1 + usize::from(self.u8())),
            _ => full,
        }
        .min(1 << 16);
        let seed = self.u8();
        let data = (0..len)
            .map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed))
            .collect();
        (data, stride)
    }
}

/// Bytes past the visible pixels: the tail of each row and of the buffer.
const PAD: u8 = 0x5A;

/// A target buffer with padding filled with [`PAD`]. Returns the buffer, size, and stride.
fn target_buffer(r: &mut Reader) -> (Vec<u8>, u32, u32, usize) {
    let width = r.size(40);
    let height = r.size(40);
    let row = (width.min(64) as usize) * 4;
    let stride = match r.u8() % 10 {
        0 => row.saturating_sub(4),
        1 => row + usize::from(r.u8() % 12),
        _ => row,
    };
    let rows = height.min(64) as usize;
    let short = if r.one_in(10) { 3 } else { 0 };
    let len = (stride * rows + usize::from(r.u8() % 8)).saturating_sub(short);
    let mut buf = vec![PAD; len];
    let fill = r.color();
    for y in 0..rows {
        for x in 0..(row / 4) {
            let i = y * stride + x * 4;
            if i + 4 <= len {
                buf[i..i + 4].copy_from_slice(&[fill.r, fill.g, fill.b, fill.a]);
            }
        }
    }
    (buf, width, height, stride)
}

/// Every byte outside the visible pixels must still be [`PAD`].
fn check_padding(buf: &[u8], width: u32, height: u32, stride: usize) {
    let row = (width as usize) * 4;
    if stride == 0 || height == 0 {
        assert!(
            buf.iter().all(|&b| b == PAD),
            "padding changed on an empty target"
        );
        return;
    }
    for (i, &b) in buf.iter().enumerate() {
        let (y, x) = (i / stride, i % stride);
        if y >= height as usize || x >= row {
            assert_eq!(b, PAD, "padding byte {i} changed");
        }
    }
}

enum Call<'s> {
    Shadow(ShadowDesc),
    Gradient(GradientDesc<'s>),
}

fn read_shadow(r: &mut Reader) -> ShadowDesc {
    let mut d = ShadowDesc::new(r.rect(), r.color());
    d.corner_radius = r.f32().abs();
    d.offset_x = r.f32();
    d.offset_y = r.f32();
    d.blur_radius = r.f32().abs();
    d.spread = r.f32() * 0.25;
    if r.one_in(9) {
        d.blur_radius = -d.blur_radius - 1.0;
    }
    d.clip = r.clip();
    d
}

fn read_stops(r: &mut Reader) -> Vec<GpuGradientStop> {
    let count = match r.u8() % 16 {
        0 => 0,
        1 => 17 + usize::from(r.u8() % 4),
        n => usize::from(n),
    };
    (0..count)
        .map(|_| GpuGradientStop::new(r.f32() / 32.0, r.color()))
        .collect()
}

fn read_shape(r: &mut Reader) -> GradientShape {
    if r.one_in(2) {
        GradientShape::Linear {
            start_x: r.f32(),
            start_y: r.f32(),
            end_x: r.f32() + 4.0,
            end_y: r.f32(),
        }
    } else {
        GradientShape::Radial {
            center_x: r.f32(),
            center_y: r.f32(),
            radius_x: r.f32().abs(),
            radius_y: r.f32().abs(),
        }
    }
}

/// A device that draws something else in the same area and then fails.
struct FailingDevice;

impl ComposeBackend for FailingDevice {
    fn kind(&self) -> BackendKind {
        BackendKind::Hardware
    }

    fn box_shadow(
        &mut self,
        target: &mut RgbaTarget<'_>,
        desc: &ShadowDesc,
    ) -> Result<(), ComposeError> {
        let mut wrong = *desc;
        wrong.color = GpuColor::rgba(255 - desc.color.r, 13, 200, desc.color.a.max(1));
        draw_box_shadow(target, &wrong)?;
        Err(ComposeError::Device)
    }

    fn fill_gradient(
        &mut self,
        target: &mut RgbaTarget<'_>,
        desc: &GradientDesc<'_>,
    ) -> Result<(), ComposeError> {
        let wrong = [GpuGradientStop::new(0.0, GpuColor::rgba(1, 2, 3, 255))];
        draw_gradient(
            target,
            &GradientDesc {
                stops: &wrong,
                ..*desc
            },
        )?;
        Err(ComposeError::Device)
    }

    fn blit_frame(
        &mut self,
        target: &mut RgbaTarget<'_>,
        frame: &VideoFrame<'_>,
        desc: &BlitDesc,
    ) -> Result<(), ComposeError> {
        let mut wrong = *desc;
        wrong.filter = match desc.filter {
            Filter::Nearest => Filter::Bilinear,
            Filter::Bilinear => Filter::Nearest,
        };
        draw_blit(target, frame, &wrong)?;
        Err(ComposeError::Device)
    }
}

/// Run one call through software alone, through a failing device, and
/// through `Compositor::open`, then compare. All three must agree on the
/// result and on every pixel.
fn run_both(
    input: &[u8],
    draw: impl Fn(&mut Compositor, &mut RgbaTarget<'_>) -> Result<(), ComposeError>,
) {
    let mut r = Reader::new(input);
    let (mut soft, width, height, stride) = target_buffer(&mut r);
    let mut dev = soft.clone();
    let stored = soft.clone();
    let soft_result = match RgbaTarget::new(&mut soft, width, height, stride) {
        Ok(mut t) => Some(draw(&mut Compositor::software(), &mut t)),
        Err(_) => None,
    };
    let mut c = Compositor::with_device(Box::new(FailingDevice));
    let dev_result = match RgbaTarget::new(&mut dev, width, height, stride) {
        Ok(mut t) => Some(draw(&mut c, &mut t)),
        Err(_) => None,
    };
    assert_eq!(soft_result, dev_result);
    assert_eq!(soft, dev, "fallback pixels differ from software");
    assert!(c.fallbacks() <= 1);
    // The same call on the real device backend, when there is one.
    let mut gpu = Compositor::open();
    let mut drawn = stored.clone();
    let gpu_result = match RgbaTarget::new(&mut drawn, width, height, stride) {
        Ok(mut t) => Some(draw(&mut gpu, &mut t)),
        Err(_) => None,
    };
    assert_eq!(soft_result, gpu_result);
    assert_eq!(soft, drawn, "device pixels differ from software");
    assert_eq!(gpu.fallbacks(), 0, "the device fell back to software");
    if soft_result.is_some() {
        check_padding(&soft, width, height, stride);
    }
}

fn shadow_or_gradient_case(input: &[u8]) {
    // The target is read from the front of the case and the call from the back.
    let split = input.len() / 2;
    let mut r = Reader::new(&input[split..]);
    let stops = read_stops(&mut r);
    let call = if r.one_in(2) {
        Call::Shadow(read_shadow(&mut r))
    } else {
        Call::Gradient(GradientDesc {
            rect: r.rect(),
            shape: read_shape(&mut r),
            stops: &stops,
            clip: r.clip(),
        })
    };
    run_both(&input[..split], |c, t| match &call {
        Call::Shadow(d) => c.box_shadow(t, d),
        Call::Gradient(d) => c.fill_gradient(t, d),
    });
}

fn blit_case(input: &[u8]) {
    let split = input.len() / 2;
    let mut r = Reader::new(&input[split..]);
    let width = r.size(24);
    let height = r.size(24);
    let (w, h) = (width.min(64) as usize, height.min(64) as usize);
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    let kind = r.u8() % 3;
    let (p0, s0) = r.plane(if kind == 0 { w * 4 } else { w }, h);
    let (p1, s1) = r.plane(if kind == 1 { cw * 2 } else { cw }, ch);
    let (p2, s2) = r.plane(cw, ch);
    let frame = match kind {
        0 => VideoFrame::Rgba {
            pixels: &p0,
            width,
            height,
            stride: s0,
        },
        1 => VideoFrame::Nv12 {
            y: &p0,
            y_stride: s0,
            uv: &p1,
            uv_stride: s1,
            width,
            height,
        },
        _ => VideoFrame::I420 {
            y: &p0,
            y_stride: s0,
            u: &p1,
            u_stride: s1,
            v: &p2,
            v_stride: s2,
            width,
            height,
        },
    };
    let mut desc = BlitDesc::new(
        i32::from(r.u8() as i8) / 2,
        i32::from(r.u8() as i8) / 2,
        r.size(64),
        r.size(64),
    );
    if r.one_in(10) {
        desc.x = if r.one_in(2) { i32::MAX } else { i32::MIN };
    }
    desc.filter = if r.one_in(2) {
        Filter::Nearest
    } else {
        Filter::Bilinear
    };
    desc.clip = r.clip();
    run_both(&input[..split], |c, t| c.blit_frame(t, &frame, &desc));
}

fn random_case(rng: &mut Rng) -> Vec<u8> {
    let len = 32 + rng.below(320);
    (0..len).map(|_| rng.byte()).collect()
}

#[test]
fn random_shadows_and_gradients_do_not_panic_and_fall_back_cleanly() {
    run_cases(
        "shadow-gradient",
        0x5AD0,
        random_case,
        shadow_or_gradient_case,
    );
}

#[test]
fn random_video_blits_do_not_panic_and_fall_back_cleanly() {
    run_cases("blit", 0xB117, random_case, blit_case);
}

/// A case that draws on a 20 by 20 target, for the mutated runs to start from.
fn seed_case(rng: &mut Rng) -> Vec<u8> {
    let mut case = random_case(rng);
    // Width and height tags and values: plain sizes of 20.
    case[..4].copy_from_slice(&[5, 20, 5, 20]);
    case
}

#[test]
fn mutated_cases_do_not_panic_and_fall_back_cleanly() {
    let mut base = Rng::new(0xC0DE);
    let seeds: Vec<Vec<u8>> = (0..8).map(|_| seed_case(&mut base)).collect();
    let mut pick = 0usize;
    run_cases(
        "mutated",
        0x3A7E,
        move |rng| {
            pick += 1;
            mutate(rng, &seeds[pick % seeds.len()])
        },
        |input| {
            shadow_or_gradient_case(input);
            blit_case(input);
        },
    );
}

#[test]
fn large_blur_on_a_large_target_stays_fast() {
    // The biggest blur, over a 512 by 512 target, must finish well inside the
    // case limit even in a debug build.
    let started = Instant::now();
    let mut buf = vec![255u8; 512 * 512 * 4];
    let mut t = RgbaTarget::packed(&mut buf, 512, 512).unwrap();
    let mut d = ShadowDesc::new(
        GpuRect::new(-100.0, -100.0, 700.0, 700.0),
        GpuColor::rgba(0, 0, 0, 255),
    );
    d.blur_radius = 1e9;
    Compositor::software().box_shadow(&mut t, &d).unwrap();
    assert!(
        started.elapsed() < CASE_LIMIT,
        "took {:?}",
        started.elapsed()
    );
    // Deep inside the box the shadow is solid.
    assert_eq!(t.pixel(256, 256), Some([0, 0, 0, 255]));
}

#[test]
fn large_blur_on_the_device_matches_and_stays_fast() {
    // The same call through Compositor::open, which uses the device when
    // there is one. It must finish inside the case limit and match software.
    let draw = |c: &mut Compositor| {
        let mut buf: Vec<u8> = (0..512 * 512 * 4).map(|i| (i % 251) as u8).collect();
        let mut t = RgbaTarget::packed(&mut buf, 512, 512).unwrap();
        let mut d = ShadowDesc::new(
            GpuRect::new(-100.0, -100.0, 700.0, 700.0),
            GpuColor::rgba(30, 60, 90, 200),
        );
        d.blur_radius = 1e9;
        d.corner_radius = 40.0;
        c.box_shadow(&mut t, &d).unwrap();
        buf
    };
    let started = Instant::now();
    let mut gpu = Compositor::open();
    let got = draw(&mut gpu);
    assert!(
        started.elapsed() < CASE_LIMIT,
        "took {:?}",
        started.elapsed()
    );
    assert_eq!(gpu.fallbacks(), 0);
    assert!(got == draw(&mut Compositor::software()));
}
