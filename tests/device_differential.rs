// SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0-OR-Commercial
// Copyright (c) 2026 Buntato
//! Seeded scenes drawn on the device and on the CPU must give the same bytes.
//!
//! `tests/gpu_equality.rs` checks a fixed number of single calls. These run
//! until the time budget is spent (`ANVIL_ROBUSTNESS_MS`, 1.5 seconds by
//! default) and draw whole scenes: two to eight calls in a row on one target,
//! mixing shadows, gradients, and blits, with odd sizes up to 301 by 211,
//! padded strides, blurs up to the cap, clips that miss or cut, coordinates
//! far outside the target, and calls too big for the device, which must be
//! drawn on the CPU with the same result. Broken inputs (NaN, infinity,
//! frames with short planes) must give the same error both ways and leave the
//! target alone.
//!
//! Every byte must match, bilinear blits included: their weights and sums
//! are integers on both sides.
//!
//! Each test has a fixed seed. Set `ANVIL_SEED` to add a number to every
//! seed for a run over different scenes, and `ANVIL_FIRST_CASE` to replay a
//! reported case.
//!
//! With no usable device the tests write `SKIPPED` and the reason straight
//! to stderr, so it shows without `--nocapture`, and pass. Set
//! `ANVIL_REQUIRE_DEVICE=1` to make a missing device a failure, for machines
//! that are known to have one.

use std::io::Write;
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use anvil_gpu::{
    BackendKind, BlitDesc, ComposeError, Compositor, Filter, GpuColor, GpuGradientStop, GpuRect,
    GradientDesc, GradientShape, RgbaTarget, ShadowDesc, VideoFrame, device_pass_draws,
};

fn test_budget() -> Duration {
    let millis = std::env::var("ANVIL_ROBUSTNESS_MS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(1500);
    Duration::from_millis(millis)
}

/// One scene that runs longer than this counts as a hang.
const CASE_LIMIT: Duration = Duration::from_secs(30);

const MAX_CASES: u64 = 200_000;

/// Added to every seed, from `ANVIL_SEED`.
fn seed_offset() -> u64 {
    std::env::var("ANVIL_SEED")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
}

struct Rng(u64);

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

    fn below(&mut self, n: u64) -> u64 {
        if n == 0 { 0 } else { self.next_u64() % n }
    }

    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (self.below(1 << 20) as f32 / (1 << 20) as f32) * (hi - lo)
    }

    fn color(&mut self) -> GpuColor {
        let v = self.next_u64().to_le_bytes();
        let a = match self.below(4) {
            0 => 255,
            1 => 0,
            _ => v[3],
        };
        GpuColor::rgba(v[0], v[1], v[2], a)
    }

    fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| self.next_u64() as u8).collect()
    }

    /// A coordinate: usually near the target, sometimes far off it.
    fn coord(&mut self, span: f32) -> f32 {
        match self.below(12) {
            0 => self.range(-1e6, 1e6),
            1 => [-3e15, 3e15, 1e30, -1e30][self.below(4) as usize],
            _ => self.range(-0.3 * span, 1.3 * span),
        }
    }

    fn clip(&mut self, w: f32, h: f32) -> Option<GpuRect> {
        match self.below(4) {
            0 => Some(GpuRect::new(
                self.range(-10.0, w),
                self.range(-10.0, h),
                self.range(0.0, w),
                self.range(0.0, h),
            )),
            // A clip that misses the target.
            1 if self.below(4) == 0 => Some(GpuRect::new(w + 5.0, 0.0, 10.0, 10.0)),
            _ => None,
        }
    }
}

/// `Some(device compositor)`, or `None` after saying why there is none.
fn device_or_skip(label: &str) -> Option<Compositor> {
    let c = Compositor::open();
    if c.active_backend() == BackendKind::Hardware {
        return Some(c);
    }
    let why = c.probe().reason().unwrap_or("unknown").to_string();
    if std::env::var("ANVIL_REQUIRE_DEVICE").is_ok_and(|v| v == "1") {
        panic!("{label}: ANVIL_REQUIRE_DEVICE=1 but there is no usable device: {why}");
    }
    // The test harness only captures print! and eprint!, so this shows.
    let _ = writeln!(
        std::io::stderr().lock(),
        "SKIPPED {label}: no usable device here ({why}); nothing was compared"
    );
    None
}

/// One draw call in a scene, with the data it borrows.
enum Call {
    Shadow(ShadowDesc),
    Gradient {
        rect: GpuRect,
        shape: GradientShape,
        stops: Vec<GpuGradientStop>,
        clip: Option<GpuRect>,
    },
    Blit {
        planes: Vec<Vec<u8>>,
        strides: Vec<usize>,
        kind: u8,
        width: u32,
        height: u32,
        desc: BlitDesc,
    },
}

impl Call {
    /// Short description for failure messages.
    fn describe(&self) -> String {
        match self {
            Call::Shadow(d) => format!("shadow {d:?}"),
            Call::Gradient {
                rect,
                shape,
                stops,
                clip,
            } => {
                format!(
                    "gradient {rect:?} {shape:?} {} stops, clip {clip:?}",
                    stops.len()
                )
            }
            Call::Blit {
                strides,
                kind,
                width,
                height,
                desc,
                ..
            } => format!("blit kind {kind} frame {width}x{height} strides {strides:?} {desc:?}"),
        }
    }

    fn draw(&self, c: &mut Compositor, t: &mut RgbaTarget<'_>) -> Result<(), ComposeError> {
        match self {
            Call::Shadow(d) => c.box_shadow(t, d),
            Call::Gradient {
                rect,
                shape,
                stops,
                clip,
            } => c.fill_gradient(
                t,
                &GradientDesc {
                    rect: *rect,
                    shape: *shape,
                    stops,
                    clip: *clip,
                },
            ),
            Call::Blit {
                planes,
                strides,
                kind,
                width,
                height,
                desc,
            } => {
                let frame = match kind {
                    0 => VideoFrame::Rgba {
                        pixels: &planes[0],
                        width: *width,
                        height: *height,
                        stride: strides[0],
                    },
                    1 => VideoFrame::Nv12 {
                        y: &planes[0],
                        y_stride: strides[0],
                        uv: &planes[1],
                        uv_stride: strides[1],
                        width: *width,
                        height: *height,
                    },
                    _ => VideoFrame::I420 {
                        y: &planes[0],
                        y_stride: strides[0],
                        u: &planes[1],
                        u_stride: strides[1],
                        v: &planes[2],
                        v_stride: strides[2],
                        width: *width,
                        height: *height,
                    },
                };
                c.blit_frame(t, &frame, desc)
            }
        }
    }
}

fn shadow(rng: &mut Rng, w: f32, h: f32, broken: bool) -> Call {
    let mut d = ShadowDesc::new(
        GpuRect::new(
            rng.coord(w),
            rng.coord(h),
            rng.range(0.0, w),
            rng.range(0.0, h),
        ),
        rng.color(),
    );
    d.blur_radius = match rng.below(8) {
        0 => 0.0,
        1 => rng.range(60.0, 400.0),
        _ => rng.range(0.0, 30.0),
    };
    d.corner_radius = if rng.below(3) == 0 {
        0.0
    } else {
        rng.range(0.0, 40.0)
    };
    d.offset_x = rng.range(-20.0, 20.0);
    d.offset_y = rng.range(-20.0, 20.0);
    d.spread = rng.range(-10.0, 12.0);
    d.clip = rng.clip(w, h);
    if broken {
        match rng.below(4) {
            0 => d.blur_radius = f32::NAN,
            1 => d.offset_x = f32::INFINITY,
            2 => d.spread = f32::NAN,
            _ => d.rect = GpuRect::new(f32::NAN, 0.0, 10.0, 10.0),
        }
    }
    Call::Shadow(d)
}

fn gradient(rng: &mut Rng, w: f32, h: f32, broken: bool) -> Call {
    let count = 1 + rng.below(16) as usize;
    let mut stops: Vec<GpuGradientStop> = (0..count)
        .map(|_| GpuGradientStop::new(rng.range(-0.3, 1.3), rng.color()))
        .collect();
    let shape = if rng.below(2) == 0 {
        GradientShape::Linear {
            start_x: rng.coord(w),
            start_y: rng.coord(h),
            end_x: rng.coord(w),
            end_y: rng.coord(h),
        }
    } else {
        GradientShape::Radial {
            center_x: rng.coord(w),
            center_y: rng.coord(h),
            radius_x: rng.range(0.0, w),
            radius_y: rng.range(0.0, h),
        }
    };
    if broken {
        stops[0] = GpuGradientStop::new(f32::NAN, rng.color());
    }
    Call::Gradient {
        rect: GpuRect::new(
            rng.range(-0.3 * w, w),
            rng.range(-0.3 * h, h),
            rng.range(0.0, 1.3 * w),
            rng.range(0.0, 1.3 * h),
        ),
        shape,
        stops,
        clip: rng.clip(w, h),
    }
}

fn blit(rng: &mut Rng, w: f32, h: f32, broken: bool) -> Call {
    let (fw, fh) = (1 + rng.below(40) as usize, 1 + rng.below(40) as usize);
    let (cw, ch) = (fw.div_ceil(2), fh.div_ceil(2));
    let pad = |rng: &mut Rng| rng.below(2) as usize * rng.below(9) as usize;
    let kind = rng.below(3) as u8;
    let (rows, widths): (Vec<usize>, Vec<usize>) = match kind {
        0 => (vec![fh], vec![fw * 4]),
        1 => (vec![fh, ch], vec![fw, cw * 2]),
        _ => (vec![fh, ch, ch], vec![fw, cw, cw]),
    };
    let strides: Vec<usize> = widths.iter().map(|&row| row + pad(rng)).collect();
    let mut planes: Vec<Vec<u8>> = strides
        .iter()
        .zip(&rows)
        .map(|(&s, &r)| rng.bytes(s * r))
        .collect();
    if broken {
        // A plane one byte short.
        let p = rng.below(planes.len() as u64) as usize;
        planes[p].pop();
    }
    let mut desc = BlitDesc::new(
        rng.range(-0.3 * w, w) as i32,
        rng.range(-0.3 * h, h) as i32,
        1 + rng.below(2 * w as u64 + 1) as u32,
        1 + rng.below(2 * h as u64 + 1) as u32,
    );
    if rng.below(5) == 0 {
        (desc.width, desc.height) = (fw as u32, fh as u32);
    }
    desc.filter = if rng.below(2) == 0 {
        Filter::Nearest
    } else {
        Filter::Bilinear
    };
    desc.clip = rng.clip(w, h);
    Call::Blit {
        planes,
        strides,
        kind,
        width: fw as u32,
        height: fh as u32,
        desc,
    }
}

/// Draw one random scene both ways and compare every byte.
fn scene(case: u64, seed: u64, make: fn(&mut Rng, f32, f32, bool) -> Call, mixed: bool) {
    let mut rng = Rng::new(seed.wrapping_add(case.wrapping_mul(0x9E37_79B9)));
    let width = 1 + rng.below(301) as u32;
    let height = 1 + rng.below(211) as u32;
    let stride = width as usize * 4 + rng.below(3) as usize * 4 + rng.below(4) as usize;
    let tail = rng.below(5) as usize;
    let mut base = rng.bytes(stride * height as usize + tail);
    if rng.below(2) == 0 {
        for y in 0..height as usize {
            for x in 0..width as usize {
                base[y * stride + x * 4 + 3] = 255;
            }
        }
    }
    let (w, h) = (width as f32, height as f32);
    let makers = [shadow, gradient, blit];
    let calls: Vec<Call> = (0..2 + rng.below(7))
        .map(|_| {
            let broken = rng.below(10) == 0;
            let f = if mixed {
                makers[rng.below(3) as usize]
            } else {
                make
            };
            f(&mut rng, w, h, broken)
        })
        .collect();
    let mut soft = base.clone();
    let mut hard = base;
    let mut cpu = Compositor::software();
    let mut gpu = Compositor::open();
    for (n, call) in calls.iter().enumerate() {
        let before_soft = soft.clone();
        let a = call.draw(
            &mut cpu,
            &mut RgbaTarget::new(&mut soft, width, height, stride).unwrap(),
        );
        let b = call.draw(
            &mut gpu,
            &mut RgbaTarget::new(&mut hard, width, height, stride).unwrap(),
        );
        assert_eq!(a, b, "case {case}, call {n}: results differ");
        if a.is_err() {
            assert_eq!(
                soft, before_soft,
                "case {case}, call {n}: an error changed pixels"
            );
        }
        if let Some(i) = (0..soft.len()).find(|&i| soft[i] != hard[i]) {
            panic!(
                "case {case}, call {n} ({width}x{height}, stride {stride}): byte {i} (x {}, y {}, channel {}) is {} on the CPU and {} on the device; {}",
                (i % stride) / 4,
                i / stride,
                (i % stride) % 4,
                soft[i],
                hard[i],
                call.describe()
            );
        }
        // Start the next call from the same pixels on both sides.
        hard.copy_from_slice(&soft);
    }
    assert_eq!(
        gpu.fallbacks(),
        0,
        "case {case}: the device fell back to the CPU"
    );
}

fn run(label: &'static str, seed: u64, make: fn(&mut Rng, f32, f32, bool) -> Call, mixed: bool) {
    if device_or_skip(label).is_none() {
        return;
    }
    let seed = seed.wrapping_add(seed_offset());
    let before = device_pass_draws();
    let budget = test_budget();
    let started = Instant::now();
    // `ANVIL_FIRST_CASE` replays a reported case without running the ones before it.
    let mut case = std::env::var("ANVIL_FIRST_CASE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let first = case;
    while case < MAX_CASES && started.elapsed() < budget {
        let (tx, rx) = mpsc::channel();
        let worker = thread::spawn(move || {
            scene(case, seed, make, mixed);
            let _ = tx.send(());
        });
        match rx.recv_timeout(CASE_LIMIT) {
            Ok(()) => worker.join().unwrap(),
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                let _ = worker.join();
                panic!("{label}: case {case} failed (seed {seed:#x})");
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                panic!("{label}: case {case} did not finish within {CASE_LIMIT:?}")
            }
        }
        case += 1;
    }
    let draws = device_pass_draws() - before;
    assert!(draws > 0, "{label}: nothing was drawn on the device");
    eprintln!(
        "{label}: {} scenes in {:?} (seed {seed:#x}); device pass draws during it (all tests): {draws}",
        case - first,
        started.elapsed(),
    );
}

#[test]
fn shadow_scenes_match_the_cpu() {
    run("shadow scenes", 0x5ade_0001, shadow, false);
}

#[test]
fn gradient_scenes_match_the_cpu() {
    run("gradient scenes", 0x6ade_0002, gradient, false);
}

#[test]
fn blit_scenes_match_the_cpu() {
    run("blit scenes", 0xb11e_0003, blit, false);
}

#[test]
fn mixed_scenes_match_the_cpu() {
    run("mixed scenes", 0x313e_0004, shadow, true);
}

/// Seed 0xb11e0003, case 211: a bilinear I420 blit (an 18 by 9 frame
/// scaled to 425 by 88 at (184, -3)) whose float sum came out at exactly
/// 54.5 on the CPU and just under it on the device, one red code apart.
/// With integer weights it must match.
#[test]
fn the_bilinear_i420_scene_that_was_one_code_apart_matches() {
    if device_or_skip("blit scene 211").is_none() {
        return;
    }
    scene(211, 0xb11e_0003, blit, false);
}

/// Seed 0x313e138c, case 27: call 6 paints a shadow whose blue channel came
/// out at 0.5 - 2^-25 before rounding. The CPU rounded that to 0 and the
/// device's floor(x + 0.5) to 1. Both now round the same way.
#[test]
fn the_shadow_paint_that_rounded_differently_matches() {
    if device_or_skip("mixed scene 27").is_none() {
        return;
    }
    scene(27, 0x313e_138c, shadow, true);
}

/// Seed 0x6ade0002 + 1234, case 83: call 5 fills a gradient whose blue mix at
/// (150, 117) came out at 26.499998 on the CPU and rounded to 27 before shader
/// division and square root were refined to 0.5 ULP.
#[test]
fn the_gradient_scene_that_was_one_code_apart_matches() {
    if device_or_skip("gradient scene 83").is_none() {
        return;
    }
    scene(83, 0x6ade_0002u64.wrapping_add(1234), gradient, false);
}
