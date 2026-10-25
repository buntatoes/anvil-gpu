// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 Buntato
//! The device passes must give the same bytes as the CPU passes.
//!
//! Each test opens [`Compositor::open`]. When this machine has no usable
//! device (no EGL display, no OpenGL 3.3 core context, or a self-test that
//! did not match), the test prints why and passes without checking: a
//! missing device is not a failure. Otherwise every case is drawn on the
//! device and in software, and the whole target, row padding included, must
//! match byte for byte, with no fallback to software along the way.
//!
//! Cases come from a seeded xorshift generator, so a failure names a case
//! that reproduces. Each case runs on its own thread with a time limit.

use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use anvil_gpu::{
    BackendKind, BlitDesc, ComposeError, Compositor, Filter, GpuColor, GpuGradientStop, GpuRect,
    GradientDesc, GradientShape, RgbaTarget, ShadowDesc, VideoFrame, device_pass_draws,
};

/// One case that runs longer than this counts as a hang.
const CASE_LIMIT: Duration = Duration::from_secs(20);

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
        self.next_u64() % n
    }

    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (self.below(1 << 20) as f32 / (1 << 20) as f32) * (hi - lo)
    }

    fn color(&mut self) -> GpuColor {
        let v = self.next_u64().to_le_bytes();
        GpuColor::rgba(v[0], v[1], v[2], v[3])
    }

    fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| self.next_u64() as u8).collect()
    }

    fn clip(&mut self) -> Option<GpuRect> {
        (self.below(3) == 0).then(|| {
            GpuRect::new(
                self.range(-6.0, 40.0),
                self.range(-6.0, 40.0),
                self.range(0.0, 60.0),
                self.range(0.0, 60.0),
            )
        })
    }
}

/// The device compositor, or `None` (with a note) when there is none here.
fn device() -> Option<Compositor> {
    let c = Compositor::open();
    if c.active_backend() == BackendKind::Hardware {
        Some(c)
    } else {
        eprintln!(
            "skipped: no usable device here ({})",
            c.probe().reason().unwrap_or("unknown")
        );
        None
    }
}

/// A target of odd size with random pixels and row padding.
struct Target {
    pixels: Vec<u8>,
    width: u32,
    height: u32,
    stride: usize,
}

impl Target {
    fn random(rng: &mut Rng) -> Target {
        let width = 1 + rng.below(61) as u32;
        let height = 1 + rng.below(61) as u32;
        let stride = width as usize * 4 + rng.below(3) as usize * 4 + rng.below(4) as usize;
        let tail = rng.below(5) as usize;
        let mut pixels = rng.bytes(stride * height as usize + tail);
        if rng.below(2) == 0 {
            // An opaque target, the common case for a page.
            for y in 0..height as usize {
                for x in 0..width as usize {
                    pixels[y * stride + x * 4 + 3] = 255;
                }
            }
        }
        Target {
            pixels,
            width,
            height,
            stride,
        }
    }
}

/// Draw one call in software and on `gpu` over copies of `t` and compare.
fn same_on_both(
    label: &str,
    gpu: &mut Compositor,
    t: &Target,
    draw: impl Fn(&mut Compositor, &mut RgbaTarget<'_>) -> Result<(), ComposeError>,
) {
    let mut soft = t.pixels.clone();
    let mut hard = t.pixels.clone();
    let a = draw(
        &mut Compositor::software(),
        &mut RgbaTarget::new(&mut soft, t.width, t.height, t.stride).unwrap(),
    );
    let b = draw(
        gpu,
        &mut RgbaTarget::new(&mut hard, t.width, t.height, t.stride).unwrap(),
    );
    assert_eq!(a, b, "{label}: results differ");
    assert_eq!(
        gpu.fallbacks(),
        0,
        "{label}: the device fell back to software"
    );
    if soft != hard {
        let (i, (s, h)) = soft
            .iter()
            .zip(&hard)
            .enumerate()
            .find(|(_, (s, h))| s != h)
            .unwrap();
        panic!("{label}: byte {i} is {s} in software and {h} on the device");
    }
}

/// Run `body(case)` for each case on a thread with [`CASE_LIMIT`].
fn run_cases(label: &'static str, cases: u64, body: fn(&mut Compositor, u64)) {
    let Some(_) = device() else {
        return;
    };
    let before = device_pass_draws();
    for case in 0..cases {
        let (tx, rx) = mpsc::channel();
        let worker = thread::spawn(move || {
            let mut gpu = Compositor::open();
            body(&mut gpu, case);
            let _ = tx.send(());
        });
        match rx.recv_timeout(CASE_LIMIT) {
            Ok(()) => worker.join().unwrap(),
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                let _ = worker.join();
                panic!("{label}: case {case} failed");
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                panic!("{label}: case {case} did not finish within {CASE_LIMIT:?}")
            }
        }
    }
    assert!(
        device_pass_draws() > before,
        "{label}: nothing was drawn on the device"
    );
}

fn shadow_case(gpu: &mut Compositor, case: u64) {
    let mut rng = Rng::new(0x5AD0 + case);
    let t = Target::random(&mut rng);
    let mut d = ShadowDesc::new(
        GpuRect::new(
            rng.range(-20.0, 60.0),
            rng.range(-20.0, 60.0),
            rng.range(0.0, 50.0),
            rng.range(0.0, 50.0),
        ),
        rng.color(),
    );
    d.blur_radius = match rng.below(10) {
        0 => 0.0,
        1 => rng.range(60.0, 256.0),
        _ => rng.range(0.0, 30.0),
    };
    d.corner_radius = if rng.below(3) == 0 {
        0.0
    } else {
        rng.range(0.0, 16.0)
    };
    d.offset_x = rng.range(-10.0, 10.0);
    d.offset_y = rng.range(-10.0, 10.0);
    d.spread = rng.range(-6.0, 8.0);
    d.clip = rng.clip();
    same_on_both(&format!("shadow {case}: {d:?}"), gpu, &t, |c, t| {
        c.box_shadow(t, &d)
    });
}

#[test]
fn shadows_match_software() {
    run_cases("shadow", 400, shadow_case);
}

fn gradient_case(gpu: &mut Compositor, case: u64) {
    let mut rng = Rng::new(0x6AD1 + case);
    let t = Target::random(&mut rng);
    // Every stop count from 1 to 16, both shapes.
    let count = 1 + (case % 16) as usize;
    let stops: Vec<GpuGradientStop> = (0..count)
        .map(|_| GpuGradientStop::new(rng.range(-0.3, 1.3), rng.color()))
        .collect();
    let shape = if (case / 16) & 1 == 0 {
        GradientShape::Linear {
            start_x: rng.range(-20.0, 80.0),
            start_y: rng.range(-20.0, 80.0),
            end_x: rng.range(-20.0, 80.0),
            end_y: rng.range(-20.0, 80.0),
        }
    } else {
        GradientShape::Radial {
            center_x: rng.range(-10.0, 70.0),
            center_y: rng.range(-10.0, 70.0),
            radius_x: rng.range(0.0, 50.0),
            radius_y: rng.range(0.0, 50.0),
        }
    };
    let d = GradientDesc {
        rect: GpuRect::new(
            rng.range(-20.0, 60.0),
            rng.range(-20.0, 60.0),
            rng.range(0.0, 80.0),
            rng.range(0.0, 80.0),
        ),
        shape,
        stops: &stops,
        clip: rng.clip(),
    };
    same_on_both(&format!("gradient {case}: {d:?}"), gpu, &t, |c, t| {
        c.fill_gradient(t, &d)
    });
}

#[test]
fn linear_and_radial_gradients_with_1_to_16_stops_match_software() {
    run_cases("gradient", 320, gradient_case);
}

fn blit_case(gpu: &mut Compositor, case: u64) {
    let mut rng = Rng::new(0xB117 + case);
    let t = Target::random(&mut rng);
    let (w, h) = (1 + rng.below(33) as usize, 1 + rng.below(33) as usize);
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    // Some planes padded, some packed.
    let pad = |rng: &mut Rng| rng.below(2) as usize * rng.below(7) as usize;
    let (s0, s1, s2) = (
        w * 4 + pad(&mut rng),
        w + pad(&mut rng),
        cw * 2 + pad(&mut rng),
    );
    let (s3, s4) = (cw + pad(&mut rng), cw + pad(&mut rng));
    let rgba = rng.bytes(s0 * h);
    let luma = rng.bytes(s1 * h);
    let uv = rng.bytes(s2 * ch);
    let u = rng.bytes(s3 * ch);
    let v = rng.bytes(s4 * ch);
    let frame = match case % 3 {
        0 => VideoFrame::Rgba {
            pixels: &rgba,
            width: w as u32,
            height: h as u32,
            stride: s0,
        },
        1 => VideoFrame::Nv12 {
            y: &luma,
            y_stride: s1,
            uv: &uv,
            uv_stride: s2,
            width: w as u32,
            height: h as u32,
        },
        _ => VideoFrame::I420 {
            y: &luma,
            y_stride: s1,
            u: &u,
            u_stride: s3,
            v: &v,
            v_stride: s4,
            width: w as u32,
            height: h as u32,
        },
    };
    let mut d = BlitDesc::new(
        rng.below(50) as i32 - 15,
        rng.below(50) as i32 - 15,
        1 + rng.below(80) as u32,
        1 + rng.below(80) as u32,
    );
    if rng.below(5) == 0 {
        // 1:1 now and then.
        (d.width, d.height) = (w as u32, h as u32);
    }
    d.filter = if (case / 3) & 1 == 0 {
        Filter::Nearest
    } else {
        Filter::Bilinear
    };
    d.clip = rng.clip();
    same_on_both(&format!("blit {case}: {d:?}"), gpu, &t, |c, t| {
        c.blit_frame(t, &frame, &d)
    });
}

#[test]
fn rgba_nv12_and_i420_blits_with_both_filters_match_software() {
    run_cases("blit", 360, blit_case);
}
