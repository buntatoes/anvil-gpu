// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 Buntato
//! The compositor passes on OpenGL 3.3 core through EGL (Linux).
//!
//! [`GlDevice::open`] makes an offscreen context, builds one program per
//! pass from [`crate::PASSES_GLSL`], and draws the self-test scene. Only a
//! device that passes all three steps is used. [`shared`] keeps one such
//! device per process for [`crate::Compositor::open`].

use std::sync::mpsc;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use crate::compose::ComposeError;
use crate::gpu::{
    self, Faults, Pass, PassDevice, PassParams, SharedDevice, Slot, TexFormat, TexId,
};
use crate::sys_gl::{ContextRequest, GlContext, GlCurrent, GlProgram, GlTexture, Uniform};

/// How long opening a device may take before it counts as unavailable.
const OPEN_LIMIT: Duration = Duration::from_secs(20);

/// The step at which opening a GL device stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GlStop {
    /// No EGL display, no GL 3.3 core context, or a missing entry point.
    Context,
    /// A pass did not compile or link.
    Compile,
    /// The self-test scene did not match the CPU bytes.
    SelfTest,
    /// Opening took longer than [`OPEN_LIMIT`].
    TimedOut,
}

/// An OpenGL context with every pass built.
#[derive(Debug)]
pub(crate) struct GlDevice {
    context: GlContext,
    programs: Vec<GlProgram>,
}

/// The vertex and fragment source for `pass`.
fn sources(pass: Pass, faults: Faults) -> (String, String) {
    const VERSION: &str = "#version 330 core\n";
    // Comments may come before #version, but nothing else may; keep only
    // what follows its line and put the defines right after it. The line
    // may end in CRLF if the file was checked out that way.
    let body = crate::PASSES_GLSL
        .split_once(VERSION.trim_end())
        .and_then(|(_, rest)| rest.split_once('\n'))
        .map_or(crate::PASSES_GLSL, |(_, body)| body);
    let vertex = format!("{VERSION}#define VERTEX_SHADER\n{body}");
    let mut fragment = format!(
        "{VERSION}#define FRAGMENT_SHADER\n#define {}\n{body}",
        pass.glsl_symbol()
    );
    if faults.break_shaders {
        fragment.push_str("\nthis line does not compile\n");
    }
    (vertex, fragment)
}

impl GlDevice {
    /// Make a context, build the passes, and run the self-test.
    pub(crate) fn open(faults: Faults) -> Result<GlDevice, GlStop> {
        let request = if faults.bad_context {
            ContextRequest { major: 9, minor: 9 }
        } else {
            ContextRequest::GL33
        };
        let mut context = GlContext::create(request).map_err(|_| GlStop::Context)?;
        let programs = context
            .with_current(|gl| {
                Pass::ALL
                    .iter()
                    .map(|&pass| {
                        let (vs, fs) = sources(pass, faults);
                        gl.program(&vs, &fs)
                    })
                    .collect::<Result<Vec<_>, String>>()
            })
            .map_err(|_| GlStop::Context)?
            .map_err(|_| GlStop::Compile)?;
        let mut device = GlDevice { context, programs };
        let matched = device
            .with_passes(gpu::self_test)
            .map_err(|_| GlStop::SelfTest)?;
        if !matched {
            return Err(GlStop::SelfTest);
        }
        Ok(device)
    }

    /// [`GlDevice::open`] on a helper thread, giving up after `limit`.
    ///
    /// A driver that hangs while making a context or compiling must not
    /// hang the caller. The helper thread is left behind in that case.
    pub(crate) fn open_with_limit(faults: Faults, limit: Duration) -> Result<GlDevice, GlStop> {
        let (tx, rx) = mpsc::channel();
        let spawned = std::thread::Builder::new()
            .name("anvil-gpu-gl-open".into())
            .spawn(move || {
                let _ = tx.send(GlDevice::open(faults));
            });
        if spawned.is_err() {
            return Err(GlStop::Context);
        }
        match rx.recv_timeout(limit) {
            Ok(result) => result,
            Err(mpsc::RecvTimeoutError::Timeout) => Err(GlStop::TimedOut),
            Err(mpsc::RecvTimeoutError::Disconnected) => Err(GlStop::Context),
        }
    }
}

/// The process-wide GL device, opened once.
pub(crate) fn shared() -> Result<Arc<Mutex<GlDevice>>, GlStop> {
    static SHARED: OnceLock<Result<Arc<Mutex<GlDevice>>, GlStop>> = OnceLock::new();
    SHARED
        .get_or_init(|| {
            GlDevice::open_with_limit(Faults::default(), OPEN_LIMIT)
                .map(|d| Arc::new(Mutex::new(d)))
        })
        .clone()
}

/// A [`PassDevice`] over a current context.
struct GlPasses<'a, 'c> {
    gl: &'a mut GlCurrent<'c>,
    programs: &'a [GlProgram],
    textures: Vec<GlTexture>,
}

fn device_error<T>(r: Result<T, crate::sys_gl::GlError>) -> Result<T, ComposeError> {
    r.map_err(|_| ComposeError::Device)
}

impl GlPasses<'_, '_> {
    fn add(&mut self, t: GlTexture) -> TexId {
        self.textures.push(t);
        TexId(self.textures.len() - 1)
    }

    fn get(&self, t: TexId) -> Result<GlTexture, ComposeError> {
        self.textures.get(t.0).copied().ok_or(ComposeError::Device)
    }
}

impl PassDevice for GlPasses<'_, '_> {
    fn max_side(&self) -> u32 {
        self.gl.max_side()
    }

    fn upload(
        &mut self,
        format: TexFormat,
        width: u32,
        height: u32,
        data: &[u8],
    ) -> Result<TexId, ComposeError> {
        let t = device_error(self.gl.texture(format, width, height, Some(data)))?;
        Ok(self.add(t))
    }

    fn render_target(
        &mut self,
        format: TexFormat,
        width: u32,
        height: u32,
    ) -> Result<TexId, ComposeError> {
        let t = device_error(self.gl.texture(format, width, height, None))?;
        Ok(self.add(t))
    }

    fn draw(
        &mut self,
        pass: Pass,
        out: TexId,
        inputs: &[(Slot, TexId)],
        p: &PassParams,
    ) -> Result<(), ComposeError> {
        let out = self.get(out)?;
        let bound = inputs
            .iter()
            .map(|&(slot, t)| Ok((slot.glsl_name(), self.get(t)?)))
            .collect::<Result<Vec<_>, ComposeError>>()?;
        let program = *self
            .programs
            .get(pass.index())
            .ok_or(ComposeError::Device)?;
        let uniforms = [
            (c"uOrigin", Uniform::I2(p.origin)),
            (c"uShape", Uniform::F4(p.shape)),
            (c"uCorner", Uniform::F1(p.corner)),
            (c"uAxis", Uniform::I2(p.axis)),
            (c"uRadius", Uniform::I1(p.radius)),
            (c"uColor", Uniform::F4(p.shadow_color)),
            (c"uMargin", Uniform::I1(p.margin)),
            (c"uKind", Uniform::I1(p.kind)),
            (c"uGeometry", Uniform::F4(p.geometry)),
            (c"uStopCount", Uniform::I1(p.stop_count)),
            (c"uStopOffsets", Uniform::F1v(&p.stop_offsets)),
            (c"uStopColors", Uniform::F4v(&p.stop_colors)),
            (c"uDstRect", Uniform::I4(p.dst_rect)),
            (c"uFilter", Uniform::I1(p.filter)),
        ];
        let size = self
            .gl
            .texture_size(out)
            .map_err(|_| ComposeError::Device)?;
        let region = [
            (
                c"uRegion",
                Uniform::F4([0.0, 0.0, size.0 as f32, size.1 as f32]),
            ),
            (
                c"uViewportSize",
                Uniform::F2([size.0 as f32, size.1 as f32]),
            ),
        ];
        let all: Vec<_> = region.iter().chain(uniforms.iter()).copied().collect();
        device_error(self.gl.draw(program, out, &bound, &all))
    }

    fn read_rgba(&mut self, tex: TexId, out: &mut [u8]) -> Result<(), ComposeError> {
        let t = self.get(tex)?;
        device_error(self.gl.read_rgba(t, out))
    }
}

impl SharedDevice for GlDevice {
    fn with_passes<R>(
        &mut self,
        f: impl FnOnce(&mut dyn PassDevice) -> R,
    ) -> Result<R, ComposeError> {
        let programs = &self.programs;
        self.context
            .with_current(|gl| {
                let mut passes = GlPasses {
                    gl,
                    programs,
                    textures: Vec::new(),
                };
                f(&mut passes)
            })
            .map_err(|_| ComposeError::Device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::DeviceBackend;
    use crate::gpu::testing::{draw_scene, within_limit};
    use crate::{
        BackendKind, Compositor, GpuColor, GpuGradientStop, GpuRect, GradientDesc, GradientShape,
        RgbaTarget, device_pass_draws,
    };

    /// True when this machine has a working GL device. Prints why not.
    fn have_gl() -> bool {
        match shared() {
            Ok(_) => true,
            Err(stop) => {
                eprintln!("skipped: no usable OpenGL device here ({stop:?})");
                false
            }
        }
    }

    /// What `Compositor::open` does, with faults put in.
    fn compositor_with(faults: Faults) -> (Compositor, Result<(), GlStop>) {
        match GlDevice::open_with_limit(faults, OPEN_LIMIT) {
            Ok(d) => (
                Compositor::with_device(Box::new(DeviceBackend::new(
                    Arc::new(Mutex::new(d)),
                    faults,
                ))),
                Ok(()),
            ),
            Err(stop) => (Compositor::software(), Err(stop)),
        }
    }

    #[test]
    fn a_machine_with_a_gl_context_compiles_and_matches() {
        within_limit(|| match shared() {
            Ok(_) => {}
            // No device here: nothing to check.
            Err(stop @ (GlStop::Context | GlStop::TimedOut)) => {
                eprintln!("skipped: no usable OpenGL device here ({stop:?})");
            }
            // A context was made, so the passes themselves are at fault.
            Err(stop) => panic!("the GL passes failed on a working context: {stop:?}"),
        });
    }

    #[test]
    fn the_gl_device_draws_the_scene_like_software() {
        within_limit(|| {
            if !have_gl() {
                return;
            }
            let want = draw_scene(&mut Compositor::software());
            let before = device_pass_draws();
            let mut c = Compositor::open();
            assert_eq!(c.active_backend(), BackendKind::Hardware);
            assert_eq!(draw_scene(&mut c), want);
            assert_eq!(c.fallbacks(), 0);
            assert_eq!(c.active_backend(), BackendKind::Hardware);
            // Eight draws for the shadow, one per gradient and blit.
            assert!(device_pass_draws() >= before + 13);
        });
    }

    #[test]
    fn shaders_that_do_not_compile_leave_software() {
        within_limit(|| {
            if !have_gl() {
                return;
            }
            let faults = Faults {
                break_shaders: true,
                ..Faults::default()
            };
            let (mut c, opened) = compositor_with(faults);
            assert_eq!(opened, Err(GlStop::Compile));
            assert_eq!(c.active_backend(), BackendKind::Software);
            assert_eq!(draw_scene(&mut c), draw_scene(&mut Compositor::software()));
        });
    }

    #[test]
    fn a_context_that_cannot_be_made_leaves_software() {
        within_limit(|| {
            if !have_gl() {
                return;
            }
            let faults = Faults {
                bad_context: true,
                ..Faults::default()
            };
            let (mut c, opened) = compositor_with(faults);
            assert_eq!(opened, Err(GlStop::Context));
            assert_eq!(c.active_backend(), BackendKind::Software);
            assert_eq!(draw_scene(&mut c), draw_scene(&mut Compositor::software()));
        });
    }

    #[test]
    fn a_draw_that_fails_midway_is_redrawn_in_software() {
        within_limit(|| {
            if !have_gl() {
                return;
            }
            let want = draw_scene(&mut Compositor::software());
            // The scene starts with a blurred shadow: draw 1 is the mask, 2 to 7
            // the blur passes, 8 the paint.
            for fail_draw in 1..=8 {
                let faults = Faults {
                    fail_draw: Some(fail_draw),
                    ..Faults::default()
                };
                let (mut c, opened) = compositor_with(faults);
                assert_eq!(opened, Ok(()));
                assert_eq!(c.active_backend(), BackendKind::Hardware);
                assert_eq!(draw_scene(&mut c), want, "failing draw {fail_draw}");
                assert_eq!(c.fallbacks(), 1);
                assert_eq!(c.active_backend(), BackendKind::Software);
            }
        });
    }

    #[test]
    fn calls_from_many_threads_share_the_device_safely() {
        within_limit(|| {
            if !have_gl() {
                return;
            }
            let want = draw_scene(&mut Compositor::software());
            let workers: Vec<_> = (0..4)
                .map(|_| {
                    std::thread::spawn(|| {
                        let mut c = Compositor::open();
                        let got: Vec<Vec<u8>> = (0..3).map(|_| draw_scene(&mut c)).collect();
                        (got, c.fallbacks())
                    })
                })
                .collect();
            for w in workers {
                let (got, fallbacks) = w.join().unwrap();
                assert_eq!(fallbacks, 0);
                for g in got {
                    assert_eq!(g, want);
                }
            }
        });
    }

    #[test]
    fn a_call_past_the_device_limits_is_drawn_on_the_cpu() {
        within_limit(|| {
            if !have_gl() {
                return;
            }
            // 4200 by 4100 is more texels than one device texture may hold.
            let (w, h) = (4200u32, 4100u32);
            let stops = [
                GpuGradientStop::new(0.0, GpuColor::rgba(0, 0, 0, 255)),
                GpuGradientStop::new(1.0, GpuColor::rgba(255, 255, 255, 255)),
            ];
            let desc = GradientDesc {
                rect: GpuRect::new(0.0, 0.0, w as f32, h as f32),
                shape: GradientShape::Linear {
                    start_x: 0.0,
                    start_y: 0.0,
                    end_x: w as f32,
                    end_y: h as f32,
                },
                stops: &stops,
                clip: None,
            };
            let mut want = vec![9u8; (w * h * 4) as usize];
            let mut got = want.clone();
            Compositor::software()
                .fill_gradient(&mut RgbaTarget::packed(&mut want, w, h).unwrap(), &desc)
                .unwrap();
            let mut c = Compositor::open();
            c.fill_gradient(&mut RgbaTarget::packed(&mut got, w, h).unwrap(), &desc)
                .unwrap();
            assert!(got == want);
            assert_eq!(c.fallbacks(), 0);
            assert_eq!(c.active_backend(), BackendKind::Hardware);
        });
    }
}
