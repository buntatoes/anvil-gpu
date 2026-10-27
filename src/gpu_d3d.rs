// SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0-OR-Commercial
// Copyright (c) 2026 Buntato
//! The compositor passes on Direct3D 11 (Windows).
//!
//! [`D3dPassDevice::open`] makes a device (hardware, else WARP), compiles
//! every entry point of [`crate::PASSES_HLSL`] with IEEE strictness, and
//! draws the self-test scene. Only a device that passes all three steps is
//! used. [`shared`] keeps one such device per process for
//! [`crate::Compositor::open`].
//!
//! The self-test keeps a wrong result from being used on any driver whose
//! rounding differs from the CPU: a device that does not match the software
//! bytes leaves the compositor on software.

use std::sync::mpsc;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use crate::compose::ComposeError;
use crate::gpu::{
    self, Faults, Pass, PassDevice, PassParams, SharedDevice, Slot, TexFormat, TexId,
};
use crate::sys_d3d::{self, D3dDevice, D3dTexture, PixelShader, VertexShader};

/// How long opening a device may take before it counts as unavailable.
const OPEN_LIMIT: Duration = Duration::from_secs(20);

/// The step at which opening a Direct3D device stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum D3dStop {
    /// No d3d11.dll, or no hardware or WARP device at feature level 11_0.
    Device,
    /// An entry point did not compile, or the device refused the bytecode.
    Compile,
    /// The self-test scene did not match the CPU bytes.
    SelfTest,
    /// Opening took longer than [`OPEN_LIMIT`].
    TimedOut,
}

/// A Direct3D 11 device with every pass built.
pub(crate) struct D3dPassDevice {
    device: D3dDevice,
    vertex: VertexShader,
    pixel: Vec<PixelShader>,
}

fn compile(entry: &str, profile: &str, faults: Faults) -> Result<Vec<u8>, String> {
    let mut source = crate::PASSES_HLSL.to_string();
    if faults.break_shaders {
        source.push_str("\nthis line does not compile\n");
    }
    crate::sys::d3d_compile_with(&source, entry, profile, crate::sys::COMPILE_IEEE_STRICT)
}

impl D3dPassDevice {
    /// Make a device, build the passes, and run the self-test.
    pub(crate) fn open(faults: Faults) -> Result<D3dPassDevice, D3dStop> {
        // Feature level 0xf000 does not exist, so asking for it fails the
        // way a machine without a usable device does.
        let level = if faults.bad_context {
            0xf000
        } else {
            sys_d3d::FEATURE_LEVEL_11_0
        };
        let device = D3dDevice::create(level).map_err(|_| D3dStop::Device)?;
        let vs = compile("VSPass", "vs_5_0", faults).map_err(|_| D3dStop::Compile)?;
        let vertex = device.vertex_shader(&vs).map_err(|_| D3dStop::Compile)?;
        let pixel = Pass::ALL
            .iter()
            .map(|pass| {
                let ps =
                    compile(pass.hlsl_entry(), "ps_5_0", faults).map_err(|_| D3dStop::Compile)?;
                device.pixel_shader(&ps).map_err(|_| D3dStop::Compile)
            })
            .collect::<Result<Vec<_>, D3dStop>>()?;
        let mut device = D3dPassDevice {
            device,
            vertex,
            pixel,
        };
        let matched = device
            .with_passes(gpu::self_test)
            .map_err(|_| D3dStop::SelfTest)?;
        if !matched {
            return Err(D3dStop::SelfTest);
        }
        Ok(device)
    }

    /// [`D3dPassDevice::open`] on a helper thread, giving up after `limit`.
    /// The helper thread is left behind if the driver hangs.
    pub(crate) fn open_with_limit(
        faults: Faults,
        limit: Duration,
    ) -> Result<D3dPassDevice, D3dStop> {
        let (tx, rx) = mpsc::channel();
        let spawned = std::thread::Builder::new()
            .name("anvil-gpu-d3d-open".into())
            .spawn(move || {
                let _ = tx.send(D3dPassDevice::open(faults));
            });
        if spawned.is_err() {
            return Err(D3dStop::Device);
        }
        match rx.recv_timeout(limit) {
            Ok(result) => result,
            Err(mpsc::RecvTimeoutError::Timeout) => Err(D3dStop::TimedOut),
            Err(mpsc::RecvTimeoutError::Disconnected) => Err(D3dStop::Device),
        }
    }
}

/// The process-wide Direct3D device, opened once.
pub(crate) fn shared() -> Result<Arc<Mutex<D3dPassDevice>>, D3dStop> {
    static SHARED: OnceLock<Result<Arc<Mutex<D3dPassDevice>>, D3dStop>> = OnceLock::new();
    SHARED
        .get_or_init(|| {
            D3dPassDevice::open_with_limit(Faults::default(), OPEN_LIMIT)
                .map(|d| Arc::new(Mutex::new(d)))
        })
        .clone()
}

/// A [`PassDevice`] for one call. Its textures are released when it drops.
struct D3dPasses<'a> {
    device: &'a D3dDevice,
    vertex: &'a VertexShader,
    pixel: &'a [PixelShader],
    textures: Vec<D3dTexture>,
}

fn device_error<T>(r: Result<T, sys_d3d::D3dError>) -> Result<T, ComposeError> {
    r.map_err(|_| ComposeError::Device)
}

impl D3dPasses<'_> {
    fn add(&mut self, t: D3dTexture) -> TexId {
        self.textures.push(t);
        TexId(self.textures.len() - 1)
    }

    fn get(&self, t: TexId) -> Result<&D3dTexture, ComposeError> {
        self.textures.get(t.0).ok_or(ComposeError::Device)
    }
}

impl PassDevice for D3dPasses<'_> {
    fn max_side(&self) -> u32 {
        sys_d3d::MAX_SIDE
    }

    fn upload(
        &mut self,
        format: TexFormat,
        width: u32,
        height: u32,
        data: &[u8],
    ) -> Result<TexId, ComposeError> {
        let t = device_error(self.device.texture(format, width, height, Some(data)))?;
        Ok(self.add(t))
    }

    fn render_target(
        &mut self,
        format: TexFormat,
        width: u32,
        height: u32,
    ) -> Result<TexId, ComposeError> {
        let t = device_error(self.device.texture(format, width, height, None))?;
        Ok(self.add(t))
    }

    fn draw(
        &mut self,
        pass: Pass,
        out: TexId,
        inputs: &[(Slot, TexId)],
        params: &PassParams,
    ) -> Result<(), ComposeError> {
        let target = self.get(out)?;
        let bound = inputs
            .iter()
            .map(|&(slot, t)| Ok((slot.hlsl_register(), self.get(t)?)))
            .collect::<Result<Vec<_>, ComposeError>>()?;
        let pixel = self.pixel.get(pass.index()).ok_or(ComposeError::Device)?;
        let (w, h) = target.size();
        let constants = gpu::hlsl_constants(params, w, h);
        device_error(
            self.device
                .draw(self.vertex, pixel, target, &bound, &constants),
        )
    }

    fn read_rgba(&mut self, tex: TexId, out: &mut [u8]) -> Result<(), ComposeError> {
        let t = self.get(tex)?;
        device_error(self.device.read_rgba(t, out))
    }
}

impl SharedDevice for D3dPassDevice {
    fn with_passes<R>(
        &mut self,
        f: impl FnOnce(&mut dyn PassDevice) -> R,
    ) -> Result<R, ComposeError> {
        // A removed device fails every call; leave it to software.
        if self.device.removed() {
            return Err(ComposeError::Device);
        }
        let mut passes = D3dPasses {
            device: &self.device,
            vertex: &self.vertex,
            pixel: &self.pixel,
            textures: Vec::new(),
        };
        let result = f(&mut passes);
        drop(passes);
        if self.device.removed() {
            return Err(ComposeError::Device);
        }
        Ok(result)
    }
}

// These tests are built for Windows but have not been run on a Windows
// machine. They skip when no device can be made.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::DeviceBackend;
    use crate::gpu::testing::{draw_scene, within_limit};
    use crate::{BackendKind, Compositor, device_pass_draws};

    fn have_d3d() -> bool {
        match shared() {
            Ok(_) => true,
            Err(stop) => {
                eprintln!("skipped: no usable Direct3D 11 device here ({stop:?})");
                false
            }
        }
    }

    fn compositor_with(faults: Faults) -> (Compositor, Result<(), D3dStop>) {
        match D3dPassDevice::open_with_limit(faults, OPEN_LIMIT) {
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
    fn a_machine_with_a_d3d_device_compiles_and_matches() {
        within_limit(|| match shared() {
            Ok(_) => {}
            Err(stop @ (D3dStop::Device | D3dStop::TimedOut)) => {
                eprintln!("skipped: no usable Direct3D 11 device here ({stop:?})");
            }
            Err(stop) => panic!("the HLSL passes failed on a working device: {stop:?}"),
        });
    }

    #[test]
    fn the_d3d_device_draws_the_scene_like_software() {
        within_limit(|| {
            if !have_d3d() {
                return;
            }
            let want = draw_scene(&mut Compositor::software());
            let before = device_pass_draws();
            let mut c = Compositor::open();
            assert_eq!(c.active_backend(), BackendKind::Hardware);
            assert_eq!(draw_scene(&mut c), want);
            assert_eq!(c.fallbacks(), 0);
            assert!(device_pass_draws() >= before + 13);
        });
    }

    #[test]
    fn hlsl_that_does_not_compile_leaves_software() {
        within_limit(|| {
            if !have_d3d() {
                return;
            }
            let faults = Faults {
                break_shaders: true,
                ..Faults::default()
            };
            let (mut c, opened) = compositor_with(faults);
            assert_eq!(opened, Err(D3dStop::Compile));
            assert_eq!(c.active_backend(), BackendKind::Software);
            assert_eq!(draw_scene(&mut c), draw_scene(&mut Compositor::software()));
        });
    }

    #[test]
    fn a_device_that_cannot_be_made_leaves_software() {
        within_limit(|| {
            if !have_d3d() {
                return;
            }
            let faults = Faults {
                bad_context: true,
                ..Faults::default()
            };
            let (mut c, opened) = compositor_with(faults);
            assert_eq!(opened, Err(D3dStop::Device));
            assert_eq!(c.active_backend(), BackendKind::Software);
            assert_eq!(draw_scene(&mut c), draw_scene(&mut Compositor::software()));
        });
    }

    #[test]
    fn a_d3d_draw_that_fails_midway_is_redrawn_in_software() {
        within_limit(|| {
            if !have_d3d() {
                return;
            }
            let want = draw_scene(&mut Compositor::software());
            for fail_draw in 1..=8 {
                let faults = Faults {
                    fail_draw: Some(fail_draw),
                    ..Faults::default()
                };
                let (mut c, opened) = compositor_with(faults);
                assert_eq!(opened, Ok(()));
                assert_eq!(draw_scene(&mut c), want, "failing draw {fail_draw}");
                assert_eq!(c.fallbacks(), 1);
                assert_eq!(c.active_backend(), BackendKind::Software);
            }
        });
    }
}
