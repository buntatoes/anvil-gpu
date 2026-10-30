// SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0-OR-Commercial
// Copyright (c) 2026 Buntato
//! Backend choice for the compositor passes, with an automatic software fallback.
//!
//! [`SoftwareBackend`] is always there and is the reference. A device backend
//! is used only when [`select_backend`] says the probe found a usable device,
//! the shader passes compiled, and a self-test scene drawn on the device
//! matched the software bytes exactly. Any device error puts the touched
//! pixels back, drops the device for good, and redraws the same call in
//! software, so the caller always gets the software result.
//!
//! On Linux the device is OpenGL 3.3 core through EGL: the passes in
//! [`crate::PASSES_GLSL`] run into offscreen textures and are read back. On
//! Windows it is Direct3D 11 with [`crate::PASSES_HLSL`]. On other platforms
//! there is no device draw path, so [`Compositor::open`] draws in software.

use std::sync::OnceLock;

use crate::compose::{
    self, BlitDesc, Bounds, ComposeError, GpuSceneBatch, GradientDesc, RgbaTarget, ShadowDesc,
    VideoFrame,
};

/// Which kind of backend drew a call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendKind {
    Software,
    Hardware,
}

/// One way of drawing the compositor passes.
///
/// The [`Compositor`] checks every input before it calls a backend, so a
/// backend only sees sizes, strides, and values that the software path
/// accepts. A backend may write only inside the bounds of the call (the
/// clipped shape, gradient rectangle, or blit rectangle). When it returns an
/// error, the compositor restores those pixels and redraws in software. A
/// backend must not panic.
pub trait ComposeBackend {
    fn kind(&self) -> BackendKind;
    fn box_shadow(
        &mut self,
        target: &mut RgbaTarget<'_>,
        desc: &ShadowDesc,
    ) -> Result<(), ComposeError>;
    fn fill_gradient(
        &mut self,
        target: &mut RgbaTarget<'_>,
        desc: &GradientDesc<'_>,
    ) -> Result<(), ComposeError>;
    fn fill_rounded_gradient(
        &mut self,
        target: &mut RgbaTarget<'_>,
        desc: &GradientDesc<'_>,
        corner_radius: f32,
    ) -> Result<(), ComposeError> {
        compose::draw_rounded_gradient(target, desc, corner_radius)
    }
    fn blit_frame(
        &mut self,
        target: &mut RgbaTarget<'_>,
        frame: &VideoFrame<'_>,
        desc: &BlitDesc,
    ) -> Result<(), ComposeError>;
    fn execute_scene_batch(
        &mut self,
        target: &mut RgbaTarget<'_>,
        batch: &GpuSceneBatch,
    ) -> Result<(), ComposeError> {
        compose::draw_scene_batch(target, batch)
    }
}

/// The CPU passes in [`crate::draw_box_shadow`], [`crate::draw_gradient`], and
/// [`crate::draw_blit`].
#[derive(Clone, Copy, Debug, Default)]
pub struct SoftwareBackend;

impl ComposeBackend for SoftwareBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Software
    }

    fn box_shadow(
        &mut self,
        target: &mut RgbaTarget<'_>,
        desc: &ShadowDesc,
    ) -> Result<(), ComposeError> {
        compose::draw_box_shadow(target, desc)
    }

    fn fill_gradient(
        &mut self,
        target: &mut RgbaTarget<'_>,
        desc: &GradientDesc<'_>,
    ) -> Result<(), ComposeError> {
        compose::draw_gradient(target, desc)
    }

    fn blit_frame(
        &mut self,
        target: &mut RgbaTarget<'_>,
        frame: &VideoFrame<'_>,
        desc: &BlitDesc,
    ) -> Result<(), ComposeError> {
        compose::draw_blit(target, frame, desc)
    }
}

/// The system graphics API a probe looked at.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HardwareApi {
    /// Nothing to probe on this platform.
    None,
    /// `libEGL.so.1` on Linux.
    Egl,
    /// `d3d11.dll` and `d3dcompiler_47.dll` on Windows.
    Direct3d11,
}

/// What the hardware probe found.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HardwareProbe {
    pub api: HardwareApi,
    /// EGL: a display initialized (surfaceless platform first, then the
    /// default display). Direct3D: a hardware or WARP device was created. A
    /// display or device can still be a software driver; the probe cannot
    /// tell.
    pub device_opened: bool,
    /// EGL version, or the Direct3D feature level as `(11, 0)`. `(0, 0)` when
    /// nothing opened.
    pub api_version: (i32, i32),
    /// A context was made and every compositor pass compiled for this API.
    /// On Linux that is an OpenGL 3.3 core context and the GLSL passes; on
    /// Windows the `DXBC` compile of every HLSL entry point.
    pub passes_compiled: bool,
    /// A device draw path exists in this build and its self-test scene
    /// matched the software bytes exactly.
    pub submit_ready: bool,
}

impl HardwareProbe {
    /// A probe that found nothing.
    pub const fn none() -> Self {
        Self {
            api: HardwareApi::None,
            device_opened: false,
            api_version: (0, 0),
            passes_compiled: false,
            submit_ready: false,
        }
    }

    /// Why the hardware path is not used, or `None` when it would be.
    pub fn reason(&self) -> Option<&'static str> {
        if self.api == HardwareApi::None {
            Some("no graphics API to probe on this platform")
        } else if !self.device_opened {
            Some("no display or device opened")
        } else if !self.passes_compiled {
            Some("no device context, or the compositor passes did not compile")
        } else if !self.submit_ready {
            Some("no device draw path, or its self-test did not match software")
        } else {
            None
        }
    }
}

/// Software unless the probe found a device, compiled passes, and a draw path
/// whose self-test matched.
pub fn select_backend(probe: &HardwareProbe) -> BackendKind {
    if probe.reason().is_none() {
        BackendKind::Hardware
    } else {
        BackendKind::Software
    }
}

/// Probe the platform once and cache the answer for the process.
///
/// On Linux this opens the EGL display, makes an OpenGL context, builds the
/// passes, and draws the self-test scene, which takes a moment on the first
/// call. On Windows it does the same with a Direct3D 11 device and the HLSL
/// passes. A driver that does not answer within 20 seconds counts as missing.
pub fn probe_hardware() -> HardwareProbe {
    static PROBE: OnceLock<HardwareProbe> = OnceLock::new();
    *PROBE.get_or_init(run_probe)
}

#[cfg(target_os = "linux")]
fn run_probe() -> HardwareProbe {
    use crate::gpu_gl::GlStop;
    let Some(display) = crate::sys_gl::egl_display() else {
        return HardwareProbe {
            api: HardwareApi::Egl,
            ..HardwareProbe::none()
        };
    };
    let stop = crate::gpu_gl::shared().err();
    HardwareProbe {
        api: HardwareApi::Egl,
        device_opened: true,
        api_version: display.version,
        passes_compiled: matches!(stop, None | Some(GlStop::SelfTest)),
        submit_ready: stop.is_none(),
    }
}

#[cfg(target_os = "windows")]
fn run_probe() -> HardwareProbe {
    use crate::gpu_d3d::D3dStop;
    let passes_compiled = crate::PASS_ENTRY_POINTS.iter().all(|&(entry, profile)| {
        crate::sys::d3d_compile(crate::PASSES_HLSL, entry, profile).is_ok()
    });
    match crate::sys::d3d11_create_device() {
        Some((level, _)) => {
            let stop = crate::gpu_d3d::shared().err();
            HardwareProbe {
                api: HardwareApi::Direct3d11,
                device_opened: true,
                api_version: (((level >> 12) & 0xf) as i32, ((level >> 8) & 0xf) as i32),
                passes_compiled: passes_compiled && matches!(stop, None | Some(D3dStop::SelfTest)),
                submit_ready: stop.is_none(),
            }
        }
        None => HardwareProbe {
            api: HardwareApi::Direct3d11,
            passes_compiled,
            ..HardwareProbe::none()
        },
    }
}

#[cfg(not(any(target_os = "linux", target_os = "windows")))]
fn run_probe() -> HardwareProbe {
    HardwareProbe::none()
}

/// The device backend for a probe that selected hardware.
#[cfg(target_os = "linux")]
fn open_device(_probe: &HardwareProbe) -> Option<Box<dyn ComposeBackend + Send>> {
    let device = crate::gpu_gl::shared().ok()?;
    Some(Box::new(crate::gpu::DeviceBackend::new(
        device,
        crate::gpu::Faults::default(),
    )))
}

/// The device backend for a probe that selected hardware.
#[cfg(target_os = "windows")]
fn open_device(_probe: &HardwareProbe) -> Option<Box<dyn ComposeBackend + Send>> {
    let device = crate::gpu_d3d::shared().ok()?;
    Some(Box::new(crate::gpu::DeviceBackend::new(
        device,
        crate::gpu::Faults::default(),
    )))
}

/// No device draw path on this platform.
#[cfg(not(any(target_os = "linux", target_os = "windows")))]
fn open_device(_probe: &HardwareProbe) -> Option<Box<dyn ComposeBackend + Send>> {
    None
}

/// Draws compositor passes on the best backend and falls back to software.
pub struct Compositor {
    software: SoftwareBackend,
    device: Option<Box<dyn ComposeBackend + Send>>,
    probe: HardwareProbe,
    fallbacks: u64,
}

impl std::fmt::Debug for Compositor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Compositor")
            .field("active", &self.active_backend())
            .field("probe", &self.probe)
            .field("fallbacks", &self.fallbacks)
            .finish()
    }
}

impl Default for Compositor {
    fn default() -> Self {
        Self::open()
    }
}

impl Compositor {
    /// Probe (once per process) and pick a backend.
    pub fn open() -> Self {
        let probe = probe_hardware();
        let device = match select_backend(&probe) {
            BackendKind::Hardware => open_device(&probe),
            BackendKind::Software => None,
        };
        Self {
            software: SoftwareBackend,
            device,
            probe,
            fallbacks: 0,
        }
    }

    /// Software only. No probe runs.
    pub fn software() -> Self {
        Self {
            software: SoftwareBackend,
            device: None,
            probe: HardwareProbe::none(),
            fallbacks: 0,
        }
    }

    /// Use `device` first and software after any device error.
    pub fn with_device(device: Box<dyn ComposeBackend + Send>) -> Self {
        Self {
            device: Some(device),
            ..Self::software()
        }
    }

    /// The backend the next call will try first.
    pub fn active_backend(&self) -> BackendKind {
        self.device
            .as_ref()
            .map_or(BackendKind::Software, |d| d.kind())
    }

    pub fn probe(&self) -> &HardwareProbe {
        &self.probe
    }

    /// Calls a device failed and software redrew.
    pub fn fallbacks(&self) -> u64 {
        self.fallbacks
    }

    /// Draw a blurred box shadow.
    pub fn box_shadow(
        &mut self,
        target: &mut RgbaTarget<'_>,
        desc: &ShadowDesc,
    ) -> Result<(), ComposeError> {
        let Some(bounds) = compose::shadow_bounds(target, desc)? else {
            return Ok(());
        };
        self.run(target, bounds, |b, t| b.box_shadow(t, desc))
    }

    /// Fill a linear or radial gradient.
    pub fn fill_gradient(
        &mut self,
        target: &mut RgbaTarget<'_>,
        desc: &GradientDesc<'_>,
    ) -> Result<(), ComposeError> {
        let Some(bounds) = compose::gradient_bounds(target, desc)? else {
            return Ok(());
        };
        self.run(target, bounds, |b, t| b.fill_gradient(t, desc))
    }

    /// Fill a linear or radial gradient clipped to an SDF rounded rectangle.
    pub fn fill_rounded_gradient(
        &mut self,
        target: &mut RgbaTarget<'_>,
        desc: &GradientDesc<'_>,
        corner_radius: f32,
    ) -> Result<(), ComposeError> {
        if !corner_radius.is_finite() || corner_radius < 0.0 {
            return Err(ComposeError::BadInput);
        }
        let Some(bounds) = compose::gradient_bounds(target, desc)? else {
            return Ok(());
        };
        self.run(target, bounds, |b, t| {
            b.fill_rounded_gradient(t, desc, corner_radius)
        })
    }

    /// Scale, convert, and copy a video frame.
    pub fn blit_frame(
        &mut self,
        target: &mut RgbaTarget<'_>,
        frame: &VideoFrame<'_>,
        desc: &BlitDesc,
    ) -> Result<(), ComposeError> {
        let Some(bounds) = compose::blit_bounds(target, frame, desc)? else {
            return Ok(());
        };
        self.run(target, bounds, |b, t| b.blit_frame(t, frame, desc))
    }

    /// Execute an entire [`GpuSceneBatch`] in a single GPU session, falling back
    /// transactionally to [`SoftwareBackend`] on any device error.
    pub fn execute_scene_batch(
        &mut self,
        target: &mut RgbaTarget<'_>,
        batch: &GpuSceneBatch,
    ) -> Result<(), ComposeError> {
        compose::validate_scene_batch(batch)?;
        if batch.is_empty() || target.width() == 0 || target.height() == 0 {
            return Ok(());
        }
        let full = Bounds {
            x0: 0,
            y0: 0,
            x1: target.width(),
            y1: target.height(),
        };
        let res = self.run(target, full, |b, t| b.execute_scene_batch(t, batch));
        if res.is_ok() {
            crate::gpu::record_scene_batch_telemetry(batch.len());
        }
        res
    }

    fn run(
        &mut self,
        target: &mut RgbaTarget<'_>,
        bounds: Bounds,
        op: impl Fn(&mut dyn ComposeBackend, &mut RgbaTarget<'_>) -> Result<(), ComposeError>,
    ) -> Result<(), ComposeError> {
        if let Some(device) = self.device.as_mut() {
            let saved = target.save(bounds);
            if op(device.as_mut(), target).is_ok() {
                return Ok(());
            }
            target.restore(bounds, &saved);
            self.device = None;
            self.fallbacks += 1;
        }
        op(&mut self.software, target)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Filter, GpuColor, GpuGradientStop, GpuRect, GradientShape, Nv12Surface};

    fn probe(device_opened: bool, passes_compiled: bool, submit_ready: bool) -> HardwareProbe {
        HardwareProbe {
            api: HardwareApi::Egl,
            device_opened,
            api_version: (1, 5),
            passes_compiled,
            submit_ready,
        }
    }

    #[test]
    fn hardware_is_selected_only_when_every_check_passes() {
        assert_eq!(
            select_backend(&HardwareProbe::none()),
            BackendKind::Software
        );
        for opened in [false, true] {
            for compiled in [false, true] {
                for ready in [false, true] {
                    let want = if opened && compiled && ready {
                        BackendKind::Hardware
                    } else {
                        BackendKind::Software
                    };
                    let p = probe(opened, compiled, ready);
                    assert_eq!(select_backend(&p), want);
                    assert_eq!(p.reason().is_none(), want == BackendKind::Hardware);
                }
            }
        }
        // The probe is cached, and open follows it.
        let found = probe_hardware();
        assert!(!found.submit_ready || found.passes_compiled);
        assert_eq!(Compositor::open().active_backend(), select_backend(&found));
        assert_eq!(probe_hardware(), found);
        // The Windows runners have at least WARP, and the passes must compile there.
        #[cfg(target_os = "windows")]
        {
            assert!(found.device_opened);
            assert!(found.passes_compiled);
        }
        #[cfg(target_os = "linux")]
        assert_eq!(found.api, HardwareApi::Egl);
    }

    // A device that scribbles over the whole call area and then fails.
    struct BrokenDevice {
        calls: u32,
    }

    impl BrokenDevice {
        fn scribble(
            &mut self,
            target: &mut RgbaTarget<'_>,
            bounds: Bounds,
        ) -> Result<(), ComposeError> {
            self.calls += 1;
            let junk = vec![0xA5; (bounds.width() * bounds.height() * 4) as usize];
            target.restore(bounds, &junk);
            Err(ComposeError::Device)
        }
    }

    impl ComposeBackend for BrokenDevice {
        fn kind(&self) -> BackendKind {
            BackendKind::Hardware
        }

        fn box_shadow(
            &mut self,
            target: &mut RgbaTarget<'_>,
            desc: &ShadowDesc,
        ) -> Result<(), ComposeError> {
            let b = compose::shadow_bounds(target, desc)?.unwrap();
            self.scribble(target, b)
        }

        fn fill_gradient(
            &mut self,
            target: &mut RgbaTarget<'_>,
            desc: &GradientDesc<'_>,
        ) -> Result<(), ComposeError> {
            let b = compose::gradient_bounds(target, desc)?.unwrap();
            self.scribble(target, b)
        }

        fn blit_frame(
            &mut self,
            target: &mut RgbaTarget<'_>,
            frame: &VideoFrame<'_>,
            desc: &BlitDesc,
        ) -> Result<(), ComposeError> {
            let b = compose::blit_bounds(target, frame, desc)?.unwrap();
            self.scribble(target, b)
        }
    }

    // A device that draws with the software passes, standing in for a working one.
    struct GoodDevice;

    impl ComposeBackend for GoodDevice {
        fn kind(&self) -> BackendKind {
            BackendKind::Hardware
        }

        fn box_shadow(
            &mut self,
            target: &mut RgbaTarget<'_>,
            desc: &ShadowDesc,
        ) -> Result<(), ComposeError> {
            SoftwareBackend.box_shadow(target, desc)
        }

        fn fill_gradient(
            &mut self,
            target: &mut RgbaTarget<'_>,
            desc: &GradientDesc<'_>,
        ) -> Result<(), ComposeError> {
            SoftwareBackend.fill_gradient(target, desc)
        }

        fn blit_frame(
            &mut self,
            target: &mut RgbaTarget<'_>,
            frame: &VideoFrame<'_>,
            desc: &BlitDesc,
        ) -> Result<(), ComposeError> {
            SoftwareBackend.blit_frame(target, frame, desc)
        }
    }

    fn draw_scene(c: &mut Compositor) -> Vec<u8> {
        let mut buf = [30u8, 40, 50, 255].repeat(40 * 30);
        let mut t = RgbaTarget::packed(&mut buf, 40, 30).unwrap();
        let mut shadow = ShadowDesc::new(
            GpuRect::new(6.0, 5.0, 20.0, 12.0),
            GpuColor::rgba(0, 0, 0, 160),
        );
        shadow.blur_radius = 6.0;
        shadow.corner_radius = 3.0;
        shadow.offset_y = 2.0;
        c.box_shadow(&mut t, &shadow).unwrap();
        let stops = [
            GpuGradientStop::new(0.0, GpuColor::rgba(250, 120, 20, 255)),
            GpuGradientStop::new(1.0, GpuColor::rgba(20, 80, 250, 200)),
        ];
        let grad = GradientDesc {
            rect: GpuRect::new(4.0, 20.0, 30.0, 8.0),
            shape: GradientShape::Radial {
                center_x: 19.0,
                center_y: 24.0,
                radius_x: 15.0,
                radius_y: 4.0,
            },
            stops: &stops,
            clip: None,
        };
        c.fill_gradient(&mut t, &grad).unwrap();
        let rgba = [200u8, 30, 90, 255].repeat(16);
        let surface = Nv12Surface::from_rgba_bt709(4, 4, &rgba).unwrap();
        let mut blit = BlitDesc::new(28, 2, 9, 7);
        blit.filter = Filter::Bilinear;
        c.blit_frame(&mut t, &VideoFrame::from_nv12(&surface), &blit)
            .unwrap();
        buf
    }

    #[test]
    fn a_failing_device_falls_back_with_the_same_pixels() {
        let reference = draw_scene(&mut Compositor::software());
        let mut c = Compositor::with_device(Box::new(BrokenDevice { calls: 0 }));
        assert_eq!(c.active_backend(), BackendKind::Hardware);
        let got = draw_scene(&mut c);
        assert_eq!(got, reference);
        // The device is dropped after its first error.
        assert_eq!(c.fallbacks(), 1);
        assert_eq!(c.active_backend(), BackendKind::Software);
    }

    #[test]
    fn a_working_device_stays_selected() {
        let reference = draw_scene(&mut Compositor::software());
        let mut c = Compositor::with_device(Box::new(GoodDevice));
        let got = draw_scene(&mut c);
        assert_eq!(got, reference);
        assert_eq!(c.fallbacks(), 0);
        assert_eq!(c.active_backend(), BackendKind::Hardware);
    }

    #[test]
    fn bad_input_is_refused_before_any_backend_runs() {
        let mut c = Compositor::with_device(Box::new(BrokenDevice { calls: 0 }));
        let mut buf = vec![0u8; 16];
        let mut t = RgbaTarget::packed(&mut buf, 2, 2).unwrap();
        let mut shadow = ShadowDesc::new(
            GpuRect::new(0.0, 0.0, 1.0, 1.0),
            GpuColor::rgba(0, 0, 0, 255),
        );
        shadow.offset_x = f32::NAN;
        assert_eq!(c.box_shadow(&mut t, &shadow), Err(ComposeError::BadInput));
        // Nothing reached the device, so it is still in place.
        assert_eq!(c.fallbacks(), 0);
        assert_eq!(c.active_backend(), BackendKind::Hardware);
        assert_eq!(buf, vec![0u8; 16]);
    }
}
