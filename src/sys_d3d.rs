// SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0-OR-Commercial
// Copyright (c) 2026 Buntato
//! Direct3D 11 calls for the device draw path (Windows).
//!
//! Like `sys` and `sys_gl`, this module may use `unsafe`; each block says
//! why it is sound. `d3d11.dll` is loaded at run time with `LoadLibraryA`,
//! so nothing is linked besides kernel32. The calls made, by interface:
//!
//! - `d3d11.dll`: `D3D11CreateDevice` (hardware first, then WARP, feature
//!   level 11_0, no flags).
//! - `ID3D11Device`: `CreateBuffer`, `CreateTexture2D`,
//!   `CreateShaderResourceView`, `CreateRenderTargetView`,
//!   `CreateVertexShader`, `CreatePixelShader`, `CreateRasterizerState`,
//!   `GetDeviceRemovedReason`.
//! - `ID3D11DeviceContext`: `VSSetConstantBuffers`, `PSSetShaderResources`,
//!   `PSSetShader`, `VSSetShader`, `Draw`, `Map`, `Unmap`,
//!   `PSSetConstantBuffers`, `IASetPrimitiveTopology`, `OMSetRenderTargets`,
//!   `RSSetState`, `RSSetViewports`, `CopyResource`, `UpdateSubresource`.
//! - `IUnknown::Release` on every object, once, when its owner drops.
//!
//! Methods are called through their slot in the interface's vtable. The
//! slot numbers come from the declaration order in d3d11.h and are listed
//! next to each call.
//!
//! Objects are owned by [`Com`], which releases its one reference on drop.
//! No raw pointer leaves this module. This path has only been built for
//! Windows and type-checked, never run on a Windows machine.

use std::ffi::{c_char, c_void};
use std::ptr::{NonNull, null, null_mut};

use crate::gpu::TexFormat;

// SAFETY: these are the documented kernel32 signatures. They are only
// called below with NUL-terminated names, and every result is checked
// for null before use.
#[link(name = "kernel32")]
unsafe extern "system" {
    fn LoadLibraryA(name: *const c_char) -> *mut c_void;
    fn GetProcAddress(module: *mut c_void, name: *const c_char) -> *mut c_void;
}

/// A failed Direct3D call, by name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct D3dError(pub(crate) &'static str);

/// Size of the `PassConstants` buffer in `shaders/passes.hlsl`.
pub(crate) const CONSTANT_BYTES: usize = 448;

/// D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION at feature level 11_0.
pub(crate) const MAX_SIDE: u32 = 16384;

/// Number of texture registers the passes use (t0 to t7).
pub(crate) const TEXTURE_SLOTS: usize = 8;

/// D3D_FEATURE_LEVEL_11_0, the level the passes (Shader Model 5.0) need.
pub(crate) const FEATURE_LEVEL_11_0: u32 = 0xb000;
const SDK_VERSION: u32 = 7;
const DRIVER_HARDWARE: u32 = 1;
const DRIVER_WARP: u32 = 5;
const USAGE_DEFAULT: u32 = 0;
const USAGE_IMMUTABLE: u32 = 1;
const USAGE_STAGING: u32 = 3;
const BIND_CONSTANT_BUFFER: u32 = 0x4;
const BIND_SHADER_RESOURCE: u32 = 0x8;
const BIND_RENDER_TARGET: u32 = 0x20;
const CPU_ACCESS_READ: u32 = 0x20000;
const MAP_READ: u32 = 1;
const FILL_SOLID: u32 = 3;
const CULL_NONE: u32 = 1;
const TOPOLOGY_TRIANGLESTRIP: u32 = 5;

fn dxgi_format(format: TexFormat) -> u32 {
    match format {
        TexFormat::Rgba8 => 0x1c, // DXGI_FORMAT_R8G8B8A8_UNORM
        TexFormat::R8 => 0x3d,    // DXGI_FORMAT_R8_UNORM
        TexFormat::Rg8 => 0x31,   // DXGI_FORMAT_R8G8_UNORM
        TexFormat::R32f => 0x29,  // DXGI_FORMAT_R32_FLOAT
    }
}

#[repr(C)]
struct SampleDesc {
    count: u32,
    quality: u32,
}

#[repr(C)]
struct Texture2dDesc {
    width: u32,
    height: u32,
    mip_levels: u32,
    array_size: u32,
    format: u32,
    sample: SampleDesc,
    usage: u32,
    bind_flags: u32,
    cpu_access: u32,
    misc_flags: u32,
}

#[repr(C)]
struct BufferDesc {
    byte_width: u32,
    usage: u32,
    bind_flags: u32,
    cpu_access: u32,
    misc_flags: u32,
    structure_stride: u32,
}

#[repr(C)]
struct SubresourceData {
    sys_mem: *const c_void,
    pitch: u32,
    slice_pitch: u32,
}

#[repr(C)]
struct MappedSubresource {
    data: *mut c_void,
    row_pitch: u32,
    depth_pitch: u32,
}

#[repr(C)]
struct RasterizerDesc {
    fill_mode: u32,
    cull_mode: u32,
    front_counter_clockwise: i32,
    depth_bias: i32,
    depth_bias_clamp: f32,
    slope_scaled_depth_bias: f32,
    depth_clip_enable: i32,
    scissor_enable: i32,
    multisample_enable: i32,
    antialiased_line_enable: i32,
}

#[repr(C)]
struct Viewport {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    min_depth: f32,
    max_depth: f32,
}

type Obj = *mut c_void;

/// One owned reference to a COM object.
struct Com(NonNull<c_void>);

// SAFETY: the device is created without D3D11_CREATE_DEVICE_SINGLETHREADED,
// so it and the objects it makes may be used and released from any thread.
// The immediate context is not thread-safe, but it is only reachable through
// a `&D3dDevice`, which the crate keeps behind a mutex (see gpu_d3d.rs), and
// `D3dDevice` is not `Sync`.
unsafe impl Send for Com {}

impl Com {
    /// Take ownership of `ptr`, or `None` when it is null.
    ///
    /// SAFETY (for callers): `ptr` must be null or a live COM object that the
    /// caller owns one reference to, which moves into the result.
    unsafe fn take(ptr: Obj) -> Option<Com> {
        NonNull::new(ptr).map(Com)
    }

    fn raw(&self) -> Obj {
        self.0.as_ptr()
    }

    /// The function in vtable slot `index`, or an error for an empty slot.
    ///
    /// SAFETY (for callers): `index` must be a slot of this object's
    /// interface, and `F` must be an `unsafe extern "system" fn` type with
    /// that method's signature, including the object as first argument.
    unsafe fn method<F: Copy>(&self, index: usize) -> Result<F, D3dError> {
        const { assert!(size_of::<F>() == size_of::<usize>()) };
        // SAFETY: a live COM object starts with a pointer to its vtable, an
        // array of function pointers that holds at least `index + 1`
        // entries (see above). The entry is read as a plain address first
        // and only turned into `F` when it is not null.
        unsafe {
            let vtbl = *(self.raw() as *const *const usize);
            if vtbl.is_null() {
                return Err(D3dError("object without a vtable"));
            }
            let entry = *vtbl.add(index);
            if entry == 0 {
                return Err(D3dError("empty vtable slot"));
            }
            Ok(std::mem::transmute_copy::<usize, F>(&entry))
        }
    }
}

impl Drop for Com {
    fn drop(&mut self) {
        type Release = unsafe extern "system" fn(Obj) -> u32;
        // SAFETY: slot 2 of every COM interface is IUnknown::Release, and
        // this value owns exactly one reference, given up here once.
        unsafe {
            if let Ok(release) = self.method::<Release>(2) {
                release(self.raw());
            }
        }
    }
}

/// Turn an HRESULT and an out pointer into an owned object.
///
/// SAFETY (for callers): `out` must be what the call that returned `hr`
/// wrote: null, or an owned reference when `hr` is a success code.
unsafe fn created(hr: i32, out: Obj, what: &'static str) -> Result<Com, D3dError> {
    if hr < 0 {
        // A failed call leaves the out pointer null, but release it if not.
        // SAFETY: as above.
        drop(unsafe { Com::take(out) });
        return Err(D3dError(what));
    }
    // SAFETY: as above.
    unsafe { Com::take(out) }.ok_or(D3dError(what))
}

/// A texture with a shader view, and a render target view when it was made
/// as a render target.
pub(crate) struct D3dTexture {
    texture: Com,
    view: Com,
    target: Option<Com>,
    format: TexFormat,
    width: u32,
    height: u32,
}

impl D3dTexture {
    pub(crate) fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }
}

/// A compiled vertex shader on a device.
pub(crate) struct VertexShader(Com);

/// A compiled pixel shader on a device.
pub(crate) struct PixelShader(Com);

/// A Direct3D 11 device, its immediate context, and the fixed state the
/// passes need: a rasterizer state that culls nothing and the constant
/// buffer.
pub(crate) struct D3dDevice {
    device: Com,
    context: Com,
    rasterizer: Com,
    constants: Com,
}

type CreateDeviceFn = unsafe extern "system" fn(
    Obj,
    u32,
    Obj,
    u32,
    *const u32,
    u32,
    u32,
    *mut Obj,
    *mut u32,
    *mut Obj,
) -> i32;
type CreateBufferFn =
    unsafe extern "system" fn(Obj, *const BufferDesc, *const SubresourceData, *mut Obj) -> i32;
type CreateTexture2dFn =
    unsafe extern "system" fn(Obj, *const Texture2dDesc, *const SubresourceData, *mut Obj) -> i32;
type CreateViewFn = unsafe extern "system" fn(Obj, Obj, *const c_void, *mut Obj) -> i32;
type CreateShaderFn = unsafe extern "system" fn(Obj, *const c_void, usize, Obj, *mut Obj) -> i32;
type CreateRasterizerFn = unsafe extern "system" fn(Obj, *const RasterizerDesc, *mut Obj) -> i32;
type RemovedReasonFn = unsafe extern "system" fn(Obj) -> i32;
type SetBuffersFn = unsafe extern "system" fn(Obj, u32, u32, *const Obj);
type SetViewsFn = unsafe extern "system" fn(Obj, u32, u32, *const Obj);
type SetShaderFn = unsafe extern "system" fn(Obj, Obj, *const Obj, u32);
type DrawFn = unsafe extern "system" fn(Obj, u32, u32);
type MapFn = unsafe extern "system" fn(Obj, Obj, u32, u32, u32, *mut MappedSubresource) -> i32;
type UnmapFn = unsafe extern "system" fn(Obj, Obj, u32);
type SetTopologyFn = unsafe extern "system" fn(Obj, u32);
type SetTargetsFn = unsafe extern "system" fn(Obj, u32, *const Obj, Obj);
type SetStateFn = unsafe extern "system" fn(Obj, Obj);
type SetViewportsFn = unsafe extern "system" fn(Obj, u32, *const Viewport);
type CopyResourceFn = unsafe extern "system" fn(Obj, Obj, Obj);
type UpdateFn = unsafe extern "system" fn(Obj, Obj, u32, *const c_void, *const c_void, u32, u32);

// ID3D11Device vtable slots.
const DEV_CREATE_BUFFER: usize = 3;
const DEV_CREATE_TEXTURE2D: usize = 5;
const DEV_CREATE_SRV: usize = 7;
const DEV_CREATE_RTV: usize = 9;
const DEV_CREATE_VERTEX_SHADER: usize = 12;
const DEV_CREATE_PIXEL_SHADER: usize = 15;
const DEV_CREATE_RASTERIZER: usize = 22;
const DEV_REMOVED_REASON: usize = 39;
// ID3D11DeviceContext vtable slots.
const CTX_VS_SET_CONSTANT_BUFFERS: usize = 7;
const CTX_PS_SET_SHADER_RESOURCES: usize = 8;
const CTX_PS_SET_SHADER: usize = 9;
const CTX_VS_SET_SHADER: usize = 11;
const CTX_DRAW: usize = 13;
const CTX_MAP: usize = 14;
const CTX_UNMAP: usize = 15;
const CTX_PS_SET_CONSTANT_BUFFERS: usize = 16;
const CTX_IA_SET_TOPOLOGY: usize = 24;
const CTX_OM_SET_RENDER_TARGETS: usize = 33;
const CTX_RS_SET_STATE: usize = 43;
const CTX_RS_SET_VIEWPORTS: usize = 44;
const CTX_COPY_RESOURCE: usize = 47;
const CTX_UPDATE_SUBRESOURCE: usize = 48;

fn create_device_fn() -> Result<CreateDeviceFn, D3dError> {
    // SAFETY: the names are NUL-terminated literals. The module is never
    // freed, so the function pointer stays valid for the process.
    let proc = unsafe {
        let dll = LoadLibraryA(c"d3d11.dll".as_ptr());
        if dll.is_null() {
            return Err(D3dError("d3d11.dll not available"));
        }
        GetProcAddress(dll, c"D3D11CreateDevice".as_ptr())
    };
    if proc.is_null() {
        return Err(D3dError("D3D11CreateDevice not found"));
    }
    // SAFETY: `proc` is the non-null D3D11CreateDevice export, whose
    // signature in d3d11.h matches `CreateDeviceFn`.
    Ok(unsafe { std::mem::transmute::<*mut c_void, CreateDeviceFn>(proc) })
}

impl D3dDevice {
    /// Make a device at `feature_level`: hardware first, then WARP.
    pub(crate) fn create(feature_level: u32) -> Result<D3dDevice, D3dError> {
        let create = create_device_fn()?;
        let levels = [feature_level];
        let mut last = D3dError("D3D11CreateDevice failed");
        for driver_type in [DRIVER_HARDWARE, DRIVER_WARP] {
            let mut device: Obj = null_mut();
            let mut context: Obj = null_mut();
            let mut level = 0u32;
            // SAFETY: a null adapter and software module are allowed for
            // these driver types. The level array outlives the call, and the
            // three out pointers are valid for writes. On success the call
            // hands back one owned reference in each of device and context.
            let hr = unsafe {
                create(
                    null_mut(),
                    driver_type,
                    null_mut(),
                    0,
                    levels.as_ptr(),
                    1,
                    SDK_VERSION,
                    &mut device,
                    &mut level,
                    &mut context,
                )
            };
            // SAFETY: both out pointers are null or owned references from
            // the call above.
            let made = unsafe {
                (
                    created(hr, device, "D3D11CreateDevice failed"),
                    created(hr, context, "D3D11CreateDevice gave no context"),
                )
            };
            match made {
                (Ok(device), Ok(context)) => {
                    return D3dDevice::finish(device, context);
                }
                (Err(e), _) | (_, Err(e)) => last = e,
            }
        }
        Err(last)
    }

    fn finish(device: Com, context: Com) -> Result<D3dDevice, D3dError> {
        let raster = RasterizerDesc {
            fill_mode: FILL_SOLID,
            cull_mode: CULL_NONE,
            front_counter_clockwise: 0,
            depth_bias: 0,
            depth_bias_clamp: 0.0,
            slope_scaled_depth_bias: 0.0,
            depth_clip_enable: 1,
            scissor_enable: 0,
            multisample_enable: 0,
            antialiased_line_enable: 0,
        };
        let mut out: Obj = null_mut();
        // SAFETY: slot 22 is ID3D11Device::CreateRasterizerState. The
        // description and out pointer are valid for the call.
        let rasterizer = unsafe {
            let f = device.method::<CreateRasterizerFn>(DEV_CREATE_RASTERIZER)?;
            let hr = f(device.raw(), &raster, &mut out);
            created(hr, out, "CreateRasterizerState failed")?
        };
        let desc = BufferDesc {
            byte_width: CONSTANT_BYTES as u32,
            usage: USAGE_DEFAULT,
            bind_flags: BIND_CONSTANT_BUFFER,
            cpu_access: 0,
            misc_flags: 0,
            structure_stride: 0,
        };
        let mut out: Obj = null_mut();
        // SAFETY: slot 3 is ID3D11Device::CreateBuffer. The description and
        // out pointer are valid; no initial data is passed.
        let constants = unsafe {
            let f = device.method::<CreateBufferFn>(DEV_CREATE_BUFFER)?;
            let hr = f(device.raw(), &desc, null(), &mut out);
            created(hr, out, "CreateBuffer failed")?
        };
        Ok(D3dDevice {
            device,
            context,
            rasterizer,
            constants,
        })
    }

    /// True when the device was removed or reset and can no longer draw.
    pub(crate) fn removed(&self) -> bool {
        // SAFETY: slot 39 is ID3D11Device::GetDeviceRemovedReason, which
        // takes no arguments besides the device.
        unsafe {
            match self.device.method::<RemovedReasonFn>(DEV_REMOVED_REASON) {
                Ok(f) => f(self.device.raw()) != 0,
                Err(_) => true,
            }
        }
    }

    fn shader(&self, slot: usize, dxbc: &[u8], what: &'static str) -> Result<Com, D3dError> {
        if dxbc.len() < 4 || &dxbc[..4] != b"DXBC" {
            return Err(D3dError("not DXBC bytecode"));
        }
        let mut out: Obj = null_mut();
        // SAFETY: `slot` is CreateVertexShader (12) or CreatePixelShader
        // (15), which share this signature. The bytecode pointer and length
        // describe `dxbc`, which outlives the call; the runtime checks the
        // bytecode itself. No class linkage is passed.
        unsafe {
            let f = self.device.method::<CreateShaderFn>(slot)?;
            let hr = f(
                self.device.raw(),
                dxbc.as_ptr().cast(),
                dxbc.len(),
                null_mut(),
                &mut out,
            );
            created(hr, out, what)
        }
    }

    pub(crate) fn vertex_shader(&self, dxbc: &[u8]) -> Result<VertexShader, D3dError> {
        self.shader(DEV_CREATE_VERTEX_SHADER, dxbc, "CreateVertexShader failed")
            .map(VertexShader)
    }

    pub(crate) fn pixel_shader(&self, dxbc: &[u8]) -> Result<PixelShader, D3dError> {
        self.shader(DEV_CREATE_PIXEL_SHADER, dxbc, "CreatePixelShader failed")
            .map(PixelShader)
    }

    /// A texture filled from `data` (rows top first, no padding), or with
    /// `None` an empty render target the passes can also read.
    pub(crate) fn texture(
        &self,
        format: TexFormat,
        width: u32,
        height: u32,
        data: Option<&[u8]>,
    ) -> Result<D3dTexture, D3dError> {
        if width == 0 || height == 0 || width > MAX_SIDE || height > MAX_SIDE {
            return Err(D3dError("texture size out of range"));
        }
        let pitch = width as usize * format.texel_bytes();
        if data.is_some_and(|d| d.len() != pitch * height as usize) {
            return Err(D3dError("texture data has the wrong length"));
        }
        let desc = Texture2dDesc {
            width,
            height,
            mip_levels: 1,
            array_size: 1,
            format: dxgi_format(format),
            sample: SampleDesc {
                count: 1,
                quality: 0,
            },
            usage: if data.is_some() {
                USAGE_IMMUTABLE
            } else {
                USAGE_DEFAULT
            },
            bind_flags: if data.is_some() {
                BIND_SHADER_RESOURCE
            } else {
                BIND_SHADER_RESOURCE | BIND_RENDER_TARGET
            },
            cpu_access: 0,
            misc_flags: 0,
        };
        let initial = data.map(|d| SubresourceData {
            sys_mem: d.as_ptr().cast(),
            pitch: pitch as u32,
            slice_pitch: 0,
        });
        let initial_ptr = initial
            .as_ref()
            .map_or(null(), |i| i as *const SubresourceData);
        let mut out: Obj = null_mut();
        // SAFETY: slot 5 is ID3D11Device::CreateTexture2D. The description
        // is valid. When given, the initial data points at `data`, which
        // holds exactly `pitch * height` bytes (checked above) and outlives
        // the call; the runtime copies it.
        let texture = unsafe {
            let f = self
                .device
                .method::<CreateTexture2dFn>(DEV_CREATE_TEXTURE2D)?;
            let hr = f(self.device.raw(), &desc, initial_ptr, &mut out);
            created(hr, out, "CreateTexture2D failed")?
        };
        let mut out: Obj = null_mut();
        // SAFETY: slot 7 is CreateShaderResourceView. The texture is live
        // and was made with the shader-resource bind flag; a null
        // description views the whole texture in its own format.
        let view = unsafe {
            let f = self.device.method::<CreateViewFn>(DEV_CREATE_SRV)?;
            let hr = f(self.device.raw(), texture.raw(), null(), &mut out);
            created(hr, out, "CreateShaderResourceView failed")?
        };
        let target = if data.is_none() {
            let mut out: Obj = null_mut();
            // SAFETY: slot 9 is CreateRenderTargetView, called the same way
            // on a texture made with the render-target bind flag.
            Some(unsafe {
                let f = self.device.method::<CreateViewFn>(DEV_CREATE_RTV)?;
                let hr = f(self.device.raw(), texture.raw(), null(), &mut out);
                created(hr, out, "CreateRenderTargetView failed")?
            })
        } else {
            None
        };
        Ok(D3dTexture {
            texture,
            view,
            target,
            format,
            width,
            height,
        })
    }

    /// Draw the four-vertex quad over all of `out` with `pixel`, reading
    /// `inputs` from the given texture registers.
    pub(crate) fn draw(
        &self,
        vertex: &VertexShader,
        pixel: &PixelShader,
        out: &D3dTexture,
        inputs: &[(usize, &D3dTexture)],
        constants: &[u8; CONSTANT_BYTES],
    ) -> Result<(), D3dError> {
        let Some(target) = &out.target else {
            return Err(D3dError(
                "drawing into a texture that is not a render target",
            ));
        };
        let mut views: [Obj; TEXTURE_SLOTS] = [null_mut(); TEXTURE_SLOTS];
        for &(slot, tex) in inputs {
            if slot >= TEXTURE_SLOTS {
                return Err(D3dError("texture register out of range"));
            }
            if std::ptr::eq(tex, out) {
                return Err(D3dError("a pass may not read its own output"));
            }
            views[slot] = tex.view.raw();
        }
        let none: [Obj; TEXTURE_SLOTS] = [null_mut(); TEXTURE_SLOTS];
        let viewport = Viewport {
            x: 0.0,
            y: 0.0,
            width: out.width as f32,
            height: out.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let ctx = &self.context;
        let c = ctx.raw();
        let rtv = target.raw();
        let cb = self.constants.raw();
        // SAFETY: each slot is the ID3D11DeviceContext method named by its
        // constant, with the signature given by its type. Every object
        // passed is live and owned by `self`, `vertex`, `pixel`, `out`, or
        // `inputs` for the whole block. The arrays passed with a count hold
        // that many entries; null entries unbind a register. The constant
        // data is CONSTANT_BYTES long, the size the buffer was made with,
        // and the call copies it. Shader resources are unbound first, so
        // the output is never bound for reading and writing at once, and
        // unbound again at the end so the next pass can draw into any of
        // these textures.
        unsafe {
            let set_views = ctx.method::<SetViewsFn>(CTX_PS_SET_SHADER_RESOURCES)?;
            let set_targets = ctx.method::<SetTargetsFn>(CTX_OM_SET_RENDER_TARGETS)?;
            let set_viewports = ctx.method::<SetViewportsFn>(CTX_RS_SET_VIEWPORTS)?;
            let set_state = ctx.method::<SetStateFn>(CTX_RS_SET_STATE)?;
            let set_topology = ctx.method::<SetTopologyFn>(CTX_IA_SET_TOPOLOGY)?;
            let set_vs = ctx.method::<SetShaderFn>(CTX_VS_SET_SHADER)?;
            let set_ps = ctx.method::<SetShaderFn>(CTX_PS_SET_SHADER)?;
            let update = ctx.method::<UpdateFn>(CTX_UPDATE_SUBRESOURCE)?;
            let vs_buffers = ctx.method::<SetBuffersFn>(CTX_VS_SET_CONSTANT_BUFFERS)?;
            let ps_buffers = ctx.method::<SetBuffersFn>(CTX_PS_SET_CONSTANT_BUFFERS)?;
            let draw = ctx.method::<DrawFn>(CTX_DRAW)?;

            set_views(c, 0, TEXTURE_SLOTS as u32, none.as_ptr());
            set_targets(c, 1, &rtv, null_mut());
            set_viewports(c, 1, &viewport);
            set_state(c, self.rasterizer.raw());
            set_topology(c, TOPOLOGY_TRIANGLESTRIP);
            set_vs(c, vertex.0.raw(), null(), 0);
            set_ps(c, pixel.0.raw(), null(), 0);
            update(c, cb, 0, null(), constants.as_ptr().cast(), 0, 0);
            vs_buffers(c, 0, 1, &cb);
            ps_buffers(c, 0, 1, &cb);
            set_views(c, 0, TEXTURE_SLOTS as u32, views.as_ptr());
            draw(c, 4, 0);
            set_views(c, 0, TEXTURE_SLOTS as u32, none.as_ptr());
            set_targets(c, 0, null(), null_mut());
        }
        Ok(())
    }

    /// Copy an RGBA8 texture into `out`, rows top first, no padding.
    pub(crate) fn read_rgba(&self, tex: &D3dTexture, out: &mut [u8]) -> Result<(), D3dError> {
        let row = tex.width as usize * 4;
        if tex.format != TexFormat::Rgba8 || out.len() != row * tex.height as usize {
            return Err(D3dError("read size or format does not match"));
        }
        let desc = Texture2dDesc {
            width: tex.width,
            height: tex.height,
            mip_levels: 1,
            array_size: 1,
            format: dxgi_format(TexFormat::Rgba8),
            sample: SampleDesc {
                count: 1,
                quality: 0,
            },
            usage: USAGE_STAGING,
            bind_flags: 0,
            cpu_access: CPU_ACCESS_READ,
            misc_flags: 0,
        };
        let mut made: Obj = null_mut();
        // SAFETY: slot 5 is CreateTexture2D, with a valid description and no
        // initial data.
        let staging = unsafe {
            let f = self
                .device
                .method::<CreateTexture2dFn>(DEV_CREATE_TEXTURE2D)?;
            let hr = f(self.device.raw(), &desc, null(), &mut made);
            created(hr, made, "CreateTexture2D (staging) failed")?
        };
        let ctx = &self.context;
        let mut mapped = MappedSubresource {
            data: null_mut(),
            row_pitch: 0,
            depth_pitch: 0,
        };
        // SAFETY: slot 47 is CopyResource and slot 14 is Map. Both textures
        // are live and have the same size and format, as CopyResource
        // requires. Map waits for the copy and fills `mapped`, which is
        // valid for writes.
        let hr = unsafe {
            let copy = ctx.method::<CopyResourceFn>(CTX_COPY_RESOURCE)?;
            let map = ctx.method::<MapFn>(CTX_MAP)?;
            copy(ctx.raw(), staging.raw(), tex.texture.raw());
            map(ctx.raw(), staging.raw(), 0, MAP_READ, 0, &mut mapped)
        };
        if hr < 0 {
            return Err(D3dError("Map failed"));
        }
        let pitch = mapped.row_pitch as usize;
        let result = if mapped.data.is_null() || pitch < row {
            Err(D3dError("Map gave an unusable pointer or pitch"))
        } else {
            for (y, dst) in out.chunks_exact_mut(row).enumerate() {
                // SAFETY: a successful Map of a staging texture gives
                // `height` rows of at least `row_pitch` bytes each, starting
                // at `data`, readable until Unmap. `pitch >= row` was
                // checked, so each slice stays inside its row.
                let src = unsafe {
                    std::slice::from_raw_parts(mapped.data.cast::<u8>().add(y * pitch), row)
                };
                dst.copy_from_slice(src);
            }
            Ok(())
        };
        // SAFETY: slot 15 is Unmap, for the subresource mapped above. No
        // slice of the mapped memory lives past this point.
        unsafe {
            let unmap = ctx.method::<UnmapFn>(CTX_UNMAP)?;
            unmap(ctx.raw(), staging.raw(), 0);
        }
        result
    }
}
