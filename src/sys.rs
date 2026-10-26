// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 Buntato
//! The Windows probe calls: `D3DCompile` and a throwaway Direct3D 11 device.
//!
//! This module, `sys_gl` (EGL and OpenGL on Linux), and `sys_d3d` (the
//! Direct3D 11 draw path on Windows) are the only modules allowed to use
//! `unsafe`. The rest of the crate is built with `#![deny(unsafe_code)]`.
//! Each block below says why it is sound.
//!
//! - `LoadLibraryA` and `GetProcAddress` for `d3dcompiler_47.dll`
//!   (`D3DCompile`) and `d3d11.dll` (`D3D11CreateDevice`). The probe device
//!   is released again right away.
//!
//! No call here keeps a pointer past the function that made it, and no call
//! hands out a raw pointer to the rest of the crate.

/// `D3DCOMPILE_IEEE_STRICTNESS`: no fast-math rewrites, for the draw path.
#[cfg(target_os = "windows")]
pub(crate) const COMPILE_IEEE_STRICT: u32 = 1 << 13;

/// Compile one HLSL entry point with `D3DCompile` from `d3dcompiler_47.dll`.
///
/// Returns the `DXBC` bytecode, or the compiler message on failure.
#[cfg(target_os = "windows")]
pub(crate) fn d3d_compile(source: &str, entry: &str, profile: &str) -> Result<Vec<u8>, String> {
    d3d_compile_with(source, entry, profile, 0)
}

/// [`d3d_compile`] with extra `D3DCOMPILE_*` flags added to the usual ones.
#[cfg(target_os = "windows")]
pub(crate) fn d3d_compile_with(
    source: &str,
    entry: &str,
    profile: &str,
    extra_flags: u32,
) -> Result<Vec<u8>, String> {
    use std::ffi::{CString, c_void};
    use std::ptr::{null, null_mut};

    #[repr(C)]
    struct BlobVtbl {
        query_interface: usize,
        add_ref: usize,
        release: unsafe extern "system" fn(*mut Blob) -> u32,
        get_buffer_pointer: unsafe extern "system" fn(*mut Blob) -> *const u8,
        get_buffer_size: unsafe extern "system" fn(*mut Blob) -> usize,
    }

    #[repr(C)]
    struct Blob {
        vtbl: *const BlobVtbl,
    }

    type D3DCompileFn = unsafe extern "system" fn(
        src_data: *const u8,
        src_data_size: usize,
        source_name: *const i8,
        defines: *const c_void,
        include: *const c_void,
        entry_point: *const i8,
        target: *const i8,
        flags1: u32,
        flags2: u32,
        code: *mut *mut Blob,
        error_msgs: *mut *mut Blob,
    ) -> i32;

    // SAFETY: these are the documented kernel32 signatures. They are only
    // called below with NUL-terminated names, and every result is checked
    // for null before use.
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn LoadLibraryA(name: *const i8) -> *mut c_void;
        fn GetProcAddress(module: *mut c_void, name: *const i8) -> *mut c_void;
    }

    // Copy a blob out and release it. A null blob gives an empty vector.
    //
    // SAFETY (for callers): `blob` must be null or a live ID3DBlob that the
    // caller owns one reference to. That reference is released here.
    unsafe fn take_blob(blob: *mut Blob) -> Vec<u8> {
        if blob.is_null() {
            return Vec::new();
        }
        // SAFETY: `blob` is a live ID3DBlob (see above). Its vtable starts with
        // the three IUnknown slots, then GetBufferPointer and GetBufferSize, as
        // declared in d3dcommon.h. The buffer is valid until Release, and it is
        // copied out before Release runs.
        unsafe {
            let ptr = ((*(*blob).vtbl).get_buffer_pointer)(blob);
            let len = ((*(*blob).vtbl).get_buffer_size)(blob);
            let bytes = if !ptr.is_null() && len > 0 {
                std::slice::from_raw_parts(ptr, len).to_vec()
            } else {
                Vec::new()
            };
            ((*(*blob).vtbl).release)(blob);
            bytes
        }
    }

    let c_entry = CString::new(entry).map_err(|_| "invalid entry point".to_string())?;
    let c_profile = CString::new(profile).map_err(|_| "invalid profile".to_string())?;
    // SAFETY: the names are NUL-terminated literals. The module is never freed,
    // so the function pointer below stays valid.
    let proc = unsafe {
        let dll = LoadLibraryA(c"d3dcompiler_47.dll".as_ptr());
        if dll.is_null() {
            return Err("d3dcompiler_47.dll not available".to_string());
        }
        GetProcAddress(dll, c"D3DCompile".as_ptr())
    };
    if proc.is_null() {
        return Err("D3DCompile symbol not found".to_string());
    }
    // SAFETY: `proc` is the non-null D3DCompile export, whose signature in
    // d3dcompiler.h matches `D3DCompileFn`.
    let d3d_compile = unsafe { std::mem::transmute::<*mut c_void, D3DCompileFn>(proc) };
    let mut code_blob: *mut Blob = null_mut();
    let mut err_blob: *mut Blob = null_mut();
    // SAFETY: the source pointer and length describe `source`, which outlives
    // the call. The name, entry, and profile strings are NUL-terminated and
    // outlive the call. Defines and include may be null. The two out pointers
    // are valid for writes. Flags are D3DCOMPILE_ENABLE_STRICTNESS (1 << 11)
    // and D3DCOMPILE_OPTIMIZATION_LEVEL3 (1 << 15), plus `extra_flags`, which
    // only change how the code is compiled.
    let hr = unsafe {
        d3d_compile(
            source.as_ptr(),
            source.len(),
            c"compositor.hlsl".as_ptr(),
            null(),
            null(),
            c_entry.as_ptr(),
            c_profile.as_ptr(),
            (1 << 11) | (1 << 15) | extra_flags,
            0,
            &mut code_blob,
            &mut err_blob,
        )
    };
    // SAFETY: D3DCompile hands back owned references (or null) in both out
    // pointers. Each is taken exactly once.
    let (code, errors) = unsafe { (take_blob(code_blob), take_blob(err_blob)) };
    if hr < 0 || code.is_empty() {
        if errors.is_empty() {
            return Err(format!("D3DCompile failed (HRESULT 0x{hr:08x})"));
        }
        return Err(String::from_utf8_lossy(&errors).trim().to_string());
    }
    if code.len() < 4 || &code[0..4] != b"DXBC" {
        return Err("compiled blob missing DXBC signature".to_string());
    }
    Ok(code)
}

/// Try to create a Direct3D 11 device: hardware first, then WARP.
///
/// Returns the negotiated feature level and a label for the driver type. The
/// device and context are released before returning.
#[cfg(target_os = "windows")]
pub(crate) fn d3d11_create_device() -> Option<(u32, &'static str)> {
    use std::ffi::c_void;
    use std::ptr::null_mut;

    #[repr(C)]
    struct UnknownVtbl {
        query_interface: usize,
        add_ref: usize,
        release: unsafe extern "system" fn(*mut Unknown) -> u32,
    }

    #[repr(C)]
    struct Unknown {
        vtbl: *const UnknownVtbl,
    }

    type D3D11CreateDeviceFn = unsafe extern "system" fn(
        adapter: *mut c_void,
        driver_type: u32,
        software: *mut c_void,
        flags: u32,
        feature_levels: *const u32,
        num_levels: u32,
        sdk_version: u32,
        device: *mut *mut Unknown,
        out_level: *mut u32,
        context: *mut *mut Unknown,
    ) -> i32;

    // SAFETY: these are the documented kernel32 signatures. They are only
    // called below with NUL-terminated names, and every result is checked
    // for null before use.
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn LoadLibraryA(name: *const i8) -> *mut c_void;
        fn GetProcAddress(module: *mut c_void, name: *const i8) -> *mut c_void;
    }

    // SAFETY: the names are NUL-terminated literals. The module is never freed,
    // so the function pointer below stays valid.
    let proc = unsafe {
        let dll = LoadLibraryA(c"d3d11.dll".as_ptr());
        if dll.is_null() {
            return None;
        }
        GetProcAddress(dll, c"D3D11CreateDevice".as_ptr())
    };
    if proc.is_null() {
        return None;
    }
    // SAFETY: `proc` is the non-null D3D11CreateDevice export, whose signature
    // in d3d11.h matches `D3D11CreateDeviceFn`.
    let create_device = unsafe { std::mem::transmute::<*mut c_void, D3D11CreateDeviceFn>(proc) };
    let levels = [0xb000u32];
    // Driver types 1 (hardware) and 5 (WARP) from D3D_DRIVER_TYPE.
    for &(driver_type, label) in &[
        (1u32, "Direct3D 11 Hardware (SM 5.0 DXBC)"),
        (5u32, "Direct3D 11 WARP (SM 5.0 DXBC)"),
    ] {
        let mut device: *mut Unknown = null_mut();
        let mut context: *mut Unknown = null_mut();
        let mut level = 0u32;
        // SAFETY: a null adapter and null software module are allowed. The
        // level array outlives the call. The out pointers are valid for writes.
        // 7 is D3D11_SDK_VERSION.
        let hr = unsafe {
            create_device(
                null_mut(),
                driver_type,
                null_mut(),
                0,
                levels.as_ptr(),
                1,
                7,
                &mut device,
                &mut level,
                &mut context,
            )
        };
        if hr >= 0 && !device.is_null() {
            // SAFETY: on success both objects are owned COM references whose
            // vtables start with IUnknown. Each is released exactly once.
            unsafe {
                if !context.is_null() {
                    ((*(*context).vtbl).release)(context);
                }
                ((*(*device).vtbl).release)(device);
            }
            return Some((level, label));
        }
    }
    None
}
