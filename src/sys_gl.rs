// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 Buntato
//! EGL and OpenGL calls for the Linux device backend.
//!
//! Like `sys.rs`, this module may use `unsafe`; the rest of the crate may not.
//! Nothing here is linked at build time. `libEGL.so.1` is opened with
//! `dlopen` the first time it is needed, and GL entry points come from
//! `eglGetProcAddress` (or, when that returns null, from `libOpenGL.so.0` or
//! `libGL.so.1`). Each unsafe block says why it is sound.
//!
//! The safe surface is small: [`egl_display`] opens one EGL display per
//! process, [`GlContext::create`] makes an offscreen OpenGL 3.3 core context,
//! and [`GlContext::with_current`] runs a closure with that context current
//! on the calling thread, handing it a [`GlCurrent`] whose methods upload
//! textures, build programs, draw into a texture, and read pixels back. Every
//! size and length that reaches a raw pointer is checked first, and textures
//! are tracked with their size and format so a read can never write past the
//! caller's buffer.
//!
//! EGL entry points: `eglGetProcAddress`, `eglQueryString`, `eglGetDisplay`,
//! `eglGetPlatformDisplayEXT`, `eglInitialize`, `eglBindAPI`, `eglQueryAPI`,
//! `eglChooseConfig`, `eglCreateContext`, `eglDestroyContext`,
//! `eglCreatePbufferSurface`, `eglDestroySurface`, `eglMakeCurrent`,
//! `eglGetCurrentContext`, `eglGetCurrentDisplay`, `eglGetCurrentSurface`.
//!
//! GL entry points are listed once, in the `gl_api!` table below.

use std::ffi::{CStr, CString, c_char, c_int, c_uint, c_void};
use std::marker::PhantomData;
use std::ptr::{null, null_mut};
use std::sync::{Mutex, MutexGuard, OnceLock};

use crate::gpu::TexFormat;

// Declarations from <dlfcn.h>. build.rs links libdl.
unsafe extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}

const RTLD_NOW: c_int = 2;

// EGL constants from the EGL 1.5 headers and the extensions named.
const EGL_EXTENSIONS: c_int = 0x3055;
const EGL_NONE: c_int = 0x3038;
const EGL_SURFACE_TYPE: c_int = 0x3033;
const EGL_PBUFFER_BIT: c_int = 0x0001;
const EGL_RENDERABLE_TYPE: c_int = 0x3040;
const EGL_OPENGL_BIT: c_int = 0x0008;
const EGL_OPENGL_API: c_uint = 0x30A2;
const EGL_WIDTH: c_int = 0x3057;
const EGL_HEIGHT: c_int = 0x3056;
const EGL_CONTEXT_MAJOR_VERSION: c_int = 0x3098;
const EGL_CONTEXT_MINOR_VERSION: c_int = 0x30FB;
const EGL_CONTEXT_OPENGL_PROFILE_MASK: c_int = 0x30FD;
const EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT: c_int = 0x0001;
const EGL_DRAW: c_int = 0x3059;
const EGL_READ: c_int = 0x305A;
// EGL_MESA_platform_surfaceless.
const EGL_PLATFORM_SURFACELESS_MESA: c_uint = 0x31DD;

// GL constants from the OpenGL 3.3 core specification.
const GL_NO_ERROR: u32 = 0;
const GL_TEXTURE_2D: u32 = 0x0DE1;
const GL_TEXTURE0: u32 = 0x84C0;
const GL_TEXTURE_MIN_FILTER: u32 = 0x2801;
const GL_TEXTURE_MAG_FILTER: u32 = 0x2800;
const GL_TEXTURE_WRAP_S: u32 = 0x2802;
const GL_TEXTURE_WRAP_T: u32 = 0x2803;
const GL_NEAREST: i32 = 0x2600;
const GL_CLAMP_TO_EDGE: i32 = 0x812F;
const GL_UNPACK_ALIGNMENT: u32 = 0x0CF5;
const GL_PACK_ALIGNMENT: u32 = 0x0D05;
const GL_MAX_TEXTURE_SIZE: u32 = 0x0D33;
const GL_MAX_VIEWPORT_DIMS: u32 = 0x0D3A;
const GL_FRAMEBUFFER: u32 = 0x8D40;
const GL_COLOR_ATTACHMENT0: u32 = 0x8CE0;
const GL_FRAMEBUFFER_COMPLETE: u32 = 0x8CD5;
const GL_TRIANGLE_STRIP: u32 = 0x0005;
const GL_VERTEX_SHADER: u32 = 0x8B31;
const GL_FRAGMENT_SHADER: u32 = 0x8B30;
const GL_COMPILE_STATUS: u32 = 0x8B81;
const GL_LINK_STATUS: u32 = 0x8B82;
const GL_RGBA: u32 = 0x1908;
const GL_RED: u32 = 0x1903;
const GL_RG: u32 = 0x8227;
const GL_RGBA8: i32 = 0x8058;
const GL_R8: i32 = 0x8229;
const GL_RG8: i32 = 0x822B;
const GL_R32F: i32 = 0x822E;
const GL_UNSIGNED_BYTE: u32 = 0x1401;
const GL_FLOAT: u32 = 0x1406;

/// Most texture units a draw binds. GL 3.3 guarantees 16 for fragment shaders.
const MAX_INPUTS: usize = 8;

/// Why a GL call did not happen or did not work.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GlError(pub &'static str);

/// An EGL handle (display, context, config, or surface).
#[derive(Clone, Copy, Debug)]
struct Handle(*mut c_void);

// SAFETY: EGL handles are opaque process-wide names, not pointers into
// thread-local data. EGL 1.4 and later require every entry point to be
// callable from any thread. The one thread-bound thing, a current context,
// is only made current while GL_LOCK is held and is released before the lock
// is given up (see GlContext::with_current).
unsafe impl Send for Handle {}
// SAFETY: as above. A shared Handle is only read.
unsafe impl Sync for Handle {}

type GetProcFn = unsafe extern "C" fn(*const c_char) -> *mut c_void;

/// EGL entry points. Plain function pointers are `Send` and `Sync`.
struct Egl {
    get_proc_address: GetProcFn,
    query_string: unsafe extern "C" fn(*mut c_void, c_int) -> *const c_char,
    get_display: unsafe extern "C" fn(*mut c_void) -> *mut c_void,
    initialize: unsafe extern "C" fn(*mut c_void, *mut c_int, *mut c_int) -> c_uint,
    bind_api: unsafe extern "C" fn(c_uint) -> c_uint,
    query_api: unsafe extern "C" fn() -> c_uint,
    choose_config: unsafe extern "C" fn(
        *mut c_void,
        *const c_int,
        *mut *mut c_void,
        c_int,
        *mut c_int,
    ) -> c_uint,
    create_context:
        unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void, *const c_int) -> *mut c_void,
    destroy_context: unsafe extern "C" fn(*mut c_void, *mut c_void) -> c_uint,
    create_pbuffer_surface:
        unsafe extern "C" fn(*mut c_void, *mut c_void, *const c_int) -> *mut c_void,
    destroy_surface: unsafe extern "C" fn(*mut c_void, *mut c_void) -> c_uint,
    make_current:
        unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void, *mut c_void) -> c_uint,
    get_current_context: unsafe extern "C" fn() -> *mut c_void,
    get_current_display: unsafe extern "C" fn() -> *mut c_void,
    get_current_surface: unsafe extern "C" fn(c_int) -> *mut c_void,
}

/// Look up `name` in `handle` and cast it to the function pointer type `F`.
///
/// SAFETY (for callers): `handle` must come from a successful `dlopen` that is
/// never closed, and `F` must be an `unsafe extern "C" fn` type matching the
/// C declaration of `name`.
unsafe fn symbol<F: Copy>(handle: *mut c_void, name: &CStr) -> Option<F> {
    // SAFETY: `handle` is a live library handle (see above) and `name` is
    // NUL-terminated.
    let p = unsafe { dlsym(handle, name.as_ptr()) };
    if p.is_null() {
        return None;
    }
    debug_assert_eq!(size_of::<F>(), size_of::<*mut c_void>());
    // SAFETY: `p` is a non-null export named `name`; the caller promises that
    // `F` is its function pointer type. On Linux, function and data pointers
    // have the same size and representation.
    Some(unsafe { std::mem::transmute_copy::<*mut c_void, F>(&p) })
}

impl Egl {
    fn load() -> Option<Egl> {
        // SAFETY: the name is a NUL-terminated literal and dlopen does not
        // keep it. A null return is handled. The handle is never closed:
        // unloading a GL driver that may still own threads is not safe.
        let lib = unsafe { dlopen(c"libEGL.so.1".as_ptr(), RTLD_NOW) };
        if lib.is_null() {
            return None;
        }
        // SAFETY: `lib` is never closed, and every type below is the EGL 1.5
        // C declaration of that name (EGLDisplay, EGLConfig, EGLContext,
        // EGLSurface are pointers, EGLint is int, EGLenum and EGLBoolean are
        // unsigned int).
        unsafe {
            Some(Egl {
                get_proc_address: symbol(lib, c"eglGetProcAddress")?,
                query_string: symbol(lib, c"eglQueryString")?,
                get_display: symbol(lib, c"eglGetDisplay")?,
                initialize: symbol(lib, c"eglInitialize")?,
                bind_api: symbol(lib, c"eglBindAPI")?,
                query_api: symbol(lib, c"eglQueryAPI")?,
                choose_config: symbol(lib, c"eglChooseConfig")?,
                create_context: symbol(lib, c"eglCreateContext")?,
                destroy_context: symbol(lib, c"eglDestroyContext")?,
                create_pbuffer_surface: symbol(lib, c"eglCreatePbufferSurface")?,
                destroy_surface: symbol(lib, c"eglDestroySurface")?,
                make_current: symbol(lib, c"eglMakeCurrent")?,
                get_current_context: symbol(lib, c"eglGetCurrentContext")?,
                get_current_display: symbol(lib, c"eglGetCurrentDisplay")?,
                get_current_surface: symbol(lib, c"eglGetCurrentSurface")?,
            })
        }
    }

    /// Extension string of `display` (null for client extensions), or "".
    fn extensions(&self, display: *mut c_void) -> String {
        // SAFETY: eglQueryString accepts EGL_NO_DISPLAY (null) for client
        // extensions and an initialized display otherwise. It returns null or
        // a NUL-terminated string owned by EGL that lives as long as the
        // display; it is copied out right away.
        unsafe {
            let p = (self.query_string)(display, EGL_EXTENSIONS);
            if p.is_null() {
                String::new()
            } else {
                CStr::from_ptr(p).to_string_lossy().into_owned()
            }
        }
    }
}

fn has_extension(list: &str, name: &str) -> bool {
    list.split_ascii_whitespace().any(|e| e == name)
}

/// The EGL display this process uses, opened on first use.
pub(crate) struct EglDisplay {
    egl: Egl,
    display: Handle,
    /// EGL version from `eglInitialize`.
    pub version: (i32, i32),
    /// Contexts can be made current without a surface.
    surfaceless_context: bool,
}

/// The EGL display for this process, or `None` when there is none.
///
/// The surfaceless platform (`EGL_MESA_platform_surfaceless`) is tried first,
/// since it needs no display server. Otherwise the default display is used,
/// which on a desktop connects to the display server. Either way this runs
/// once per process. The display is never terminated: `eglTerminate` would
/// pull it from under any other user in the same process.
pub(crate) fn egl_display() -> Option<&'static EglDisplay> {
    static DISPLAY: OnceLock<Option<EglDisplay>> = OnceLock::new();
    DISPLAY.get_or_init(open_display).as_ref()
}

fn open_display() -> Option<EglDisplay> {
    let egl = Egl::load()?;
    let client = egl.extensions(null_mut());
    let initialize = |display: *mut c_void| -> Option<(i32, i32)> {
        if display.is_null() {
            return None;
        }
        let (mut major, mut minor): (c_int, c_int) = (0, 0);
        // SAFETY: `display` is a non-null EGLDisplay from eglGetDisplay or
        // eglGetPlatformDisplayEXT. Both out pointers are valid for writes for
        // the whole call and are not kept.
        let ok = unsafe { (egl.initialize)(display, &mut major, &mut minor) };
        (ok != 0).then_some((major, minor))
    };
    // (display, version), surfaceless platform first.
    let mut found = None;
    if has_extension(&client, "EGL_EXT_platform_base")
        && has_extension(&client, "EGL_MESA_platform_surfaceless")
    {
        type PlatformFn = unsafe extern "C" fn(c_uint, *mut c_void, *const c_int) -> *mut c_void;
        // SAFETY: the name is NUL-terminated. A null return is handled.
        let p = unsafe { (egl.get_proc_address)(c"eglGetPlatformDisplayEXT".as_ptr()) };
        if !p.is_null() {
            // SAFETY: `p` is the non-null eglGetPlatformDisplayEXT entry
            // point from EGL_EXT_platform_base, declared exactly as PlatformFn.
            let get_platform = unsafe { std::mem::transmute::<*mut c_void, PlatformFn>(p) };
            // SAFETY: the surfaceless platform takes EGL_DEFAULT_DISPLAY (null)
            // as its native display, and the attribute list may be null.
            let display =
                unsafe { get_platform(EGL_PLATFORM_SURFACELESS_MESA, null_mut(), null()) };
            if let Some(version) = initialize(display) {
                found = Some((display, version));
            }
        }
    }
    if found.is_none() {
        // SAFETY: EGL_DEFAULT_DISPLAY is a null native display. A null
        // return (EGL_NO_DISPLAY) is handled by `initialize`.
        let display = unsafe { (egl.get_display)(null_mut()) };
        if let Some(version) = initialize(display) {
            found = Some((display, version));
        }
    }
    let (display, version) = found?;
    let surfaceless_context =
        has_extension(&egl.extensions(display), "EGL_KHR_surfaceless_context");
    Some(EglDisplay {
        egl,
        display: Handle(display),
        version,
        surfaceless_context,
    })
}

/// Held while any context made here is current on some thread.
static GL_LOCK: Mutex<()> = Mutex::new(());

fn gl_lock() -> MutexGuard<'static, ()> {
    // A panic while drawing leaves no GL state that the next caller relies
    // on (each call sets what it uses), so a poisoned lock is still usable.
    GL_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// The thread's EGL rendering API, switched to desktop OpenGL and put back
/// on drop, so a caller that uses OpenGL ES on the same thread keeps it.
struct ApiGuard<'d> {
    egl: &'d Egl,
    api: c_uint,
}

impl<'d> ApiGuard<'d> {
    fn bind(egl: &'d Egl) -> Result<Self, GlError> {
        // SAFETY: eglQueryAPI takes nothing and returns this thread's API.
        let api = unsafe { (egl.query_api)() };
        // SAFETY: eglBindAPI takes a plain enum and only changes this
        // thread's EGL state, which the guard puts back.
        if unsafe { (egl.bind_api)(EGL_OPENGL_API) } == 0 {
            return Err(GlError("EGL has no desktop OpenGL"));
        }
        Ok(ApiGuard { egl, api })
    }
}

impl Drop for ApiGuard<'_> {
    fn drop(&mut self) {
        // SAFETY: as above; `api` is the value eglQueryAPI gave a moment
        // ago, so it is one this EGL accepts. A failure leaves OpenGL bound.
        unsafe { (self.egl.bind_api)(self.api) };
    }
}

/// Whatever was current on this thread before, put back on drop. The
/// rendering API is put back after that (field order).
struct CurrentGuard<'d> {
    d: &'d EglDisplay,
    display: *mut c_void,
    draw: *mut c_void,
    read: *mut c_void,
    context: *mut c_void,
    _api: ApiGuard<'d>,
}

impl<'d> CurrentGuard<'d> {
    /// Make `context` current with `surface`, remembering what was current.
    fn make(d: &'d EglDisplay, surface: Handle, context: Handle) -> Result<Self, GlError> {
        let egl = &d.egl;
        // The current context is per API, so switch to OpenGL first.
        let api = ApiGuard::bind(egl)?;
        // SAFETY: the eglGetCurrent* calls take no pointers and return
        // handles (or null) for this thread and its OpenGL API.
        let (display, draw, read, current) = unsafe {
            (
                (egl.get_current_display)(),
                (egl.get_current_surface)(EGL_DRAW),
                (egl.get_current_surface)(EGL_READ),
                (egl.get_current_context)(),
            )
        };
        // SAFETY: the display is initialized, and `context` and `surface`
        // (null when surfaceless) were created on it and are still alive:
        // they are only destroyed in GlContext::drop, which needs `&mut`.
        let ok = unsafe { (egl.make_current)(d.display.0, surface.0, surface.0, context.0) };
        if ok == 0 {
            // No context changed; dropping `api` puts the API back.
            return Err(GlError("eglMakeCurrent failed"));
        }
        Ok(CurrentGuard {
            d,
            display,
            draw,
            read,
            context: current,
            _api: api,
        })
    }
}

impl Drop for CurrentGuard<'_> {
    fn drop(&mut self) {
        let egl = &self.d.egl;
        // SAFETY: restores the handles read in `make`, which were current on
        // this thread a moment ago, or releases this thread's context when
        // nothing was current. Both are valid eglMakeCurrent calls; a failure
        // leaves our context current, which is harmless because every use
        // makes it current again first.
        unsafe {
            if self.context.is_null() || self.display.is_null() {
                (egl.make_current)(self.d.display.0, null_mut(), null_mut(), null_mut());
            } else {
                (egl.make_current)(self.display, self.draw, self.read, self.context);
            }
        }
    }
}

macro_rules! gl_api {
    ($($field:ident = $name:literal: fn($($arg:ty),*) $(-> $ret:ty)?;)*) => {
        /// OpenGL 3.3 core entry points.
        struct Gl {
            $($field: unsafe extern "C" fn($($arg),*) $(-> $ret)?,)*
        }

        impl Gl {
            /// Look up every entry point. Needs a current context, because
            /// some drivers only answer then.
            fn load(egl: &Egl, fallback: *mut c_void) -> Option<Gl> {
                Some(Gl {
                    $($field: {
                        let name: &CStr = $name;
                        // SAFETY: the name is NUL-terminated. A null return
                        // is handled.
                        let mut p = unsafe { (egl.get_proc_address)(name.as_ptr()) };
                        if p.is_null() && !fallback.is_null() {
                            // SAFETY: `fallback` is a library handle that is
                            // never closed; the name is NUL-terminated.
                            p = unsafe { dlsym(fallback, name.as_ptr()) };
                        }
                        if p.is_null() {
                            return None;
                        }
                        // SAFETY: `p` is the non-null entry point of that
                        // name, and the type is its C declaration in the
                        // OpenGL 3.3 core headers (GLenum, GLuint, GLbitfield
                        // are u32, GLint and GLsizei are i32, GLboolean is u8,
                        // GLfloat is f32, GLsizeiptr is isize).
                        unsafe {
                            std::mem::transmute::<*mut c_void, unsafe extern "C" fn($($arg),*) $(-> $ret)?>(p)
                        }
                    },)*
                })
            }
        }
    };
}

gl_api! {
    get_error = c"glGetError": fn() -> u32;
    get_integerv = c"glGetIntegerv": fn(u32, *mut i32);
    pixel_storei = c"glPixelStorei": fn(u32, i32);
    viewport = c"glViewport": fn(i32, i32, i32, i32);
    create_shader = c"glCreateShader": fn(u32) -> u32;
    shader_source = c"glShaderSource": fn(u32, i32, *const *const c_char, *const i32);
    compile_shader = c"glCompileShader": fn(u32);
    get_shaderiv = c"glGetShaderiv": fn(u32, u32, *mut i32);
    get_shader_info_log = c"glGetShaderInfoLog": fn(u32, i32, *mut i32, *mut c_char);
    delete_shader = c"glDeleteShader": fn(u32);
    create_program = c"glCreateProgram": fn() -> u32;
    attach_shader = c"glAttachShader": fn(u32, u32);
    link_program = c"glLinkProgram": fn(u32);
    get_programiv = c"glGetProgramiv": fn(u32, u32, *mut i32);
    get_program_info_log = c"glGetProgramInfoLog": fn(u32, i32, *mut i32, *mut c_char);
    delete_program = c"glDeleteProgram": fn(u32);
    use_program = c"glUseProgram": fn(u32);
    get_uniform_location = c"glGetUniformLocation": fn(u32, *const c_char) -> i32;
    uniform1i = c"glUniform1i": fn(i32, i32);
    uniform2i = c"glUniform2i": fn(i32, i32, i32);
    uniform4i = c"glUniform4i": fn(i32, i32, i32, i32, i32);
    uniform1f = c"glUniform1f": fn(i32, f32);
    uniform2f = c"glUniform2f": fn(i32, f32, f32);
    uniform4f = c"glUniform4f": fn(i32, f32, f32, f32, f32);
    uniform1fv = c"glUniform1fv": fn(i32, i32, *const f32);
    uniform4fv = c"glUniform4fv": fn(i32, i32, *const f32);
    gen_textures = c"glGenTextures": fn(i32, *mut u32);
    delete_textures = c"glDeleteTextures": fn(i32, *const u32);
    bind_texture = c"glBindTexture": fn(u32, u32);
    active_texture = c"glActiveTexture": fn(u32);
    tex_parameteri = c"glTexParameteri": fn(u32, u32, i32);
    tex_image_2d = c"glTexImage2D": fn(u32, i32, i32, i32, i32, i32, u32, u32, *const c_void);
    gen_framebuffers = c"glGenFramebuffers": fn(i32, *mut u32);
    delete_framebuffers = c"glDeleteFramebuffers": fn(i32, *const u32);
    bind_framebuffer = c"glBindFramebuffer": fn(u32, u32);
    framebuffer_texture_2d = c"glFramebufferTexture2D": fn(u32, u32, u32, u32, i32);
    check_framebuffer_status = c"glCheckFramebufferStatus": fn(u32) -> u32;
    gen_vertex_arrays = c"glGenVertexArrays": fn(i32, *mut u32);
    delete_vertex_arrays = c"glDeleteVertexArrays": fn(i32, *const u32);
    bind_vertex_array = c"glBindVertexArray": fn(u32);
    draw_arrays = c"glDrawArrays": fn(u32, i32, i32);
    read_pixels = c"glReadPixels": fn(i32, i32, i32, i32, u32, u32, *mut c_void);
}

/// Library to look GL entry points up in when `eglGetProcAddress` does not
/// answer for core functions (EGL before 1.5 without
/// `EGL_KHR_get_all_proc_addresses`). Null when neither is there.
fn gl_library() -> *mut c_void {
    for name in [c"libOpenGL.so.0", c"libGL.so.1"] {
        // SAFETY: NUL-terminated literal; a null return is handled. The
        // handle is never closed.
        let lib = unsafe { dlopen(name.as_ptr(), RTLD_NOW) };
        if !lib.is_null() {
            return lib;
        }
    }
    null_mut()
}

/// What to ask EGL for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ContextRequest {
    pub major: i32,
    pub minor: i32,
}

impl ContextRequest {
    /// OpenGL 3.3 core, which `shaders/passes.glsl` is written for.
    pub const GL33: ContextRequest = ContextRequest { major: 3, minor: 3 };
}

/// A texture made by [`GlCurrent::texture`], by index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GlTexture(usize);

/// A linked program made by [`GlCurrent::program`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GlProgram(u32);

#[derive(Clone, Copy, Debug)]
struct TexInfo {
    name: u32,
    width: u32,
    height: u32,
    format: TexFormat,
}

/// An offscreen OpenGL context with no window.
pub(crate) struct GlContext {
    display: &'static EglDisplay,
    context: Handle,
    // Null when the context runs without a surface.
    surface: Handle,
    gl: Gl,
    vao: u32,
    fbo: u32,
    max_side: u32,
    renderer: String,
    textures: Vec<TexInfo>,
    programs: Vec<u32>,
}

impl std::fmt::Debug for GlContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GlContext")
            .field("renderer", &self.renderer)
            .field("max_side", &self.max_side)
            .finish()
    }
}

impl GlContext {
    /// Make a context and set up the little state every draw shares.
    pub(crate) fn create(request: ContextRequest) -> Result<GlContext, GlError> {
        let d = egl_display().ok_or(GlError("no EGL display"))?;
        let egl = &d.egl;
        let _lock = gl_lock();
        // eglChooseConfig and eglCreateContext below use the thread's
        // rendering API; it is put back when this returns.
        let _api = ApiGuard::bind(egl)?;
        let surface_bits = if d.surfaceless_context {
            0
        } else {
            EGL_PBUFFER_BIT
        };
        let config_attrs = [
            EGL_RENDERABLE_TYPE,
            EGL_OPENGL_BIT,
            EGL_SURFACE_TYPE,
            surface_bits,
            EGL_NONE,
        ];
        let mut config: *mut c_void = null_mut();
        let mut count: c_int = 0;
        // SAFETY: the attribute list ends in EGL_NONE and outlives the call.
        // Room for exactly one config is passed, with its size, and `count`
        // is valid for writes.
        let ok = unsafe {
            (egl.choose_config)(
                d.display.0,
                config_attrs.as_ptr(),
                &mut config,
                1,
                &mut count,
            )
        };
        if ok == 0 || count < 1 || config.is_null() {
            return Err(GlError("no EGL config for OpenGL"));
        }
        let context_attrs = [
            EGL_CONTEXT_MAJOR_VERSION,
            request.major,
            EGL_CONTEXT_MINOR_VERSION,
            request.minor,
            EGL_CONTEXT_OPENGL_PROFILE_MASK,
            EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,
            EGL_NONE,
        ];
        // SAFETY: `config` came from eglChooseConfig on this display, the
        // share context is EGL_NO_CONTEXT (null), and the attribute list ends
        // in EGL_NONE and outlives the call.
        let context = unsafe {
            (egl.create_context)(d.display.0, config, null_mut(), context_attrs.as_ptr())
        };
        if context.is_null() {
            return Err(GlError("no OpenGL core context of the requested version"));
        }
        let surface = if d.surfaceless_context {
            null_mut()
        } else {
            let attrs = [EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE];
            // SAFETY: same display and config; the list ends in EGL_NONE.
            let s = unsafe { (egl.create_pbuffer_surface)(d.display.0, config, attrs.as_ptr()) };
            if s.is_null() {
                // SAFETY: `context` was created above and is not current.
                unsafe { (egl.destroy_context)(d.display.0, context) };
                return Err(GlError("no pbuffer surface"));
            }
            s
        };
        let (context, surface) = (Handle(context), Handle(surface));
        let destroy = || {
            // SAFETY: both were created above on this display and are not
            // current (the guard has put back the previous state).
            unsafe {
                if !surface.0.is_null() {
                    (egl.destroy_surface)(d.display.0, surface.0);
                }
                (egl.destroy_context)(d.display.0, context.0);
            }
        };
        let setup = {
            let current = match CurrentGuard::make(d, surface, context) {
                Ok(c) => c,
                Err(e) => {
                    destroy();
                    return Err(e);
                }
            };
            let set_up = Gl::load(egl, gl_library()).map(|gl| {
                let (mut vao, mut fbo) = (0u32, 0u32);
                let mut max_texture = 0i32;
                let mut max_viewport = [0i32; 2];
                // SAFETY: the context is current on this thread (guard
                // above). Each out pointer is valid for the number of values
                // the call writes: one name each, one int for
                // GL_MAX_TEXTURE_SIZE, two for GL_MAX_VIEWPORT_DIMS.
                let error = unsafe {
                    (gl.gen_vertex_arrays)(1, &mut vao);
                    (gl.bind_vertex_array)(vao);
                    (gl.gen_framebuffers)(1, &mut fbo);
                    (gl.pixel_storei)(GL_UNPACK_ALIGNMENT, 1);
                    (gl.pixel_storei)(GL_PACK_ALIGNMENT, 1);
                    (gl.get_integerv)(GL_MAX_TEXTURE_SIZE, &mut max_texture);
                    (gl.get_integerv)(GL_MAX_VIEWPORT_DIMS, max_viewport.as_mut_ptr());
                    (gl.get_error)()
                };
                let max_side = max_texture.min(max_viewport[0]).min(max_viewport[1]).max(0) as u32;
                (gl, vao, fbo, max_side, error)
            });
            drop(current);
            set_up
        };
        let Some((gl, vao, fbo, max_side, error)) = setup else {
            destroy();
            return Err(GlError("an OpenGL 3.3 entry point is missing"));
        };
        if error != GL_NO_ERROR || max_side == 0 {
            destroy();
            return Err(GlError("OpenGL setup failed"));
        }
        let renderer = format!("EGL {}.{}", d.version.0, d.version.1);
        Ok(GlContext {
            display: d,
            context,
            surface,
            gl,
            vao,
            fbo,
            max_side,
            renderer,
            textures: Vec::new(),
            programs: Vec::new(),
        })
    }

    /// Run `f` with this context current on the calling thread.
    ///
    /// Only one context made here is current anywhere at a time. Textures
    /// made inside `f` are deleted before this returns, and whatever context
    /// the thread had current before is put back.
    pub(crate) fn with_current<R>(
        &mut self,
        f: impl FnOnce(&mut GlCurrent<'_>) -> R,
    ) -> Result<R, GlError> {
        let _lock = gl_lock();
        let guard = CurrentGuard::make(self.display, self.surface, self.context)?;
        let mut current = GlCurrent {
            ctx: self,
            _not_send: PhantomData,
        };
        // Left over if an earlier closure panicked.
        current.delete_textures();
        let out = f(&mut current);
        current.delete_textures();
        drop(guard);
        Ok(out)
    }
}

impl Drop for GlContext {
    fn drop(&mut self) {
        let d = self.display;
        let _lock = gl_lock();
        if let Ok(guard) = CurrentGuard::make(d, self.surface, self.context) {
            let names: Vec<u32> = self.textures.drain(..).map(|t| t.name).collect();
            // SAFETY: the context is current. Every name was made in it, the
            // name slice outlives the call and its length is passed.
            unsafe {
                for &p in &self.programs {
                    (self.gl.delete_program)(p);
                }
                if !names.is_empty() {
                    (self.gl.delete_textures)(names.len() as i32, names.as_ptr());
                }
                (self.gl.delete_framebuffers)(1, &self.fbo);
                (self.gl.delete_vertex_arrays)(1, &self.vao);
            }
            drop(guard);
        }
        // SAFETY: both were made in `create` on this display, and the guard
        // above has made this context not current again. Destroying a
        // context also frees anything the deletes above missed.
        unsafe {
            if !self.surface.0.is_null() {
                (d.egl.destroy_surface)(d.display.0, self.surface.0);
            }
            (d.egl.destroy_context)(d.display.0, self.context.0);
        }
    }
}

// SAFETY: a GlContext holds EGL handles (see Handle) and GL names, which are
// plain integers. GL calls only happen through GlCurrent, which exists only
// inside with_current, where GL_LOCK is held and the context is current on
// the calling thread. So moving a GlContext to another thread is fine.
unsafe impl Send for GlContext {}

/// A value for one uniform.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Uniform<'a> {
    I1(i32),
    I2([i32; 2]),
    I4([i32; 4]),
    F1(f32),
    F2([f32; 2]),
    F4([f32; 4]),
    F1v(&'a [f32]),
    F4v(&'a [[f32; 4]]),
}

/// A [`GlContext`] that is current on this thread. Not `Send`.
pub(crate) struct GlCurrent<'c> {
    ctx: &'c mut GlContext,
    _not_send: PhantomData<*const ()>,
}

impl GlCurrent<'_> {
    /// Largest texture or render target side this context takes.
    pub(crate) fn max_side(&self) -> u32 {
        self.ctx.max_side
    }

    fn check(&self, what: &'static str) -> Result<(), GlError> {
        let gl = &self.ctx.gl;
        // SAFETY: the context is current (GlCurrent only exists inside
        // with_current). glGetError takes no arguments.
        let first = unsafe { (gl.get_error)() };
        if first == GL_NO_ERROR {
            return Ok(());
        }
        // Clear any other flags so the next check starts clean.
        for _ in 0..8 {
            // SAFETY: as above.
            if unsafe { (gl.get_error)() } == GL_NO_ERROR {
                break;
            }
        }
        Err(GlError(what))
    }

    fn compile(&mut self, kind: u32, source: &str) -> Result<u32, String> {
        let text = CString::new(source).map_err(|_| "shader source has a NUL byte".to_string())?;
        let gl = &self.ctx.gl;
        let ptr = text.as_ptr();
        let mut status = 0i32;
        // SAFETY: the context is current. glShaderSource gets one string,
        // NUL-terminated (null length array), through a pointer that is valid
        // for the call; GL copies the text. `status` is valid for one write.
        let shader = unsafe {
            let shader = (gl.create_shader)(kind);
            if shader == 0 {
                return Err("glCreateShader failed".to_string());
            }
            (gl.shader_source)(shader, 1, &ptr, null());
            (gl.compile_shader)(shader);
            (gl.get_shaderiv)(shader, GL_COMPILE_STATUS, &mut status);
            shader
        };
        if status == 0 {
            let mut log = vec![0u8; 4096];
            let mut len = 0i32;
            // SAFETY: the buffer is 4096 bytes and that size is passed, so GL
            // writes at most that many; `len` is valid for one write. The
            // shader was made above in this context.
            unsafe {
                (gl.get_shader_info_log)(
                    shader,
                    log.len() as i32,
                    &mut len,
                    log.as_mut_ptr().cast(),
                );
                (gl.delete_shader)(shader);
            }
            log.truncate(len.clamp(0, 4096) as usize);
            return Err(String::from_utf8_lossy(&log).trim().to_string());
        }
        Ok(shader)
    }

    /// Compile and link a program. The program lives as long as the context.
    pub(crate) fn program(&mut self, vertex: &str, fragment: &str) -> Result<GlProgram, String> {
        let vs = self.compile(GL_VERTEX_SHADER, vertex)?;
        let fs = match self.compile(GL_FRAGMENT_SHADER, fragment) {
            Ok(fs) => fs,
            Err(e) => {
                // SAFETY: the context is current and `vs` was made in it.
                unsafe { (self.ctx.gl.delete_shader)(vs) };
                return Err(e);
            }
        };
        let gl = &self.ctx.gl;
        let mut status = 0i32;
        // SAFETY: the context is current, both shaders were made in it, and
        // `status` is valid for one write. Deleting attached shaders only
        // flags them; they go away with the program.
        let program = unsafe {
            let program = (gl.create_program)();
            if program != 0 {
                (gl.attach_shader)(program, vs);
                (gl.attach_shader)(program, fs);
                (gl.link_program)(program);
                (gl.get_programiv)(program, GL_LINK_STATUS, &mut status);
            }
            (gl.delete_shader)(vs);
            (gl.delete_shader)(fs);
            program
        };
        if program == 0 {
            return Err("glCreateProgram failed".to_string());
        }
        if status == 0 {
            let mut log = vec![0u8; 4096];
            let mut len = 0i32;
            // SAFETY: as for the shader log above.
            unsafe {
                (gl.get_program_info_log)(
                    program,
                    log.len() as i32,
                    &mut len,
                    log.as_mut_ptr().cast(),
                );
                (gl.delete_program)(program);
            }
            log.truncate(len.clamp(0, 4096) as usize);
            return Err(String::from_utf8_lossy(&log).trim().to_string());
        }
        self.ctx.programs.push(program);
        Ok(GlProgram(program))
    }

    /// Make a `width` by `height` texture, filled from `data` (rows top
    /// first, no padding) or left undefined for a render target.
    pub(crate) fn texture(
        &mut self,
        format: TexFormat,
        width: u32,
        height: u32,
        data: Option<&[u8]>,
    ) -> Result<GlTexture, GlError> {
        if width == 0 || height == 0 || width > self.ctx.max_side || height > self.ctx.max_side {
            return Err(GlError("texture size out of range"));
        }
        let len = (width as usize)
            .checked_mul(height as usize)
            .and_then(|n| n.checked_mul(format.texel_bytes()))
            .ok_or(GlError("texture size out of range"))?;
        if data.is_some_and(|d| d.len() != len) {
            return Err(GlError("texture data has the wrong length"));
        }
        let (internal, layout, kind) = match format {
            TexFormat::Rgba8 => (GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE),
            TexFormat::R8 => (GL_R8, GL_RED, GL_UNSIGNED_BYTE),
            TexFormat::Rg8 => (GL_RG8, GL_RG, GL_UNSIGNED_BYTE),
            TexFormat::R32f => (GL_R32F, GL_RED, GL_FLOAT),
        };
        let pixels: *const c_void = data.map_or(null(), |d| d.as_ptr().cast());
        let gl = &self.ctx.gl;
        let mut name = 0u32;
        // SAFETY: the context is current. `name` is valid for one write.
        // glTexImage2D reads width * height * texel bytes from `pixels`
        // (unpack alignment is 1, row length and skips are never changed from
        // 0, and no unpack buffer is ever bound), and `data` was checked to be
        // exactly that long; null means no upload. The 8-bit formats use one
        // byte per channel, R32F one f32 per texel, as `texel_bytes` says.
        unsafe {
            (gl.gen_textures)(1, &mut name);
            (gl.bind_texture)(GL_TEXTURE_2D, name);
            (gl.tex_parameteri)(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            (gl.tex_parameteri)(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            (gl.tex_parameteri)(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            (gl.tex_parameteri)(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            (gl.tex_image_2d)(
                GL_TEXTURE_2D,
                0,
                internal,
                width as i32,
                height as i32,
                0,
                layout,
                kind,
                pixels,
            );
        }
        self.ctx.textures.push(TexInfo {
            name,
            width,
            height,
            format,
        });
        self.check("glTexImage2D failed")?;
        Ok(GlTexture(self.ctx.textures.len() - 1))
    }

    /// Width and height of a texture made here.
    pub(crate) fn texture_size(&self, t: GlTexture) -> Result<(u32, u32), GlError> {
        self.info(t).map(|i| (i.width, i.height))
    }

    fn info(&self, t: GlTexture) -> Result<TexInfo, GlError> {
        self.ctx
            .textures
            .get(t.0)
            .copied()
            .ok_or(GlError("unknown texture"))
    }

    /// Attach `t` as the only color target of the shared framebuffer.
    fn attach(&mut self, t: TexInfo) -> Result<(), GlError> {
        let gl = &self.ctx.gl;
        // SAFETY: the context is current; the framebuffer and texture were
        // made in it.
        let status = unsafe {
            (gl.bind_framebuffer)(GL_FRAMEBUFFER, self.ctx.fbo);
            (gl.framebuffer_texture_2d)(
                GL_FRAMEBUFFER,
                GL_COLOR_ATTACHMENT0,
                GL_TEXTURE_2D,
                t.name,
                0,
            );
            (gl.check_framebuffer_status)(GL_FRAMEBUFFER)
        };
        if status != GL_FRAMEBUFFER_COMPLETE {
            return Err(GlError("framebuffer incomplete"));
        }
        Ok(())
    }

    /// Draw a quad over all of `out` with `program`.
    pub(crate) fn draw(
        &mut self,
        program: GlProgram,
        out: GlTexture,
        inputs: &[(&CStr, GlTexture)],
        uniforms: &[(&CStr, Uniform<'_>)],
    ) -> Result<(), GlError> {
        if inputs.len() > MAX_INPUTS || inputs.iter().any(|&(_, t)| t == out) {
            return Err(GlError("bad draw inputs"));
        }
        let target = self.info(out)?;
        let bound: Vec<TexInfo> = inputs
            .iter()
            .map(|&(_, t)| self.info(t))
            .collect::<Result<_, _>>()?;
        self.attach(target)?;
        let gl = &self.ctx.gl;
        // SAFETY: the context is current, and the program and every texture
        // were made in it. Uniform names are NUL-terminated. Each glUniform*v
        // call reads `count` values of its type from a slice of exactly
        // `count` such values. GL ignores location -1 (a name the program does
        // not use). The draw reads no vertex buffer: positions come from the
        // vertex index.
        unsafe {
            (gl.viewport)(0, 0, target.width as i32, target.height as i32);
            (gl.use_program)(program.0);
            for (unit, (&(name, _), tex)) in inputs.iter().zip(&bound).enumerate() {
                (gl.active_texture)(GL_TEXTURE0 + unit as u32);
                (gl.bind_texture)(GL_TEXTURE_2D, tex.name);
                let at = (gl.get_uniform_location)(program.0, name.as_ptr());
                (gl.uniform1i)(at, unit as i32);
            }
            for &(name, value) in uniforms {
                let at = (gl.get_uniform_location)(program.0, name.as_ptr());
                if at < 0 {
                    continue;
                }
                match value {
                    Uniform::I1(v) => (gl.uniform1i)(at, v),
                    Uniform::I2([a, b]) => (gl.uniform2i)(at, a, b),
                    Uniform::I4([a, b, c, d]) => (gl.uniform4i)(at, a, b, c, d),
                    Uniform::F1(v) => (gl.uniform1f)(at, v),
                    Uniform::F2([a, b]) => (gl.uniform2f)(at, a, b),
                    Uniform::F4([a, b, c, d]) => (gl.uniform4f)(at, a, b, c, d),
                    Uniform::F1v(v) => (gl.uniform1fv)(at, v.len() as i32, v.as_ptr()),
                    Uniform::F4v(v) => (gl.uniform4fv)(at, v.len() as i32, v.as_ptr().cast()),
                }
            }
            (gl.draw_arrays)(GL_TRIANGLE_STRIP, 0, 4);
        }
        self.check("draw failed")
    }

    /// Read an RGBA8 texture into `out`, rows top first, no padding.
    pub(crate) fn read_rgba(&mut self, t: GlTexture, out: &mut [u8]) -> Result<(), GlError> {
        let info = self.info(t)?;
        let len = (info.width as usize) * (info.height as usize) * 4;
        if info.format != TexFormat::Rgba8 || out.len() != len {
            return Err(GlError("read buffer does not match the texture"));
        }
        self.attach(info)?;
        let gl = &self.ctx.gl;
        // SAFETY: the context is current and the texture is the complete
        // color attachment of the bound framebuffer. glReadPixels writes
        // width * height * 4 bytes (pack alignment 1, no pack buffer, row
        // length and skips 0), which is exactly `out.len()`.
        unsafe {
            (gl.read_pixels)(
                0,
                0,
                info.width as i32,
                info.height as i32,
                GL_RGBA,
                GL_UNSIGNED_BYTE,
                out.as_mut_ptr().cast(),
            );
        }
        self.check("glReadPixels failed")
    }

    /// Delete every texture made since the last call.
    fn delete_textures(&mut self) {
        if self.ctx.textures.is_empty() {
            return;
        }
        let names: Vec<u32> = self.ctx.textures.drain(..).map(|t| t.name).collect();
        let gl = &self.ctx.gl;
        // SAFETY: the context is current, every name was made in it, and the
        // slice and its length are passed together. Detaching first keeps the
        // framebuffer from holding a deleted texture.
        unsafe {
            (gl.bind_framebuffer)(GL_FRAMEBUFFER, self.ctx.fbo);
            (gl.framebuffer_texture_2d)(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 0, 0);
            (gl.delete_textures)(names.len() as i32, names.as_ptr());
        }
        // Errors here change nothing for the next call; clear them.
        let _ = self.check("delete");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EGL_OPENGL_ES_API: c_uint = 0x30A0;

    #[test]
    fn the_callers_rendering_api_is_put_back() {
        let Some(d) = egl_display() else {
            eprintln!("skipped: no EGL display here");
            return;
        };
        let Ok(mut context) = GlContext::create(ContextRequest::GL33) else {
            eprintln!("skipped: no OpenGL 3.3 core context here");
            return;
        };
        // A thread that uses OpenGL ES, as EGL starts every thread.
        // SAFETY: eglBindAPI takes a plain enum and only changes this
        // thread's EGL state; eglQueryAPI takes nothing.
        unsafe {
            assert_ne!((d.egl.bind_api)(EGL_OPENGL_ES_API), 0);
        }
        context.with_current(|gl| gl.max_side()).unwrap();
        // SAFETY: as above.
        assert_eq!(unsafe { (d.egl.query_api)() }, EGL_OPENGL_ES_API);
        drop(context);
        // SAFETY: as above.
        assert_eq!(unsafe { (d.egl.query_api)() }, EGL_OPENGL_ES_API);
    }
}
