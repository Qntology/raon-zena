#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
// Disable clippy and `u128` not being FFI-safe (see #1)
#![allow(clippy::all)]
#![allow(improper_ctypes)]

include!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/generated/bindings.rs"));

/// ONNX Runtime loaded at run time instead of linked (Raon Zena change to the vendored crate).
///
/// Only `OrtGetApiBase` is a real C symbol; everything else goes through the API table it returns, so loading that
/// one function from a library found at run time is enough. A binary that never uses ONNX Runtime — or runs on a
/// machine without it — therefore starts normally, and callers can ask [`dynload::load`] first and fall back.
///
/// Search order: `ORT_DYLIB_PATH` (a file), next to the executable, its parent folder (test binaries live in
/// `target/<profile>/deps`), `%LOCALAPPDATA%\RaonZena\models\onnxruntime`, then the library the build script
/// prepared (development builds). The library is always opened by full path, never by bare name, so an unrelated
/// `onnxruntime.dll` elsewhere on the system (Windows ships an old one) is never picked up.
pub mod dynload {
    use super::OrtApiBase;
    use std::path::PathBuf;
    use std::sync::OnceLock;

    #[cfg(target_os = "windows")]
    pub const DYLIB_NAME: &str = "onnxruntime.dll";
    #[cfg(target_os = "macos")]
    pub const DYLIB_NAME: &str = "libonnxruntime.dylib";
    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    pub const DYLIB_NAME: &str = "libonnxruntime.so";

    #[cfg(all(target_os = "windows", target_arch = "x86"))]
    type GetApiBase = unsafe extern "stdcall" fn() -> *const OrtApiBase;
    #[cfg(not(all(target_os = "windows", target_arch = "x86")))]
    type GetApiBase = unsafe extern "C" fn() -> *const OrtApiBase;

    struct Loaded {
        _lib: libloading::Library,
        get_api_base: GetApiBase,
        path: PathBuf,
    }

    // The function pointer is plain code address; the library is never unloaded.
    unsafe impl Send for Loaded {}
    unsafe impl Sync for Loaded {}

    static LOADED: OnceLock<Result<Loaded, String>> = OnceLock::new();

    /// Where the library is looked for, best first.
    pub fn candidates() -> Vec<PathBuf> {
        let mut v = Vec::new();
        if let Some(p) = std::env::var_os("ORT_DYLIB_PATH") {
            v.push(PathBuf::from(p));
        }
        if let Ok(exe) = std::env::current_exe() {
            if let Some(dir) = exe.parent() {
                v.push(dir.join(DYLIB_NAME));
                if let Some(up) = dir.parent() {
                    v.push(up.join(DYLIB_NAME));
                }
            }
        }
        if let Some(d) = std::env::var_os("LOCALAPPDATA").or_else(|| std::env::var_os("XDG_DATA_HOME")) {
            v.push(
                PathBuf::from(d)
                    .join("RaonZena")
                    .join("models")
                    .join("onnxruntime")
                    .join(DYLIB_NAME),
            );
        }
        if let Some(dir) = option_env!("ORT_BUILD_LIB_DIR").filter(|d| !d.is_empty()) {
            v.push(PathBuf::from(dir).join(DYLIB_NAME));
        }
        v
    }

    #[cfg(target_os = "windows")]
    unsafe fn open(path: &std::path::Path) -> Result<libloading::Library, libloading::Error> {
        // Resolve the library's own dependencies (providers) from its folder.
        libloading::os::windows::Library::load_with_flags(path, libloading::os::windows::LOAD_WITH_ALTERED_SEARCH_PATH).map(Into::into)
    }

    #[cfg(not(target_os = "windows"))]
    unsafe fn open(path: &std::path::Path) -> Result<libloading::Library, libloading::Error> {
        libloading::Library::new(path)
    }

    fn try_load() -> Result<Loaded, String> {
        let mut tried = Vec::new();
        for path in candidates() {
            if !path.is_file() {
                continue;
            }
            match unsafe { open(&path) } {
                Ok(lib) => {
                    let sym = unsafe { lib.get::<GetApiBase>(b"OrtGetApiBase\0") };
                    match sym {
                        Ok(f) => {
                            let get_api_base = *f;
                            return Ok(Loaded {
                                _lib: lib,
                                get_api_base,
                                path,
                            });
                        }
                        Err(e) => tried.push(format!("{}: {e}", path.display())),
                    }
                }
                Err(e) => tried.push(format!("{}: {e}", path.display())),
            }
        }
        Err(if tried.is_empty() {
            format!("{DYLIB_NAME} not found (set ORT_DYLIB_PATH)")
        } else {
            format!("{DYLIB_NAME} could not be loaded: {}", tried.join("; "))
        })
    }

    /// Load the library once; the path it came from, or why it is unavailable.
    pub fn load() -> Result<&'static std::path::Path, String> {
        match LOADED.get_or_init(try_load) {
            Ok(l) => Ok(l.path.as_path()),
            Err(e) => Err(e.clone()),
        }
    }

    /// Load the library and check it serves the API version these bindings were generated for.
    pub fn api_available() -> Result<&'static std::path::Path, String> {
        let path = load()?;
        let base = api_base();
        if base.is_null() {
            return Err(format!("{}: OrtGetApiBase returned null", path.display()));
        }
        let get_api = unsafe { (*base).GetApi }.ok_or("OrtApiBase.GetApi is null")?;
        let api = unsafe { get_api(super::ORT_API_VERSION) };
        if api.is_null() {
            return Err(format!(
                "{} does not serve ORT API version {}",
                path.display(),
                super::ORT_API_VERSION
            ));
        }
        Ok(path)
    }

    pub(crate) fn api_base() -> *const OrtApiBase {
        match LOADED.get_or_init(try_load) {
            Ok(l) => unsafe { (l.get_api_base)() },
            Err(_) => std::ptr::null(),
        }
    }
}

/// `OrtGetApiBase` of the run-time loaded library (null when it is unavailable — check [`dynload::api_available`]).
pub unsafe fn OrtGetApiBase() -> *const OrtApiBase {
    dynload::api_base()
}

#[cfg(target_os = "windows")]
pub type OnnxEnumInt = i32;
#[cfg(not(target_os = "windows"))]
pub type OnnxEnumInt = u32;
