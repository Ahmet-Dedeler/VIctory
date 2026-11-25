//! Runs a title's original x86-64 code natively, inside our process.
//!
//! Xbox Series X|S executables are x86-64 PE32+ images built for a Zen 2
//! CPU, so the machine code needs no translation. What differs is the
//! platform underneath, so the loader's job is to:
//!
//! 1. [`prepare()`] a lightly patched copy of the image (pure, portable): mark
//!    it as a DLL, remove its entry point and import directory, and detach
//!    its TLS callbacks, so the Windows loader maps it, relocates it,
//!    allocates its static TLS and registers its unwind tables, but runs
//!    none of its code and resolves none of its imports;
//! 2. load that copy as a real module (debuggers, profilers, crash dumps
//!    and stack walks all see it); this step is Windows-only;
//! 3. map the DLLs the title ships beside it the same way
//!    (`load_title_dlls`);
//! 4. bind every import ourselves through an [`ImportResolver`], our
//!    runtime (`victory-hle`), the host or the title's DLLs, and route
//!    anything unresolved to a generated stub that reports exactly what
//!    was called;
//! 5. initialise the title's DLLs, then run TLS callbacks and the entry
//!    point on a thread with the stack size the image asks for; from then
//!    on, pass every thread start and end on to the attached images.

#[cfg(any(windows, test))]
mod console_insn;
#[cfg(windows)]
mod crash;
#[cfg(windows)]
mod hooks;
#[cfg(windows)]
pub mod masked;
pub mod prepare;
pub mod resolve;
#[cfg(all(windows, feature = "diagnostics"))]
mod stacks;
#[cfg(windows)]
mod threads;
#[cfg(windows)]
mod thunks;
#[cfg(windows)]
mod tls;
#[cfg(windows)]
mod windows;

#[cfg(windows)]
pub use crash::{
    locate, log_crash, log_crashes_to, note_title_output, report_crashes, report_unhandled,
};
#[cfg(all(windows, feature = "diagnostics"))]
pub use crash::save_on_crash;
pub use prepare::{PrepareError, PreparedImage, prepare};
pub use resolve::{ImportResolver, Provenance, Resolution};
#[cfg(all(windows, feature = "diagnostics"))]
pub use stacks::{dump_stacks, dump_stacks_on_schedule};
#[cfg(windows)]
pub use threads::{disable_thread_calls, in_title_image};
#[cfg(windows)]
pub use thunks::{UnimplementedPolicy, unimplemented_call_count};
#[cfg(windows)]
pub use windows::{
    BindReport, LoadedImage, MissingImport, WithTitleDlls, load, load_title_dll, load_title_dlls,
};

/// `DLL_PROCESS_ATTACH`, for TLS callbacks.
pub const DLL_PROCESS_ATTACH: u32 = 1;
/// `DLL_PROCESS_DETACH`.
pub const DLL_PROCESS_DETACH: u32 = 0;

/// The little of Windows' x86-64 context, memory and clock API that
/// `console_insn`'s unit tests use, for running them on a host that is not
/// Windows. The runtime itself is Windows-only. Every query is answered as
/// committed, writable memory: the tests ask only about memory they own.
#[cfg(all(test, not(windows)))]
#[allow(dead_code, non_snake_case, non_camel_case_types)]
mod test_windows {
    pub mod Win32 {
        pub mod System {
            pub mod Diagnostics {
                pub mod Debug {
                    #[repr(C)]
                    #[derive(Clone, Copy)]
                    pub struct M128A {
                        pub Low: u64,
                        pub High: i64,
                    }

                    #[repr(C)]
                    #[derive(Clone, Copy)]
                    pub struct XSAVE_FORMAT {
                        pub XmmRegisters: [M128A; 16],
                    }

                    #[repr(C)]
                    #[derive(Clone, Copy)]
                    pub union CONTEXT_0 {
                        pub FltSave: XSAVE_FORMAT,
                    }

                    #[repr(C)]
                    #[derive(Clone, Copy)]
                    pub struct CONTEXT {
                        pub Rax: u64,
                        pub Rcx: u64,
                        pub Rdx: u64,
                        pub Rbx: u64,
                        pub Rsp: u64,
                        pub Rbp: u64,
                        pub Rsi: u64,
                        pub Rdi: u64,
                        pub R8: u64,
                        pub R9: u64,
                        pub R10: u64,
                        pub R11: u64,
                        pub R12: u64,
                        pub R13: u64,
                        pub R14: u64,
                        pub R15: u64,
                        pub Rip: u64,
                        pub Anonymous: CONTEXT_0,
                    }
                }
            }

            pub mod Memory {
                use std::ffi::c_void;

                pub type PAGE_PROTECTION_FLAGS = u32;
                pub type VIRTUAL_ALLOCATION_TYPE = u32;
                pub type PAGE_TYPE = u32;

                pub const MEM_COMMIT: VIRTUAL_ALLOCATION_TYPE = 0x1000;
                pub const PAGE_NOACCESS: PAGE_PROTECTION_FLAGS = 0x01;
                pub const PAGE_READWRITE: PAGE_PROTECTION_FLAGS = 0x04;
                pub const PAGE_EXECUTE_READWRITE: PAGE_PROTECTION_FLAGS = 0x40;
                pub const PAGE_GUARD: PAGE_PROTECTION_FLAGS = 0x100;

                #[repr(C)]
                pub struct MEMORY_BASIC_INFORMATION {
                    pub BaseAddress: *mut c_void,
                    pub AllocationBase: *mut c_void,
                    pub AllocationProtect: PAGE_PROTECTION_FLAGS,
                    pub RegionSize: usize,
                    pub State: VIRTUAL_ALLOCATION_TYPE,
                    pub Protect: PAGE_PROTECTION_FLAGS,
                    pub Type: PAGE_TYPE,
                }

                pub unsafe fn VirtualQuery(
                    lpaddress: *const c_void,
                    lpbuffer: *mut MEMORY_BASIC_INFORMATION,
                    dwlength: usize,
                ) -> usize {
                    if lpaddress.is_null() || lpbuffer.is_null() || dwlength < size_of::<MEMORY_BASIC_INFORMATION>() {
                        return 0;
                    }
                    let answer = MEMORY_BASIC_INFORMATION {
                        BaseAddress: std::ptr::null_mut(),
                        AllocationBase: std::ptr::null_mut(),
                        AllocationProtect: PAGE_READWRITE,
                        RegionSize: isize::MAX as usize,
                        State: MEM_COMMIT,
                        Protect: PAGE_READWRITE,
                        Type: 0x20000,
                    };
                    // SAFETY: the caller passes a valid out pointer of the right size.
                    unsafe { lpbuffer.write(answer) };
                    size_of::<MEMORY_BASIC_INFORMATION>()
                }

                pub unsafe fn VirtualProtect(
                    _lpaddress: *const c_void,
                    _dwsize: usize,
                    _flnewprotect: PAGE_PROTECTION_FLAGS,
                    lpfloldprotect: *mut PAGE_PROTECTION_FLAGS,
                ) -> i32 {
                    if !lpfloldprotect.is_null() {
                        // SAFETY: the caller passes a valid out pointer.
                        unsafe { lpfloldprotect.write(PAGE_READWRITE) };
                    }
                    1
                }
            }

            pub mod Performance {
                pub unsafe fn QueryPerformanceFrequency(lpperformancecount: *mut i64) -> i32 {
                    if !lpperformancecount.is_null() {
                        // SAFETY: the caller passes a valid out pointer.
                        unsafe { lpperformancecount.write(1_000_000) };
                    }
                    1
                }

                pub unsafe fn QueryPerformanceCounter(lpperformancecount: *mut i64) -> i32 {
                    static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
                    if !lpperformancecount.is_null() {
                        let elapsed = START.get_or_init(std::time::Instant::now).elapsed().as_micros() as i64;
                        // SAFETY: the caller passes a valid out pointer.
                        unsafe { lpperformancecount.write(elapsed) };
                    }
                    1
                }
            }

            pub mod Threading {
                pub unsafe fn GetCurrentThreadId() -> u32 {
                    1
                }

                pub unsafe fn GetCurrentProcessorNumber() -> u32 {
                    0
                }
            }
        }
    }
}

/// Makes `size` bytes at `base` writable and executable.
///
/// # Safety
///
/// `base` starts an image mapped in this process and `size` is its size.
#[cfg(windows)]
pub unsafe fn make_writable(base: *const core::ffi::c_void, size: usize) -> bool {
    use windows_sys::Win32::System::Memory::{PAGE_EXECUTE_READWRITE, VirtualProtect};
    let mut before = 0u32;
    // SAFETY: the caller's contract.
    unsafe { VirtualProtect(base, size, PAGE_EXECUTE_READWRITE, &mut before) != 0 }
}
