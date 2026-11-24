//! D3D12.X on host D3D12.
//!
//! D3D12.X records commands from code inlined into the title, which writes
//! packets into the command list while its put pointer is below a limit and
//! calls the list's method otherwise. Our lists keep the limits at zero, so
//! every command arrives here as a call, to be run on host D3D12. The
//! title's GPU addresses are CPU addresses (console memory is unified).
//!
//! [`device`] is the device object titles create, [`dxgi`] its adapter side,
//! [`resource`] heaps and resources, [`fence`] fences, [`queue`] command
//! queues, [`list`] command allocators and lists, [`descriptor`]
//! descriptor heaps, [`root`] root signatures, [`query`] query heaps,
//! [`signature`] command signatures, [`pipeline`] pipeline states,
//! [`frame`] frame pacing, [`object`] what all console graphics objects
//! share, [`console`] which console model the title sees. Planned:
//! `present` (swap chain).

#[cfg(all(windows, feature = "diagnostics"))]
mod capture;
pub mod console;
#[cfg(all(windows, feature = "diagnostics"))]
mod copystat;
pub mod descriptor;
pub mod device;
#[cfg(windows)]
mod draw;
pub mod dxgi;
pub mod fence;
pub mod frame;
#[cfg(windows)]
mod host;
#[cfg(windows)]
mod indirect;
#[cfg(windows)]
mod rects;
pub mod list;
mod mapped;
pub mod object;
pub mod pipeline;
#[cfg(windows)]
mod primitive;
pub mod query;
pub mod queue;
pub mod resource;
pub mod root;
#[cfg(windows)]
mod sampler;
#[cfg(windows)]
pub mod shader_cache;
#[cfg(windows)]
mod shadow;
pub mod signature;
#[cfg(windows)]
mod texture;
#[cfg(windows)]
mod watch;

#[cfg(windows)]
pub use host::{HdrDisplayInfo, display_hdr_info, on_failure};

/// `VICTORY_BACKGROUND=1`: a run nobody sits at (someone works in other
/// programs meanwhile). Windows open without taking the focus, the real
/// keyboard is not read (scripted keys still are), the text entry dialog
/// is dismissed, the title's sound is muted, and its windows are off the
/// visible desktop ([`background_window`]).
pub fn background() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_BACKGROUND").is_ok_and(|v| v.trim() == "1"))
}
/// In a background run: where a top-level window of `width` x `height`
/// goes and how big it is, so that it covers nothing of the work of
/// whoever sits at the screen: past the bottom right corner of the desktop
/// (all monitors), at its size. Frame captures are of the frames, not of
/// the window, and show them as before.
///
/// `VICTORY_BACKGROUND_WINDOW=corner` instead makes it small
/// ([`BACKGROUND_WIDTH`] wide, in proportion; what it shows is scaled to
/// it) in the bottom right corner of the primary screen's work area;
/// `=visible` keeps such a run's windows as big and where they would be.
/// Either way they are shown behind the other windows. (A window with only
/// a corner on the screen does not stay there: macOS moves it back whole.)
#[cfg(windows)]
pub fn background_window(width: i32, height: i32) -> Option<(i32, i32, i32, i32)> {
    use windows_sys::Win32::Foundation::RECT;
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        GetSystemMetrics, SM_CXVIRTUALSCREEN, SM_CYVIRTUALSCREEN, SM_XVIRTUALSCREEN, SM_YVIRTUALSCREEN,
        SPI_GETWORKAREA, SystemParametersInfoW,
    };
    if !background() || width <= 0 || height <= 0 {
        return None;
    }
    match std::env::var("VICTORY_BACKGROUND_WINDOW").unwrap_or_default().trim() {
        "visible" => None,
        "corner" => {
            let mut area = RECT::default();
            // SAFETY: `area` receives the primary monitor's work area.
            if unsafe { SystemParametersInfoW(SPI_GETWORKAREA, 0, (&raw mut area).cast(), 0) } == 0 {
                return None;
            }
            let (w, h) = if width > BACKGROUND_WIDTH {
                let h = i64::from(height) * i64::from(BACKGROUND_WIDTH) / i64::from(width);
                (BACKGROUND_WIDTH, h.max(1) as i32)
            } else {
                (width, height)
            };
            Some((area.right - w, area.bottom - h, w, h))
        }
        _ => {
            // SAFETY: plain queries.
            let (x, y, w, h) = unsafe {
                (
                    GetSystemMetrics(SM_XVIRTUALSCREEN),
                    GetSystemMetrics(SM_YVIRTUALSCREEN),
                    GetSystemMetrics(SM_CXVIRTUALSCREEN),
                    GetSystemMetrics(SM_CYVIRTUALSCREEN),
                )
            };
            Some((x + w + 64, y + h + 64, width, height))
        }
    }
}

/// How wide a background run's windows are in a corner of the screen.
#[cfg(windows)]
pub const BACKGROUND_WIDTH: i32 = 160;

/// In a background run: puts a top-level window just shown behind every
/// other window, without activating it.
#[cfg(windows)]
pub fn background_behind(hwnd: *mut std::ffi::c_void) {
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        GA_ROOT, GetAncestor, HWND_BOTTOM, SWP_NOACTIVATE, SWP_NOMOVE, SWP_NOSIZE, SetWindowPos,
    };
    if !background() || hwnd.is_null() {
        return;
    }
    // SAFETY: plain calls on a window of ours.
    unsafe {
        if GetAncestor(hwnd, GA_ROOT) == hwnd {
            SetWindowPos(hwnd, HWND_BOTTOM, 0, 0, 0, 0, SWP_NOACTIVATE | SWP_NOMOVE | SWP_NOSIZE);
        }
    }
}
pub use mapped::{title_pages_mapped, title_pages_released, title_pages_reserved, title_pages_unmapped};
#[cfg(windows)]
pub use watch::{title_memory_remapped, tracking as write_tracking, writing as kernel_will_write};
