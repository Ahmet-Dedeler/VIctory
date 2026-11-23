//! Turns a title's image into something the Windows loader maps but does
//! not run. Pure byte manipulation, so it is testable on any host.

use victory_pe::layout::{self, dir, dll_flags, file_flags};
use victory_pe::{ImageInfo, PeError};

#[derive(Debug, thiserror::Error)]
pub enum PrepareError {
    #[error(transparent)]
    Pe(#[from] PeError),
    #[error("cannot patch image: {0}")]
    Patch(&'static str),
}

/// A patched copy of an image plus everything we need from the original.
#[derive(Debug, Clone)]
pub struct PreparedImage {
    /// Parsed from the *original* bytes: imports, TLS callbacks and entry
    /// point as the title expects them.
    pub info: ImageInfo,
    /// The patched bytes to hand to the Windows loader.
    pub bytes: Vec<u8>,
    /// BLAKE3 of the original bytes: identifies the exact build.
    pub original_hash: blake3::Hash,
}

impl PreparedImage {
    /// File name for the patched copy, unique per original build.
    pub fn file_name(&self) -> String {
        format!("victory-image-{}.dll", &self.original_hash.to_hex()[..16])
    }
}

/// Produces the patched copy. The changes are:
///
/// * `IMAGE_FILE_DLL` set, entry point cleared: the OS maps it like a DLL
///   and calls nothing;
/// * import and bound-import directories cleared: we bind imports
///   ourselves, using `info.imports` from the original;
/// * TLS callback pointer cleared: the OS still allocates static TLS for
///   the image, but callbacks only run when we call them (after binding);
/// * `FORCE_INTEGRITY` cleared, checksum zeroed: the copy is unsigned.
pub fn prepare(original: &[u8]) -> Result<PreparedImage, PrepareError> {
    let info = ImageInfo::parse(original)?;
    let mut bytes = original.to_vec();

    let fh = layout::file_header_offset(&bytes).ok_or(PrepareError::Patch("no file header"))?;
    let opt =
        layout::optional_header_offset(&bytes).ok_or(PrepareError::Patch("no optional header"))?;

    let characteristics = info.characteristics | file_flags::DLL;
    write_u16(
        &mut bytes,
        fh + layout::FILE_HEADER_CHARACTERISTICS,
        characteristics,
    )?;
    write_u32(&mut bytes, opt + layout::OPT_ADDRESS_OF_ENTRY_POINT, 0)?;
    write_u32(&mut bytes, opt + layout::OPT_CHECKSUM, 0)?;
    let dll_characteristics = info.dll_characteristics & !dll_flags::FORCE_INTEGRITY;
    write_u16(
        &mut bytes,
        opt + layout::OPT_DLL_CHARACTERISTICS,
        dll_characteristics,
    )?;

    for index in [dir::IMPORT, dir::BOUND_IMPORT] {
        if let Some(at) = layout::data_directory_offset(&bytes, index) {
            write_u32(&mut bytes, at, 0)?;
            write_u32(&mut bytes, at + 4, 0)?;
        }
    }

    if let Some(tls) = &info.tls
        && tls.callbacks_va != 0
    {
        let at = info
            .rva_to_file_offset(tls.directory_rva)
            .ok_or(PrepareError::Patch("TLS directory is not file-backed"))?;
        write_u64(
            &mut bytes,
            at as usize + layout::TLS64_ADDRESS_OF_CALLBACKS,
            0,
        )?;
        // The field has a base relocation: a loader that moves the image
        // would turn the null into the distance moved and call "callbacks"
        // read from there (Wine's does, and faults).
        let field = tls.directory_rva + layout::TLS64_ADDRESS_OF_CALLBACKS as u32;
        drop_relocation(&mut bytes, &info, field)?;
    }

    Ok(PreparedImage {
        info,
        bytes,
        original_hash: blake3::hash(original),
    })
}

/// Turns the base relocation of the pointer at `rva`, if it has one, into
/// padding (type 0, which loaders skip), so the pointer keeps its value
/// wherever the image is mapped.
fn drop_relocation(bytes: &mut [u8], info: &ImageInfo, rva: u32) -> Result<(), PrepareError> {
    let Some(directory) = layout::data_directory_offset(bytes, dir::BASERELOC) else {
        return Ok(());
    };
    let (Some(table), Some(size)) = (
        layout::read_u32(bytes, directory),
        layout::read_u32(bytes, directory + 4),
    ) else {
        return Ok(());
    };
    let Some(start) = info.rva_to_file_offset(table) else {
        return Ok(());
    };
    let (mut at, end) = (start as usize, start as usize + size as usize);
    // Blocks of `{ page rva, block size, entries: [type:4 | offset:12] }`.
    while at + 8 <= end {
        let (Some(page), Some(block)) =
            (layout::read_u32(bytes, at), layout::read_u32(bytes, at + 4))
        else {
            break;
        };
        let block = block as usize;
        if block < 8 {
            break;
        }
        if page == rva & !0xfff {
            for entry in (at + 8..(at + block).min(end)).step_by(2) {
                let Some(&[low, high]) = bytes.get(entry..entry + 2) else {
                    break;
                };
                let value = u16::from_le_bytes([low, high]);
                if value >> 12 != 0 && u32::from(value & 0xfff) == rva & 0xfff {
                    write_u16(bytes, entry, 0)?;
                }
            }
        }
        at += block;
    }
    Ok(())
}

fn write_u16(bytes: &mut [u8], at: usize, v: u16) -> Result<(), PrepareError> {
    write(bytes, at, &v.to_le_bytes())
}
fn write_u32(bytes: &mut [u8], at: usize, v: u32) -> Result<(), PrepareError> {
    write(bytes, at, &v.to_le_bytes())
}
fn write_u64(bytes: &mut [u8], at: usize, v: u64) -> Result<(), PrepareError> {
    write(bytes, at, &v.to_le_bytes())
}
fn write(bytes: &mut [u8], at: usize, v: &[u8]) -> Result<(), PrepareError> {
    bytes
        .get_mut(at..at + v.len())
        .ok_or(PrepareError::Patch("header field outside of file"))?
        .copy_from_slice(v);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patches_own_test_binary() {
        let original = std::fs::read(std::env::current_exe().unwrap()).unwrap();
        let prepared = prepare(&original).unwrap();
        assert!(!prepared.info.is_dll(), "info describes the original");
        assert!(prepared.info.import_count() > 0);

        let patched = ImageInfo::parse(&prepared.bytes).unwrap();
        assert!(patched.is_dll());
        assert_eq!(patched.entry_rva, 0);
        assert!(patched.imports.is_empty());
        assert_eq!(patched.dll_characteristics & dll_flags::FORCE_INTEGRITY, 0);
        if let Some(tls) = &patched.tls {
            assert_eq!(tls.callbacks_va, 0);
        }
        // The cleared callback pointer is no longer relocated.
        if let Some(tls) = &prepared.info.tls {
            let field = tls.directory_rva + layout::TLS64_ADDRESS_OF_CALLBACKS as u32;
            assert_eq!(relocations_of(&original, &prepared.info, field), 1);
            assert_eq!(relocations_of(&prepared.bytes, &prepared.info, field), 0);
        }
        // Nothing else moved: same sections, same size.
        assert_eq!(patched.sections.len(), prepared.info.sections.len());
        assert_eq!(prepared.bytes.len(), original.len());
    }

    /// How many base relocations `bytes` has for the pointer at `rva`.
    fn relocations_of(bytes: &[u8], info: &ImageInfo, rva: u32) -> usize {
        let directory = layout::data_directory_offset(bytes, dir::BASERELOC).unwrap();
        let table = layout::read_u32(bytes, directory).unwrap();
        let size = layout::read_u32(bytes, directory + 4).unwrap() as usize;
        let mut at = info.rva_to_file_offset(table).unwrap() as usize;
        let (end, mut found) = (at + size, 0);
        while at + 8 <= end {
            let page = layout::read_u32(bytes, at).unwrap();
            let block = layout::read_u32(bytes, at + 4).unwrap() as usize;
            for entry in (at + 8..at + block).step_by(2) {
                let value = u16::from_le_bytes([bytes[entry], bytes[entry + 1]]);
                if value >> 12 != 0 && page + u32::from(value & 0xfff) == rva {
                    found += 1;
                }
            }
            at += block.max(8);
        }
        found
    }

    #[test]
    fn file_name_is_stable_per_build() {
        let original = std::fs::read(std::env::current_exe().unwrap()).unwrap();
        let a = prepare(&original).unwrap();
        let b = prepare(&original).unwrap();
        assert_eq!(a.file_name(), b.file_name());
    }
}
