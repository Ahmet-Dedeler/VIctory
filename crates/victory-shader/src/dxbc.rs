//! The DXBC container: a header, then a table of FourCC-tagged parts.
//!
//! ```text
//! 0x00  "DXBC"
//! 0x04  digest[16]
//! 0x14  version major u16, minor u16
//! 0x18  total size u32
//! 0x1C  part count u32
//! 0x20  part offsets u32[count]
//! part: fourcc[4], size u32, data[size]
//! ```

use crate::ShaderError;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct FourCC(pub [u8; 4]);

impl FourCC {
    pub const DXIL: FourCC = FourCC(*b"DXIL");
    /// Native code another compiler made, in place of `DXIL` (VIctory upstream 0b35b7f).
    pub const ACCC: FourCC = FourCC(*b"ACCC");
    pub const STAT: FourCC = FourCC(*b"STAT");
    pub const RTS0: FourCC = FourCC(*b"RTS0");
    pub const PSV0: FourCC = FourCC(*b"PSV0");
    pub const ISG1: FourCC = FourCC(*b"ISG1");
    pub const OSG1: FourCC = FourCC(*b"OSG1");
    /// Per-primitive outputs of a mesh shader.
    pub const PSG1: FourCC = FourCC(*b"PSG1");
    pub const SFI0: FourCC = FourCC(*b"SFI0");
    pub const HASH: FourCC = FourCC(*b"HASH");
    pub const ILDB: FourCC = FourCC(*b"ILDB");
    pub const ILDN: FourCC = FourCC(*b"ILDN");
}

impl std::fmt::Debug for FourCC {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self)
    }
}

impl std::fmt::Display for FourCC {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for &b in &self.0 {
            let c = if b.is_ascii_graphic() { b as char } else { '.' };
            write!(f, "{c}")?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Part<'a> {
    pub fourcc: FourCC,
    /// Offset of the part header within the container.
    pub offset: usize,
    pub data: &'a [u8],
}

#[derive(Debug, Clone)]
pub struct Container<'a> {
    pub digest: [u8; 16],
    pub version: (u16, u16),
    pub parts: Vec<Part<'a>>,
}

const HEADER_SIZE: usize = 0x20;
const MAX_PARTS: usize = 256;

impl<'a> Container<'a> {
    pub fn parse(data: &'a [u8]) -> Result<Self, ShaderError> {
        if data.len() < HEADER_SIZE || &data[0..4] != b"DXBC" {
            return Err(ShaderError::NotDxbc);
        }
        let digest = data[4..20].try_into().expect("16 bytes");
        let version = (u16_at(data, 0x14)?, u16_at(data, 0x16)?);
        let total = u32_at(data, 0x18)? as usize;
        if total > data.len() || total < HEADER_SIZE {
            return Err(ShaderError::Malformed("container size disagrees with data"));
        }
        let data = &data[..total];
        let count = u32_at(data, 0x1C)? as usize;
        if count > MAX_PARTS {
            return Err(ShaderError::Malformed("implausible part count"));
        }
        let mut parts = Vec::with_capacity(count);
        for i in 0..count {
            let offset = u32_at(data, HEADER_SIZE + 4 * i)? as usize;
            let fourcc = FourCC(
                data.get(offset..offset + 4)
                    .ok_or(ShaderError::Malformed("part header out of bounds"))?
                    .try_into()
                    .expect("4 bytes"),
            );
            let size = u32_at(data, offset + 4)? as usize;
            let body = data
                .get(offset + 8..offset + 8 + size)
                .ok_or(ShaderError::Malformed("part data out of bounds"))?;
            parts.push(Part {
                fourcc,
                offset,
                data: body,
            });
        }
        Ok(Container {
            digest,
            version,
            parts,
        })
    }

    pub fn part(&self, fourcc: FourCC) -> Option<&Part<'a>> {
        self.parts.iter().find(|p| p.fourcc == fourcc)
    }

    /// The pixel entry name, if the PSV0 part has PSVRuntimeInfo3 or later.
    pub fn pixel_entry_name(&self) -> Option<&'a str> {
        let psv = self.part(FourCC::PSV0)?.data;
        let info_size = u32_at(psv, 0).ok()? as usize;
        let (info, mut rest) = psv.get(4..)?.split_at_checked(info_size)?;
        // PSVRuntimeInfo1 adds the stage (0 is pixel); version 3 adds
        // the entry name offset. Both fields keep their offsets in later versions.
        if *info.get(24)? != 0 {
            return None;
        }
        let name_offset = u32_at(info, 48).ok()? as usize;
        let resources = u32_at(rest, 0).ok()? as usize;
        rest = rest.get(4..)?;
        if resources != 0 {
            let stride = u32_at(rest, 0).ok()? as usize;
            // The original resource binding record is four u32s.
            if stride < 16 {
                return None;
            }
            rest = rest.get(4..)?.get(resources.checked_mul(stride)?..)?;
        }
        let strings = u32_at(rest, 0).ok()? as usize;
        let table = rest.get(4..)?.get(..strings)?;
        if table.last() != Some(&0) {
            return None;
        }
        let name = std::ffi::CStr::from_bytes_until_nul(table.get(name_offset..)?).ok()?.to_str().ok()?;
        (!name.is_empty()).then_some(name)
    }
}

pub(crate) fn u16_at(data: &[u8], at: usize) -> Result<u16, ShaderError> {
    data.get(at..at + 2)
        .map(|b| u16::from_le_bytes(b.try_into().expect("2 bytes")))
        .ok_or(ShaderError::Malformed("truncated"))
}

pub(crate) fn u32_at(data: &[u8], at: usize) -> Result<u32, ShaderError> {
    data.get(at..at + 4)
        .map(|b| u32::from_le_bytes(b.try_into().expect("4 bytes")))
        .ok_or(ShaderError::Malformed("truncated"))
}

pub(crate) fn u64_at(data: &[u8], at: usize) -> Result<u64, ShaderError> {
    data.get(at..at + 8)
        .map(|b| u64::from_le_bytes(b.try_into().expect("8 bytes")))
        .ok_or(ShaderError::Malformed("truncated"))
}

/// Builds a container from parts (test helper).
#[cfg(test)]
pub fn build(parts: &[(FourCC, &[u8])]) -> Vec<u8> {
    let table = HEADER_SIZE + 4 * parts.len();
    let total = table + parts.iter().map(|(_, d)| 8 + d.len()).sum::<usize>();
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(b"DXBC");
    out.extend_from_slice(&[0; 16]);
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&(total as u32).to_le_bytes());
    out.extend_from_slice(&(parts.len() as u32).to_le_bytes());
    let mut offset = table;
    for (_, d) in parts {
        out.extend_from_slice(&(offset as u32).to_le_bytes());
        offset += 8 + d.len();
    }
    for (fourcc, d) in parts {
        out.extend_from_slice(&fourcc.0);
        out.extend_from_slice(&(d.len() as u32).to_le_bytes());
        out.extend_from_slice(d);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_parts() {
        let bytes = build(&[(FourCC::SFI0, &[1, 2, 3, 4]), (FourCC::HASH, &[9; 20])]);
        let c = Container::parse(&bytes).unwrap();
        assert_eq!(c.parts.len(), 2);
        assert_eq!(c.part(FourCC::SFI0).unwrap().data, &[1, 2, 3, 4]);
        assert_eq!(c.part(FourCC::HASH).unwrap().data.len(), 20);
        assert!(c.part(FourCC::DXIL).is_none());
    }

    #[test]
    fn rejects_bad_input() {
        assert_eq!(Container::parse(b"nope").unwrap_err(), ShaderError::NotDxbc);
        let mut bytes = build(&[(FourCC::SFI0, &[0; 8])]);
        let last = bytes.len() - 1;
        bytes[0x20] = 0xFF; // corrupt the first part offset
        assert!(Container::parse(&bytes).is_err());
        bytes.truncate(last);
        assert!(Container::parse(&bytes).is_err());
    }

    fn psv(info_size: usize, stage: u8, offset: u32, records: u32, stride: u32, strings: &[u8]) -> Vec<u8> {
        let mut part = (info_size as u32).to_le_bytes().to_vec();
        let mut info = vec![0; info_size];
        info[24] = stage;
        if info_size >= 52 {
            info[48..52].copy_from_slice(&offset.to_le_bytes());
        }
        part.extend_from_slice(&info);
        part.extend_from_slice(&records.to_le_bytes());
        if records != 0 {
            part.extend_from_slice(&stride.to_le_bytes());
            part.resize(part.len() + (records * stride) as usize, 0);
        }
        part.extend_from_slice(&(strings.len() as u32).to_le_bytes());
        part.extend_from_slice(strings);
        part
    }

    fn pixel_name(part: &[u8]) -> Option<String> {
        let bytes = build(&[(FourCC::PSV0, part)]);
        Container::parse(&bytes).unwrap().pixel_entry_name().map(str::to_owned)
    }

    #[test]
    fn reads_pixel_names_with_and_without_resource_bindings() {
        let strings = b"\0TEXCOORD\0PS_BakeDecorations_Test\0";
        for (info, count, stride) in [(52, 0, 0), (52, 1, 16), (60, 2, 24)] {
            let part = psv(info, 0, b"\0TEXCOORD\0".len() as u32, count, stride, strings);
            assert_eq!(pixel_name(&part).as_deref(), Some("PS_BakeDecorations_Test"));
        }
    }

    #[test]
    fn skips_legacy_psv_and_non_pixel_entries() {
        let strings = b"\0PS_DecorationBlend_Test\0";
        assert_eq!(pixel_name(&psv(48, 0, 1, 0, 0, strings)), None);
        assert_eq!(pixel_name(&psv(52, 5, 1, 0, 0, strings)), None);
        let bytes = build(&[(FourCC::SFI0, &[0; 8])]);
        assert_eq!(Container::parse(&bytes).unwrap().pixel_entry_name(), None);
    }

    #[test]
    fn rejects_malformed_pixel_names() {
        let strings = b"\0PS_DecorationBlend_Test\0";
        assert_eq!(pixel_name(&psv(52, 0, 999, 0, 0, strings)), None);
        assert_eq!(pixel_name(&psv(52, 0, 0, 0, 0, strings)), None);
        assert_eq!(pixel_name(&psv(52, 0, 1, 1, 8, strings)), None);
        assert_eq!(pixel_name(&psv(52, 0, 1, 0, 0, b"\0unterminated")), None);
        assert_eq!(pixel_name(&psv(52, 0, 1, 0, 0, b"\0\xff\0")), None);

        let mut part = psv(52, 0, 1, 1, 24, strings);
        for at in 0..part.len() {
            assert_eq!(pixel_name(&part[..at]), None, "truncated at {at}");
        }
        part[56..64].fill(0xff);
        assert_eq!(pixel_name(&part), None);
    }
}
