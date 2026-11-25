//! Offline translation of shader containers found in a data file.
//!
//! Run one scan explicitly, with the container hashes the run's log names:
//!
//! ```sh
//! VICTORY_SHADER_SCAN=/path/to/shaders.rpf \
//! VICTORY_TRANSLATE_ONLY=f133e35efed2a48a,369cbe906ea72764 \
//! VICTORY_HOST_MAX_WAVE=32 VICTORY_WAVE_WHY=1 \
//! cargo test -p victory-shader --test offline -- --ignored --nocapture
//! ```

use std::collections::HashSet;

use victory_game::sgd5;
use victory_shader::{Container, FourCC, emit, native_programs};

/// Every raw DXBC container in `data` as `(offset, bytes)`. Archives keep
/// some containers uncompressed; their headers carry the total size.
fn raw_containers(data: &[u8]) -> Vec<(usize, &[u8])> {
    let mut out = Vec::new();
    for offset in memchr::memmem::find_iter(data, b"DXBC") {
        let Some(header) = data.get(offset..offset + 0x20) else {
            continue;
        };
        let size = u32::from_le_bytes(header[0x18..0x1c].try_into().expect("4 bytes")) as usize;
        let Some(bytes) = data.get(offset..offset + size) else {
            continue;
        };
        if Container::parse(bytes).is_ok() {
            out.push((offset, bytes));
        }
    }
    out
}

/// Translates `bytes` when its container hash matches `wanted`; returns the
/// hash then, once per distinct container.
fn translate_one(
    bytes: &[u8],
    at: &str,
    wanted: &[String],
    seen: &mut HashSet<String>,
) -> Option<String> {
    let hash = blake3::hash(bytes).to_hex()[..16].to_owned();
    if !seen.insert(hash.clone()) || !wanted.iter().any(|w| hash.starts_with(w)) {
        return None;
    }
    let container = Container::parse(bytes);
    match container
        .as_ref()
        .ok()
        .and_then(|c| c.part(FourCC::DXIL))
        .map(|p| native_programs(p.data))
    {
        Some(Ok(programs)) => {
            for p in &programs {
                println!(
                    "{hash} {at}: native {:?} wave64={} code={}B threads={:?}",
                    p.stage,
                    p.wave64,
                    p.code.len(),
                    match p.io {
                        victory_shader::native::StageIo::Cs(cs) => Some(cs.threads),
                        _ => None,
                    }
                );
                if let Ok(insts) = victory_shader::gfx10::decode(&p.code) {
                    let around = std::env::var("VICTORY_SHADER_DISASM")
                        .ok()
                        .and_then(|v| usize::from_str_radix(v.trim_start_matches("0x"), 16).ok());
                    let filter = std::env::var("VICTORY_SHADER_DISASM_FILTER").ok();
                    if around.is_some() || filter.is_some() {
                        for (n, inst) in insts.iter().enumerate() {
                            let text = inst.text(p.wave64);
                            if around.is_some_and(|want| inst.offset.abs_diff(want) <= 0x40)
                                || filter.as_ref().is_some_and(|f| text.contains(f))
                            {
                                let words: Vec<String> =
                                    inst.words.iter().map(|w| format!("{w:08X}")).collect();
                                println!(
                                    "{hash} {at}: {n:04} {:06x}: {:<44} {}",
                                    inst.offset,
                                    words.join(" "),
                                    text
                                );
                            }
                        }
                    }
                }
            }
        }
        Some(Err(e)) => println!("{hash} {at}: native unreadable: {e}"),
        None => println!("{hash} {at}: no DXIL part"),
    }
    match emit::translate(bytes) {
        Ok(shader) => {
            let wave = Container::parse(&shader.dxil)
                .ok()
                .and_then(|c| c.part(FourCC::PSV0).map(|p| p.data))
                .filter(|d| d.len() >= 28)
                .map(|d| {
                    let min = u32::from_le_bytes(d[20..24].try_into().expect("4 bytes"));
                    let max = u32::from_le_bytes(d[24..28].try_into().expect("4 bytes"));
                    (min, max)
                })
                .and_then(|(min, max)| (min == max && min != 0).then_some(min));
            println!(
                "{hash} {at}: TRANSLATED {:?}, host DXIL {}B, user data {} dwords, host wave {}",
                shader.stage,
                shader.dxil.len(),
                shader.user_data_dwords(),
                wave.map_or("unrestricted".to_owned(), |n| n.to_string())
            );
        }
        Err(e) => println!("{hash} {at}: ERROR {e}"),
    }
    Some(hash)
}

#[test]
#[ignore = "reads a game data file named by VICTORY_SHADER_SCAN"]
fn scan_containers_from_env() {
    let path = std::env::var_os("VICTORY_SHADER_SCAN")
        .expect("set VICTORY_SHADER_SCAN to the data file to scan");
    let wanted: Vec<String> = std::env::var("VICTORY_TRANSLATE_ONLY")
        .expect("set VICTORY_TRANSLATE_ONLY to comma-separated container hash prefixes")
        .split(',')
        .map(|h| h.trim().to_ascii_lowercase())
        .filter(|h| !h.is_empty())
        .collect();
    if let Some(n) = std::env::var("VICTORY_HOST_MAX_WAVE")
        .ok()
        .or_else(|| std::env::var("VICTORY_HOST_WAVE").ok())
        .and_then(|v| v.parse().ok())
    {
        emit::set_host_max_wave(n);
    }
    let data = std::fs::read(&path).expect("read VICTORY_SHADER_SCAN");
    let path = std::path::Path::new(&path);
    let mut seen = HashSet::new();
    let mut found = HashSet::new();
    if sgd5::is_sgd5(&data) {
        for record in sgd5::records(&data) {
            let Some(bytes) = record.unpack() else {
                continue;
            };
            let at = format!(
                "{}@{:#x} {}",
                path.display(),
                record.offset,
                record.name.unwrap_or("?")
            );
            if let Some(hash) = translate_one(&bytes, &at, &wanted, &mut seen) {
                found.insert(hash);
            }
        }
    }
    for (offset, bytes) in raw_containers(&data) {
        let at = format!("{}@{offset:#x}", path.display());
        if let Some(hash) = translate_one(bytes, &at, &wanted, &mut seen) {
            found.insert(hash);
        }
    }
    let missing: Vec<&str> = wanted
        .iter()
        .filter(|w| !found.iter().any(|h| h.starts_with(w.as_str())))
        .map(String::as_str)
        .collect();
    println!("scan complete: {} wanted containers", found.len());
    assert!(
        missing.is_empty(),
        "wanted containers not found: {}",
        missing.join(", ")
    );
}
