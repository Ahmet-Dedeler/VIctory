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

/// Every distinct compute container in the data file translated, one line
/// each, to compare whole-pack coverage before and after a change:
///
/// ```sh
/// VICTORY_SHADER_SCAN=/path/to/shaders.rpf VICTORY_HOST_MAX_WAVE=32 \
/// VICTORY_SCAN_THREADS=4 VICTORY_SCAN_WAVE64=1 \
/// cargo test -p victory-shader --test offline scan_compute -- --ignored --nocapture
/// ```
///
/// Lines read `SCAN <hash> <stage> wave64=<bool> threads=<x,y,z> <result> <name>`,
/// the result `OK wave=<n|any> bytes=<n> out=<hash of the host DXIL>`,
/// `ERROR <why>` or `PANIC`. `VICTORY_SCAN_WAVE64=1` keeps only programs the
/// console runs as 64 lanes; `VICTORY_SCAN_STAGE=all` takes the vertex, pixel
/// and mesh programs too.
#[test]
#[ignore = "reads a game data file named by VICTORY_SHADER_SCAN"]
fn scan_compute_from_env() {
    let path = std::env::var_os("VICTORY_SHADER_SCAN")
        .expect("set VICTORY_SHADER_SCAN to the data file to scan");
    if let Some(n) = std::env::var("VICTORY_HOST_MAX_WAVE")
        .ok()
        .and_then(|v| v.parse().ok())
    {
        emit::set_host_max_wave(n);
    }
    let only64 = std::env::var("VICTORY_SCAN_WAVE64").is_ok_and(|v| v == "1");
    let every_stage = std::env::var("VICTORY_SCAN_STAGE").is_ok_and(|v| v == "all");
    let threads: usize = std::env::var("VICTORY_SCAN_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2)
        .max(1);
    let data = std::fs::read(&path).expect("read VICTORY_SHADER_SCAN");

    // (hash, name, stage, wave64, threads, bytes) of each distinct container.
    type Job = (String, String, victory_shader::native::HwStage, bool, [u32; 3], Vec<u8>);
    let mut jobs: Vec<Job> = Vec::new();
    let mut seen = HashSet::new();
    let mut take = |bytes: Vec<u8>, name: String| {
        let hash = blake3::hash(&bytes).to_hex()[..16].to_owned();
        if !seen.insert(hash.clone()) {
            return;
        }
        let Ok(container) = Container::parse(&bytes) else {
            return;
        };
        let Some(Ok(programs)) = container
            .part(FourCC::DXIL)
            .map(|p| native_programs(p.data))
        else {
            return;
        };
        use victory_shader::native::HwStage;
        let Some(cs) = programs.iter().find(|p| {
            p.stage == HwStage::Cs
                || every_stage && matches!(p.stage, HwStage::Vs | HwStage::Ps | HwStage::Ms)
        }) else {
            return;
        };
        if only64 && !cs.wave64 {
            return;
        }
        let threads = match &cs.io {
            victory_shader::native::StageIo::Cs(io) => io.threads,
            _ => [0; 3],
        };
        jobs.push((hash, name, cs.stage, cs.wave64, threads, bytes));
    };
    if sgd5::is_sgd5(&data) {
        for record in sgd5::records(&data) {
            if let Some(bytes) = record.unpack() {
                take(bytes, record.name.unwrap_or("?").to_owned());
            }
        }
    }
    for (offset, bytes) in raw_containers(&data) {
        take(bytes.to_vec(), format!("@{offset:#x}"));
    }
    drop(data);
    jobs.sort_by(|a, b| a.0.cmp(&b.0));
    println!("containers: {}", jobs.len());

    let next = std::sync::atomic::AtomicUsize::new(0);
    let lines = std::sync::Mutex::new(Vec::new());
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| {
                loop {
                    let n = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some((hash, name, stage, wave64, threads, bytes)) = jobs.get(n) else {
                        break;
                    };
                    let result = match std::panic::catch_unwind(|| emit::translate(bytes)) {
                        Ok(Ok(shader)) => {
                            let wave = Container::parse(&shader.dxil)
                                .ok()
                                .and_then(|c| c.part(FourCC::PSV0).map(|p| p.data))
                                .filter(|d| d.len() >= 28)
                                .map(|d| {
                                    let min = u32::from_le_bytes(d[20..24].try_into().unwrap());
                                    let max = u32::from_le_bytes(d[24..28].try_into().unwrap());
                                    (min, max)
                                })
                                .and_then(|(min, max)| (min == max && min != 0).then_some(min));
                            format!(
                                "OK wave={} bytes={} out={}",
                                wave.map_or("any".to_owned(), |n| n.to_string()),
                                shader.dxil.len(),
                                &blake3::hash(&shader.dxil).to_hex()[..16]
                            )
                        }
                        Ok(Err(e)) => format!("ERROR {}", e.to_string().replace('\n', " ")),
                        Err(_) => "PANIC".to_owned(),
                    };
                    let line = format!(
                        "SCAN {hash} {stage:?} wave64={wave64} threads={},{},{} {result} {name}",
                        threads[0], threads[1], threads[2]
                    );
                    println!("{line}");
                    lines.lock().unwrap().push(line);
                }
            });
        }
    });
    let lines = lines.into_inner().unwrap();
    let ok = lines.iter().filter(|l| l.contains(" OK ")).count();
    let wave32 = lines.iter().filter(|l| l.contains(" OK wave=32 ")).count();
    let wave64 = lines.iter().filter(|l| l.contains(" OK wave=64 ")).count();
    println!(
        "scan complete: {} containers, {ok} translated ({wave32} host wave 32, {wave64} host wave 64), {} failed",
        lines.len(),
        lines.len() - ok
    );
}
