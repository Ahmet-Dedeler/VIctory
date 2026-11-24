//! Rectangle lists the host draws with a geometry shader: the console's
//! rectangle primitive is three vertices, the fourth corner being the sum
//! of two less the third. Direct draws with vertex buffers are expanded on
//! the CPU (`draw::rectangles`). A vertex shader that makes its corners
//! from `SV_VertexID` cannot be, nor can an indirect draw (its vertices are
//! counted on the GPU) or an indexed one: such a pipeline gets a second
//! host pipeline with a geometry shader that turns each triangle of three
//! corners into the strip of four. The vertex counts stay the title's, so
//! indirect draws need no change to their arguments: Grand Theft Auto VI
//! lights its tiles this way (`VS_TiledLighting`, an indirect rectangle
//! list per material) and blends its indirect light in with an indexed
//! one, and without them the lit scene target stays empty.
//!
//! The shader is made for the vertex shader's output signature (as
//! upstream PR #1020 does): every output passes through, the fourth
//! vertex's interpolated values are those of the two vertices beside the
//! middle one less the middle one's. It is HLSL compiled by DXC when the
//! pipeline is made, or read from `rect-gs-<source hash>.dxil` beside the
//! executable (made ahead with `dxc -T gs_6_0 -E main -O3`), as the
//! argument clamp is.
//!
//! shortcut: a vertex shader whose outputs our source cannot name (masks
//! that do not start at x, other component types) keeps the old handling.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// One element of a shader container's signature part.
struct Element {
    name: String,
    index: u32,
    /// 1 unsigned, 2 signed, 3 float.
    kind: u32,
    register: u32,
    mask: u8,
}

/// The elements of the output signature (`OSG1`, 32-byte records after a
/// stream number) of shader container `code`.
fn outputs(code: &[u8]) -> Option<Vec<Element>> {
    let word = |b: &[u8], at: usize| {
        b.get(at..at + 4)
            .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]) as usize)
    };
    if code.get(..4)? != b"DXBC" {
        return None;
    }
    for n in 0..word(code, 28)? {
        let at = word(code, 32 + n * 4)?;
        if code.get(at..at + 4)? != b"OSG1" {
            continue;
        }
        let d = code.get(at + 8..at + 8 + word(code, at + 4)?)?;
        let mut out = Vec::new();
        for k in 0..word(d, 0)? {
            let e = 8 + k * 32 + 4;
            let name_at = word(d, e)?;
            let end = d.get(name_at..)?.iter().position(|&b| b == 0)?;
            out.push(Element {
                name: String::from_utf8_lossy(&d[name_at..name_at + end]).into_owned(),
                index: word(d, e + 4)? as u32,
                kind: word(d, e + 12)? as u32,
                register: word(d, e + 16)? as u32,
                mask: *d.get(e + 20)?,
            });
        }
        return Some(out);
    }
    None
}

/// The geometry shader's source for a vertex shader with these outputs;
/// `None` without a position or with an element the source cannot name.
///
/// Which of a rectangle's three vertices sits between the other two
/// differs by convention, so the corner opposite the longest side is taken
/// as the middle one.
fn source(outputs: &[Element]) -> Option<String> {
    let mut fields = String::new();
    let mut fourth = String::new();
    let mut position = None;
    for (n, e) in outputs.iter().enumerate() {
        let comps = e.mask.count_ones();
        // Elements start at x and are contiguous in what we emit.
        if comps == 0 || u32::from(e.mask) != (1u32 << comps) - 1 {
            return None;
        }
        let ty = match e.kind {
            1 => "uint",
            2 => "int",
            3 => "float",
            _ => return None,
        };
        if !e.name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
            return None;
        }
        let system = e.name.to_ascii_uppercase().starts_with("SV_");
        let semantic = if system && e.index == 0 {
            e.name.clone()
        } else {
            format!("{}{}", e.name, e.index)
        };
        let interpolation = if e.kind == 3 { "" } else { "nointerpolation " };
        fields.push_str(&format!("    {interpolation}{ty}{comps} e{n} : {semantic};\n"));
        if e.kind == 3 {
            fourth.push_str(&format!("    d.e{n} = i[s].e{n} + i[t].e{n} - i[r].e{n};\n"));
        } else {
            fourth.push_str(&format!("    d.e{n} = i[r].e{n};\n"));
        }
        if e.name.eq_ignore_ascii_case("SV_Position") && comps == 4 {
            position = Some(n);
        }
    }
    let p = position?;
    // Diagnostics (`VICTORY_RECTS_FULL`): the first rectangle of a draw
    // covers the whole target wherever the vertex shader put it, to tell a
    // vertex shader that places no rectangle from a draw that does not run.
    #[cfg(feature = "diagnostics")]
    if std::env::var_os("VICTORY_RECTS_FULL").is_some() {
        return Some(format!(
            "struct V {{\n{fields}}};\n\
             [maxvertexcount(4)]\n\
             void main(triangle V i[3], uint n : SV_PrimitiveID, inout TriangleStream<V> o) {{\n\
             \x20   if (n != 0) return;\n\
             \x20   V d = i[0];\n\
             \x20   d.e{p} = float4(-1, -1, 0.5, 1); o.Append(d);\n\
             \x20   d.e{p} = float4(-1, 1, 0.5, 1); o.Append(d);\n\
             \x20   d.e{p} = float4(1, -1, 0.5, 1); o.Append(d);\n\
             \x20   d.e{p} = float4(1, 1, 0.5, 1); o.Append(d);\n\
             }}\n"
        ));
    }
    Some(format!(
        "struct V {{\n{fields}}};\n\
         [maxvertexcount(4)]\n\
         void main(triangle V i[3], inout TriangleStream<V> o) {{\n\
         \x20   float2 a = i[0].e{p}.xy / i[0].e{p}.w;\n\
         \x20   float2 b = i[1].e{p}.xy / i[1].e{p}.w;\n\
         \x20   float2 c = i[2].e{p}.xy / i[2].e{p}.w;\n\
         \x20   float d01 = dot(a - b, a - b), d02 = dot(a - c, a - c), d12 = dot(b - c, b - c);\n\
         \x20   uint r = (d12 >= d01 && d12 >= d02) ? 0 : ((d02 >= d01) ? 1 : 2);\n\
         \x20   uint s = (r + 1) % 3, t = (r + 2) % 3;\n\
         \x20   V d;\n{fourth}\
         \x20   o.Append(i[r]); o.Append(i[s]); o.Append(i[t]); o.Append(d);\n\
         }}\n"
    ))
}

/// The file name of the precompiled shader for `source`.
fn precompiled_name(source: &str) -> String {
    let hash = blake3::hash(source.as_bytes()).to_hex();
    format!("rect-gs-{}.dxil", &hash[..16])
}

/// The geometry shader completing the rectangles of translated vertex
/// shader `vs` (its container), made once per output signature; `None` if
/// the signature has no source, no compiler is at hand and no precompiled
/// file either, or the compiled shader hands the pixel shader other
/// registers than the vertex shader does (each logged).
pub(crate) fn geometry_shader(vs: &[u8]) -> Option<Arc<[u8]>> {
    static MADE: Mutex<Option<HashMap<String, Option<Arc<[u8]>>>>> = Mutex::new(None);
    let outputs = outputs(vs)?;
    let source = source(&outputs)?;
    let mut made = MADE.lock().unwrap_or_else(|e| e.into_inner());
    let made = made.get_or_insert_with(HashMap::new);
    if let Some(done) = made.get(&source) {
        return done.clone();
    }
    let bytes = (|| {
        let compiled = crate::indirect::dxc_compile_as(&source, windows::core::w!("gs_6_0"));
        let mut bytes = match compiled {
            Ok(bytes) => bytes,
            Err(e) => {
                let name = precompiled_name(&source);
                let path = std::env::current_exe().ok()?.parent()?.join(&name);
                match std::fs::read(&path) {
                    Ok(bytes) if bytes.starts_with(b"DXBC") => bytes,
                    _ => {
                        tracing::warn!(
                            error = %e,
                            file = %name,
                            outputs = outputs.len(),
                            "no rectangle geometry shader (no DXC, no precompiled file): these rectangle \
                             lists are not drawn"
                        );
                        return None;
                    }
                }
            }
        };
        if bytes.len() > 20 && bytes[4..20].iter().all(|&b| b == 0) {
            victory_shader::emit::sign(&mut bytes);
        }
        // The shader must hand the pixel shader what the vertex shader
        // did, in the same registers.
        let same = self::outputs(&bytes).is_some_and(|o| {
            o.len() == outputs.len()
                && o.iter().zip(&outputs).all(|(x, y)| {
                    (x.register, x.mask, x.index) == (y.register, y.mask, y.index)
                        && x.name.eq_ignore_ascii_case(&y.name)
                })
        });
        if !same {
            tracing::warn!(outputs = outputs.len(), "rectangle geometry shader: registers differ from the vertex shader's");
            return None;
        }
        tracing::info!(outputs = outputs.len(), bytes = bytes.len(), file = %precompiled_name(&source), "rectangle geometry shader made (gs_6_0)");
        Some(Arc::<[u8]>::from(bytes))
    })();
    made.insert(source, bytes.clone());
    bytes
}

/// Diagnostics (`VICTORY_RECTS_FLAT`): a pixel shader that writes one
/// colour, for rectangle variants, to see what such draws cover whatever
/// their own pixel shader computes.
#[cfg(feature = "diagnostics")]
pub(crate) fn flat_pixel_shader() -> Option<&'static [u8]> {
    static MADE: std::sync::OnceLock<Option<Vec<u8>>> = std::sync::OnceLock::new();
    MADE.get_or_init(|| {
        std::env::var_os("VICTORY_RECTS_FLAT")?;
        let source = "float4 main() : SV_Target { return float4(1.0, 0.5, 0.25, 1.0); }\n";
        let mut bytes = crate::indirect::dxc_compile_as(source, windows::core::w!("ps_6_0"))
            .inspect_err(|e| tracing::warn!(error = %e, "no flat pixel shader"))
            .ok()?;
        if bytes.len() > 20 && bytes[4..20].iter().all(|&b| b == 0) {
            victory_shader::emit::sign(&mut bytes);
        }
        tracing::info!(bytes = bytes.len(), "VICTORY_RECTS_FLAT: rectangle variants write one colour");
        Some(bytes)
    })
    .as_deref()
}

/// The source for a vertex shader that outputs a position alone, for
/// making its precompiled file (`cargo test -p victory-gpu position_only_source -- --nocapture`).
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn position_only_source() {
        let s = source(&[Element {
            name: "SV_Position".into(),
            index: 0,
            kind: 3,
            register: 0,
            mask: 0xf,
        }])
        .unwrap();
        println!("{}\n{s}", precompiled_name(&s));
        assert!(s.contains("float4 e0 : SV_Position;"));
        assert!(s.contains("d.e0 = i[s].e0 + i[t].e0 - i[r].e0;"));
    }
}
