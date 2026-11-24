//! Rectangle lists drawn from the vertex id: the console's rectangle
//! primitive is three vertices, the fourth corner being `v1 + v2 - v0`.
//! Draws with vertex buffers are expanded on the CPU (`draw::rectangles`).
//! A vertex shader that makes its corners from `SV_VertexID` cannot be: its
//! pipeline gets a second host pipeline with a geometry shader that turns
//! each triangle of three corners into the strip of four. The vertex counts
//! stay the title's, so indirect draws need no change to their arguments:
//! Grand Theft Auto VI lights its tiles this way (`VS_TiledLighting`, an
//! indirect rectangle list per material), and without it the lit scene
//! target stays empty.
//!
//! shortcut: only pipelines whose vertex shader outputs `SV_Position` alone
//! (what a pixel shader that loads by pixel needs); others keep the old
//! handling. The shader is HLSL compiled by DXC when the pipeline is made,
//! or read from `rect-gs-<source hash>.dxil` beside the executable (made
//! ahead with `dxc -T gs_6_0 -E main -O3`), as the argument clamp is.

const SOURCE: &str = r#"
struct V { float4 p : SV_Position; };
[maxvertexcount(4)]
void main(triangle V i[3], inout TriangleStream<V> o) {
    V v;
    v.p = i[0].p; o.Append(v);
    v.p = i[1].p; o.Append(v);
    v.p = i[2].p; o.Append(v);
    v.p = i[1].p + i[2].p - i[0].p; o.Append(v);
}
"#;

/// The file name of the precompiled shader for this very [`SOURCE`].
fn precompiled_name() -> String {
    let hash = blake3::hash(SOURCE.as_bytes()).to_hex();
    format!("rect-gs-{}.dxil", &hash[..16])
}

/// The geometry shader's container, made once; `None` if no compiler is at
/// hand and no precompiled file either (logged once).
pub(crate) fn geometry_shader() -> Option<&'static [u8]> {
    static MADE: std::sync::OnceLock<Option<Vec<u8>>> = std::sync::OnceLock::new();
    MADE.get_or_init(|| {
        let compiled = crate::indirect::dxc_compile_as(SOURCE, windows::core::w!("gs_6_0"));
        let mut bytes = match compiled {
            Ok(bytes) => {
                tracing::info!(bytes = bytes.len(), "rectangle geometry shader compiled by DXC (gs_6_0)");
                bytes
            }
            Err(e) => {
                let path = std::env::current_exe().ok()?.parent()?.join(precompiled_name());
                match std::fs::read(&path) {
                    Ok(bytes) if bytes.starts_with(b"DXBC") => {
                        tracing::info!(path = %path.display(), "rectangle geometry shader: precompiled container");
                        bytes
                    }
                    _ => {
                        tracing::warn!(
                            error = %e,
                            file = %precompiled_name(),
                            "no rectangle geometry shader (no DXC, no precompiled file): rectangle lists made \
                             from the vertex id are not drawn"
                        );
                        return None;
                    }
                }
            }
        };
        if bytes.len() > 20 && bytes[4..20].iter().all(|&b| b == 0) {
            victory_shader::emit::sign(&mut bytes);
        }
        Some(bytes)
    })
    .as_deref()
}

/// Whether a translated vertex shader's only output is `SV_Position` (its
/// `OSG1` part has one element, of the position system value).
pub(crate) fn position_only(dxil: &[u8]) -> bool {
    let u32_at = |b: &[u8], at: usize| {
        b.get(at..at + 4)
            .map(|v| u32::from_le_bytes(v.try_into().expect("4 bytes")) as usize)
    };
    let part = (0..u32_at(dxil, 28).unwrap_or(0)).find_map(|i| {
        let at = u32_at(dxil, 32 + 4 * i)?;
        (dxil.get(at..at + 4)? == b"OSG1")
            .then(|| dxil.get(at + 8..at + 8 + u32_at(dxil, at + 4)?))
            .flatten()
    });
    const POSITION: usize = 1;
    part.is_some_and(|data| u32_at(data, 0) == Some(1) && u32_at(data, 8 + 12) == Some(POSITION))
}
