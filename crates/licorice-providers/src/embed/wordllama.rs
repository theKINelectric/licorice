//! WordLlama `l2_supercat` (256-d): token embeddings from Llama-2's vocabulary, averaged
//! over the text and L2-normalized. MIT-licensed weights ship inside the `wordllama`
//! PyPI package; `scripts/fetch-wordllama.sh` extracts them. It is a static embedder
//! (no attention), far cheaper than a transformer and weaker on retrieval (MTEB
//! CQADupStack 24.2 vs all-MiniLM-L6-v2's 41.3); Licorice uses it where no
//! transformer model can be downloaded.

use std::path::Path;

use tokenizers::Tokenizer;

use super::{EmbedError, EmbedKind, Embedder};

const WEIGHTS: &str = "l2_supercat_256.safetensors";
const TOKENIZER: &str = "l2_supercat_tokenizer_config.json";

pub struct WordLlama {
    tokenizer: Tokenizer,
    /// Row-major `[vocab, dim]`.
    table: Vec<f32>,
    vocab: usize,
    dim: usize,
}

impl WordLlama {
    /// Loads the weights and tokenizer from `dir` (as written by the fetch script).
    pub fn load(dir: &Path) -> Result<WordLlama, EmbedError> {
        let tokenizer = Tokenizer::from_file(dir.join(TOKENIZER)).map_err(|e| EmbedError::Load(e.to_string()))?;
        let bytes = std::fs::read(dir.join(WEIGHTS)).map_err(|e| EmbedError::Load(format!("{WEIGHTS}: {e}")))?;
        let (table, vocab, dim) = read_f16_matrix(&bytes, "embedding.weight")?;
        Ok(WordLlama { tokenizer, table, vocab, dim })
    }

    /// Embeds synchronously; each vector is unit length (or zero for empty input).
    pub fn embed_sync(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbedError> {
        texts
            .iter()
            .map(|text| {
                // add_special_tokens = false, as the reference implementation does.
                let enc = self.tokenizer.encode(text.as_str(), false).map_err(|e| EmbedError::Embed(e.to_string()))?;
                let mut v = vec![0f32; self.dim];
                let ids = enc.get_ids();
                for &id in ids {
                    let row = (id as usize).min(self.vocab - 1) * self.dim;
                    for (acc, x) in v.iter_mut().zip(&self.table[row..row + self.dim]) {
                        *acc += x;
                    }
                }
                let n = (ids.len().max(1)) as f32;
                v.iter_mut().for_each(|x| *x /= n);
                let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 {
                    v.iter_mut().for_each(|x| *x /= norm);
                }
                Ok(v)
            })
            .collect()
    }
}

#[async_trait::async_trait]
impl Embedder for WordLlama {
    fn id(&self) -> &str {
        "wordllama/l2_supercat_256"
    }

    fn dim(&self) -> usize {
        self.dim
    }

    async fn embed(&self, texts: &[String], _kind: EmbedKind) -> Result<Vec<Vec<f32>>, EmbedError> {
        self.embed_sync(texts)
    }
}

/// Reads one F16 2-D tensor from a safetensors file: an 8-byte little-endian header
/// length, a JSON header, then the raw tensor bytes.
fn read_f16_matrix(bytes: &[u8], name: &str) -> Result<(Vec<f32>, usize, usize), EmbedError> {
    let bad = |m: &str| EmbedError::Load(format!("{WEIGHTS}: {m}"));
    let header_len = u64::from_le_bytes(bytes.get(..8).ok_or_else(|| bad("truncated"))?.try_into().unwrap()) as usize;
    let header: serde_json::Value =
        serde_json::from_slice(bytes.get(8..8 + header_len).ok_or_else(|| bad("truncated header"))?)
            .map_err(|e| bad(&e.to_string()))?;
    let t = header.get(name).ok_or_else(|| bad(&format!("no tensor {name}")))?;
    if t["dtype"] != "F16" {
        return Err(bad(&format!("{name} is {}, expected F16", t["dtype"])));
    }
    let shape: Vec<usize> =
        t["shape"].as_array().into_iter().flatten().filter_map(|v| v.as_u64()).map(|v| v as usize).collect();
    let offsets: Vec<usize> =
        t["data_offsets"].as_array().into_iter().flatten().filter_map(|v| v.as_u64()).map(|v| v as usize).collect();
    let [vocab, dim] = shape[..] else { return Err(bad("expected a 2-D tensor")) };
    let [start, end] = offsets[..] else { return Err(bad("bad data_offsets")) };
    let data = bytes.get(8 + header_len + start..8 + header_len + end).ok_or_else(|| bad("truncated data"))?;
    if data.len() != vocab * dim * 2 {
        return Err(bad("tensor size does not match its shape"));
    }
    let table = data.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect();
    Ok((table, vocab, dim))
}

/// IEEE 754 half to single precision.
fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1f) as u32;
    let frac = (h & 0x3ff) as u32;
    let bits = match exp {
        0 if frac == 0 => sign,
        0 => {
            // Subnormal: normalize the fraction.
            let mut e = 127 - 15 + 1;
            let mut f = frac;
            while f & 0x400 == 0 {
                f <<= 1;
                e -= 1;
            }
            sign | (e << 23) | ((f & 0x3ff) << 13)
        }
        0x1f => sign | 0x7f80_0000 | (frac << 13),
        _ => sign | ((exp + 127 - 15) << 23) | (frac << 13),
    };
    f32::from_bits(bits)
}

#[cfg(test)]
mod tests {
    use super::f16_to_f32;

    #[test]
    fn half_precision_conversion() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x7bff), 65504.0);
        assert!((f16_to_f32(0x0001) - 5.960_464_5e-8).abs() < 1e-12, "smallest subnormal");
        assert!(f16_to_f32(0x7c00).is_infinite());
    }
}
