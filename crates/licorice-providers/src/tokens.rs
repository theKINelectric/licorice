//! Explicit counting for rendered memory. The default is a reproducible encoding,
//! not a claim that every answering model uses the same tokenizer.

use std::path::Path;
use std::sync::OnceLock;

#[derive(Debug, thiserror::Error)]
#[error("token counting: {0}")]
pub struct CountError(pub String);

pub trait TokenCounter: Send + Sync {
    fn id(&self) -> &str;
    fn count(&self, text: &str) -> Result<usize, CountError>;
}

/// Ordinary-text cl100k_base tokens; literal special-token strings are not treated
/// as control tokens. For a different reader, provide its own counter explicitly.
pub struct Cl100kCounter;

impl TokenCounter for Cl100kCounter {
    fn id(&self) -> &str {
        "cl100k_base:ordinary"
    }
    fn count(&self, text: &str) -> Result<usize, CountError> {
        static BPE: OnceLock<Result<tiktoken_rs::CoreBPE, String>> = OnceLock::new();
        let bpe = BPE.get_or_init(|| tiktoken_rs::cl100k_base().map_err(|e| e.to_string()));
        match bpe {
            Ok(bpe) => Ok(bpe.encode_ordinary(text).len()),
            Err(error) => Err(CountError(error.clone())),
        }
    }
}

/// A tokenizer.json for the configured reader. No special/chat-template tokens
/// are added: the budget covers the memory text, not the whole model request.
pub struct HfCounter {
    tokenizer: tokenizers::Tokenizer,
    id: String,
}

impl HfCounter {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, CountError> {
        let bytes = std::fs::read(path).map_err(|e| CountError(e.to_string()))?;
        Self::from_bytes(&bytes)
    }
    fn from_bytes(bytes: &[u8]) -> Result<Self, CountError> {
        let mut tokenizer = tokenizers::Tokenizer::from_bytes(bytes).map_err(|e| CountError(e.to_string()))?;
        // A saved tokenizer's truncation/padding must not hide an oversized input
        // or consume the budget with batch padding.
        tokenizer.with_truncation(None).map_err(|e| CountError(e.to_string()))?;
        tokenizer.with_padding(None);
        Ok(Self { tokenizer, id: format!("hf:blake3:{}", blake3::hash(bytes).to_hex()) })
    }
}

impl TokenCounter for HfCounter {
    fn id(&self) -> &str {
        &self.id
    }
    fn count(&self, text: &str) -> Result<usize, CountError> {
        self.tokenizer.encode(text, false).map(|e| e.len()).map_err(|e| CountError(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ordinary_tokens_handle_unicode_and_literal_control_strings() {
        assert_eq!(Cl100kCounter.count("hello world").unwrap(), 2);
        assert!(Cl100kCounter.count("<|endoftext|>").unwrap() > 1);
        assert!(Cl100kCounter.count("🍌 日本語 café").unwrap() > 0);
    }
    #[test]
    fn hf_counter_disables_saved_truncation_and_padding() {
        let json=br#"{"version":"1.0","truncation":{"direction":"Right","max_length":2,"strategy":"LongestFirst","stride":0},"padding":{"strategy":{"Fixed":8},"direction":"Right","pad_to_multiple_of":null,"pad_id":0,"pad_type_id":0,"pad_token":"[UNK]"},"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"Whitespace"},"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"[UNK]":0,"hello":1,"world":2},"unk_token":"[UNK]"}}"#;
        let counter = HfCounter::from_bytes(json).unwrap();
        assert_eq!(counter.count("hello world hello world").unwrap(), 4);
        assert!(counter.id().starts_with("hf:blake3:"));
    }
}
