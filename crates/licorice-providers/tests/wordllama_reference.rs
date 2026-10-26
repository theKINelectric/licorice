//! The Rust WordLlama port must match the reference `wordllama` Python package
//! (0.4.0.post1). `tests/fixtures/wordllama_reference.json` holds its normalized
//! embeddings for a few texts. Needs the model files from `scripts/fetch-wordllama.sh`
//! (or `LICORICE_WORDLLAMA_DIR`); skipped with a message when they're absent.

use std::path::PathBuf;

use licorice_providers::embed::{EmbedKind, Embedder, WordLlama};

fn model_dir() -> Option<PathBuf> {
    let dir = std::env::var_os("LICORICE_WORDLLAMA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/wordllama"));
    dir.join("l2_supercat_256.safetensors").exists().then_some(dir)
}

#[tokio::test]
async fn matches_the_python_reference() {
    let Some(dir) = model_dir() else {
        eprintln!("skipped: WordLlama files not found; run scripts/fetch-wordllama.sh");
        return;
    };
    let model = WordLlama::load(&dir).unwrap();
    assert_eq!(model.dim(), 256);
    let fixture: serde_json::Value = serde_json::from_str(include_str!("fixtures/wordllama_reference.json")).unwrap();
    let texts: Vec<String> =
        fixture["texts"].as_array().unwrap().iter().map(|t| t.as_str().unwrap().to_owned()).collect();
    let expected: Vec<Vec<f32>> = fixture["vectors"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect())
        .collect();
    let got = model.embed(&texts, EmbedKind::Document).await.unwrap();
    for ((text, want), have) in texts.iter().zip(&expected).zip(&got) {
        let cos: f32 = want.iter().zip(have).map(|(a, b)| a * b).sum();
        let max_diff = want.iter().zip(have).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(cos > 0.99999 && max_diff < 1e-5, "{text:?}: cos {cos}, max diff {max_diff}");
    }
}
