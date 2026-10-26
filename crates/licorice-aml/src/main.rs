use std::path::PathBuf;
use std::sync::Arc;

use licorice_aml::{Adapter, Config};
use licorice_providers::chat_http::BudgetChat;
use licorice_providers::embed::{DashScopeV4, Embedder, WordLlama};

fn selected_setting(name: &str) -> Option<String> {
    std::env::var(name).ok().or_else(|| {
        let file = std::env::var("LICORICE_AML_ENV_FILE").unwrap_or_else(|_| ".env".into());
        dotenvy::from_path_iter(file).ok()?.filter_map(Result::ok).find(|(key, _)| key == name).map(|(_, value)| value)
    })
}

fn embedding_choice(profile: &str, choice: &str) -> Result<(), &'static str> {
    match (profile, choice) {
        (_, "wordllama") => Ok(()),
        ("temporal", "dashscope-v4") => Ok(()),
        ("raw", "dashscope-v4") => Err("raw is offline; use the explicit temporal profile for DashScope embeddings"),
        _ => Err("LICORICE_AML_EMBEDDER must be wordllama or dashscope-v4"),
    }
}

fn scorer_pool(value: Option<&str>) -> Result<usize, String> {
    let pool = value
        .unwrap_or("24")
        .parse::<usize>()
        .map_err(|_| "LICORICE_AML_RERANK_CANDIDATES must be an integer in 1..=96")?;
    if !(1..=96).contains(&pool) {
        return Err("LICORICE_AML_RERANK_CANDIDATES must be in 1..=96".into());
    }
    Ok(pool)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Raw is offline. Model credentials are loaded only for the explicit temporal profile.
    let api_key =
        std::env::var("LICORICE_AML_API_KEY").map_err(|_| "set LICORICE_AML_API_KEY for Add/Search authentication")?;
    let data_dir = PathBuf::from(std::env::var("LICORICE_AML_DATA_DIR").unwrap_or_else(|_| "data/aml".into()));
    let models = PathBuf::from(std::env::var("LICORICE_AML_MODELS").unwrap_or_else(|_| "models/wordllama".into()));
    let budget_tokens = std::env::var("LICORICE_AML_BUDGET_TOKENS").unwrap_or_else(|_| "2000".into()).parse()?;
    let bind: std::net::SocketAddr =
        std::env::var("LICORICE_AML_BIND").unwrap_or_else(|_| "127.0.0.1:8099".into()).parse()?;
    let profile = std::env::var("LICORICE_AML_PROFILE").unwrap_or_else(|_| "raw".into());
    let embedding = std::env::var("LICORICE_AML_EMBEDDER").unwrap_or_else(|_| "wordllama".into());
    embedding_choice(&profile, &embedding)?;
    let ledger = PathBuf::from(
        std::env::var("LICORICE_AML_LEDGER")
            .unwrap_or_else(|_| data_dir.join("model-spend.sqlite").display().to_string()),
    );
    let ceiling = if profile == "temporal" {
        std::env::var("LICORICE_AML_BUDGET_USD").unwrap_or_else(|_| "25".into()).parse()?
    } else {
        25.
    };
    let embedder: Arc<dyn Embedder> = if embedding == "dashscope-v4" {
        let key =
            selected_setting("DASHSCOPE_API_KEY").ok_or("DASHSCOPE_API_KEY is required for DashScope embeddings")?;
        let base = selected_setting("DASHSCOPE_BASE_URL")
            .ok_or("DASHSCOPE_BASE_URL must select the matching workspace/region")?;
        let dimensions =
            std::env::var("LICORICE_AML_EMBEDDING_DIMENSIONS").unwrap_or_else(|_| "1024".into()).parse()?;
        Arc::new(DashScopeV4::open(key, &base, &ledger, ceiling, dimensions)?)
    } else {
        Arc::new(WordLlama::load(&models)?)
    };
    let adapter = match profile.as_str() {
        "raw" => Adapter::open(Config { data_dir, api_key, budget_tokens }, embedder)?,
        "temporal" => {
            let key =
                selected_setting("OPENROUTER_API_KEY").ok_or("OPENROUTER_API_KEY is required for temporal Add")?;
            let model = Arc::new(BudgetChat::open(key, &ledger, ceiling)?);
            let search = std::env::var("LICORICE_AML_SEARCH").unwrap_or_else(|_| "facts".into());
            let mut policy = licorice::retrieve::RetrieveConfig { budget_tokens, ..Default::default() };
            policy.selection = match std::env::var("LICORICE_AML_SELECTION").as_deref().unwrap_or("relevance") {
                "relevance" => licorice::retrieve::SelectionMode::Relevance,
                "mmr" => licorice::retrieve::SelectionMode::Mmr,
                _ => return Err("LICORICE_AML_SELECTION must be relevance or mmr".into()),
            };
            policy.chunks = std::env::var("LICORICE_AML_MAX_EXCERPTS")
                .ok()
                .map(|value| value.parse::<usize>().map_err(|_| "LICORICE_AML_MAX_EXCERPTS must be in 0..=100"))
                .transpose()?
                .unwrap_or(policy.chunks);
            if policy.chunks > 100 {
                return Err("LICORICE_AML_MAX_EXCERPTS must be in 0..=100".into());
            }
            policy.compact_excerpt_headers = std::env::var("LICORICE_AML_COMPACT_EXCERPT_HEADERS")
                .ok()
                .map(|value| {
                    value.parse::<bool>().map_err(|_| "LICORICE_AML_COMPACT_EXCERPT_HEADERS must be true or false")
                })
                .transpose()?
                .unwrap_or(false);
            policy.filter_state_window = std::env::var("LICORICE_AML_STATE_WINDOW")
                .ok()
                .map(|v| v.parse::<bool>())
                .transpose()?
                .unwrap_or(false);
            let reranker: Option<Arc<dyn licorice_providers::rerank::Reranker>> = match search.as_str() {
                "facts" => None,
                "connected" => {
                    policy.graph_hops = 2;
                    policy.connected_evidence = true;
                    None
                }
                "reranked" => {
                    policy.graph_hops = 2;
                    policy.connected_evidence = true;
                    policy.rerank_fallback = true;
                    let pool = scorer_pool(std::env::var("LICORICE_AML_RERANK_CANDIDATES").ok().as_deref())?;
                    policy.rerank_candidates = pool;
                    policy.selection_candidates = policy.selection_candidates.max(pool);
                    let scorer: Arc<dyn licorice_providers::rerank::Reranker> =
                        Arc::new(licorice_providers::rerank::ChatReranker(model.clone()));
                    Some(if pool > 24 {
                        Arc::new(licorice_providers::rerank::BatchedReranker(scorer))
                            as Arc<dyn licorice_providers::rerank::Reranker>
                    } else {
                        scorer
                    })
                }
                _ => return Err("LICORICE_AML_SEARCH must be facts, connected or reranked".into()),
            };
            Adapter::open_with_model(Config { data_dir, api_key, budget_tokens }, embedder, model)?
                .with_search_policy(policy, reranker)?
        }
        _ => return Err("LICORICE_AML_PROFILE must be raw or temporal".into()),
    };
    let listener = tokio::net::TcpListener::bind(bind).await?;
    eprintln!("Licorice AML {profile} listening on {} (budget {budget_tokens})", listener.local_addr()?);
    axum::serve(listener, adapter.router())
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn remote_embedding_is_explicit_and_raw_stays_offline() {
        assert!(embedding_choice("raw", "wordllama").is_ok());
        assert!(embedding_choice("temporal", "wordllama").is_ok());
        assert!(embedding_choice("temporal", "dashscope-v4").is_ok());
        assert!(embedding_choice("raw", "dashscope-v4").is_err());
        assert!(embedding_choice("temporal", "unknown").is_err());
    }

    #[test]
    fn scorer_pool_defaults_to24_and_refuses_unbounded_or_invalid_values() {
        assert_eq!(scorer_pool(None).unwrap(), 24);
        assert_eq!(scorer_pool(Some("60")).unwrap(), 60);
        assert_eq!(scorer_pool(Some("96")).unwrap(), 96);
        for value in ["0", "97", "-1", "NaN", "1.5"] {
            assert!(scorer_pool(Some(value)).is_err());
        }
    }
}
