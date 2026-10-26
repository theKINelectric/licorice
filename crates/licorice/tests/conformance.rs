//! Engine-agnostic behaviour of the `Index` trait (ADR 0001). A second engine
//! (Turso, later) implements `Harness` and runs the same suite.

mod common;

use common::{Rng, brute_force, clustered, normalized, put_facts, put_vectors, space};
use licorice::{Filter, Index, LexicalKind, Store, StoreConfig, VectorKind};

/// What a test needs from an engine beyond `Index`: a way to load fixtures.
trait Harness: Index {
    async fn vectors(&self, kind: VectorKind, items: Vec<(i64, Vec<f32>)>);
    async fn facts(&self, statements: Vec<&'static str>) -> Vec<i64>;
}

impl Harness for Store {
    async fn vectors(&self, kind: VectorKind, items: Vec<(i64, Vec<f32>)>) {
        put_vectors(self, kind, items).await
    }

    async fn facts(&self, statements: Vec<&'static str>) -> Vec<i64> {
        put_facts(self, statements).await
    }
}

async fn exact_knn_matches_brute_force(h: &impl Harness) {
    let dim = 32;
    let items: Vec<(i64, Vec<f32>)> =
        clustered(7, 300, dim, 6, 0.5).into_iter().enumerate().map(|(i, v)| (i as i64 + 1, v)).collect();
    h.vectors(VectorKind::Fact, items.clone()).await;
    let mut rng = Rng::new(99);
    for _ in 0..10 {
        let q: Vec<f32> = (0..dim).map(|_| rng.gauss()).collect();
        let got: Vec<i64> =
            h.knn(VectorKind::Fact, &q, 10, Filter::None).await.unwrap().iter().map(|h| h.owner).collect();
        assert_eq!(got, brute_force(&items, &q, 10));
    }
}

async fn scores_are_cosine_and_sorted(h: &impl Harness) {
    let a = normalized(vec![1.0, 0.0, 0.0, 0.0]);
    let b = normalized(vec![1.0, 1.0, 0.0, 0.0]);
    let c = normalized(vec![0.0, 0.0, 1.0, 0.0]);
    h.vectors(VectorKind::Entity, vec![(1, a), (2, b), (3, c)]).await;
    let hits = h.knn(VectorKind::Entity, &[2.0, 0.0, 0.0, 0.0], 3, Filter::None).await.unwrap();
    let owners: Vec<i64> = hits.iter().map(|h| h.owner).collect();
    assert_eq!(owners, vec![1, 2, 3]);
    assert!((hits[0].score - 1.0).abs() < 1e-5, "{hits:?}");
    assert!((hits[1].score - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-5, "{hits:?}");
    assert!(hits[2].score.abs() < 1e-5, "{hits:?}");
}

async fn owner_filter_restricts_results(h: &impl Harness) {
    let items: Vec<(i64, Vec<f32>)> =
        clustered(3, 50, 8, 2, 0.3).into_iter().enumerate().map(|(i, v)| (i as i64 + 1, v)).collect();
    h.vectors(VectorKind::Chunk, items.clone()).await;
    let allowed = vec![5, 9, 13, 21];
    let q = items[8].1.clone();
    let hits = h.knn(VectorKind::Chunk, &q, 10, Filter::Owners(allowed.clone())).await.unwrap();
    assert_eq!(hits.len(), allowed.len());
    assert_eq!(hits[0].owner, 9, "the query's own vector ranks first");
    assert!(hits.iter().all(|h| allowed.contains(&h.owner)));
}

async fn empty_and_zero_k(h: &impl Harness) {
    assert!(h.knn(VectorKind::Summary, &[1.0, 2.0], 5, Filter::None).await.unwrap().is_empty());
    assert!(h.knn(VectorKind::Summary, &[1.0, 2.0], 0, Filter::None).await.unwrap().is_empty());
    assert!(h.lexical(LexicalKind::Fact, "anything", 5).await.unwrap().is_empty());
}

async fn lexical_ranks_and_survives_hostile_input(h: &impl Harness) {
    let ids = h
        .facts(vec![
            "Caroline is from Sweden",
            "Melanie borrowed an earring from Caroline",
            "The user drinks coffee every morning",
            "The user quit coffee in January",
        ])
        .await;
    let hits = h.lexical(LexicalKind::Fact, "Where is Caroline from?", 10).await.unwrap();
    assert_eq!(hits.first().map(|h| h.owner), Some(ids[0]), "{hits:?}");
    assert!(hits.iter().any(|h| h.owner == ids[1]));
    let coffee = h.lexical(LexicalKind::Fact, "coffee", 10).await.unwrap();
    assert_eq!(coffee.len(), 2);
    for nasty in ["\"", "AND OR NOT", "NEAR(a b)", "col:value", "*", "-coffee", "'; DROP TABLE facts; --", "", "   "] {
        h.lexical(LexicalKind::Fact, nasty, 5).await.unwrap_or_else(|e| panic!("{nasty:?}: {e}"));
    }
    let porter = h.lexical(LexicalKind::Fact, "borrowing earrings", 5).await.unwrap();
    assert_eq!(porter.first().map(|h| h.owner), Some(ids[1]), "stemming: {porter:?}");
}

macro_rules! conformance {
    ($($name:ident),* $(,)?) => {$(
        #[tokio::test(flavor = "multi_thread")]
        async fn $name() {
            let (_dir, path) = space();
            let mut cfg = StoreConfig::new(dimension_for(stringify!($name)));
            cfg.readers = 2;
            let store = Store::open(&path, cfg).await.unwrap();
            super::$name(&store).await;
            store.close().await.unwrap();
        }
    )*};
}

fn dimension_for(test: &str) -> usize {
    match test {
        "exact_knn_matches_brute_force" => 32,
        "scores_are_cosine_and_sorted" => 4,
        "owner_filter_restricts_results" => 8,
        "empty_and_zero_k" => 2,
        _ => 8,
    }
}

mod sqlite {
    use super::*;
    conformance!(
        exact_knn_matches_brute_force,
        scores_are_cosine_and_sorted,
        owner_filter_restricts_results,
        empty_and_zero_k,
        lexical_ranks_and_survives_hostile_input,
    );
}
