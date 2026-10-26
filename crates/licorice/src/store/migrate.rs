//! Embedded schema migrations keyed by `PRAGMA user_version`.

use rusqlite::{Connection, OptionalExtension};

use crate::{Error, Result};

const MIGRATIONS: &[&str] = &[
    include_str!("../../migrations/0001_init.sql"),
    include_str!("../../migrations/0002_identity_applications.sql"),
    include_str!("../../migrations/0003_ingest_indexes.sql"),
    include_str!("../../migrations/0004_temporal_updates.sql"),
    include_str!("../../migrations/0005_state_corrections.sql"),
    include_str!("../../migrations/0006_update_targets.sql"),
    include_str!("../../migrations/0007_source_lookup.sql"),
    include_str!("../../migrations/0008_being_memory.sql"),
    include_str!("../../migrations/0009_being_relationships.sql"),
    include_str!("../../migrations/0010_being_ontology.sql"),
    include_str!("../../migrations/0011_being_reflection.sql"),
];

/// Applies pending migrations, each in its own transaction.
pub(crate) fn migrate(conn: &mut Connection) -> Result<()> {
    let current: i64 = conn.pragma_query_value(None, "user_version", |r| r.get(0))?;
    let supported = MIGRATIONS.len() as i64;
    if current > supported {
        return Err(Error::SchemaTooNew { found: current, supported });
    }
    for (index, sql) in MIGRATIONS.iter().enumerate().skip(current as usize) {
        let tx = conn.transaction()?;
        tx.execute_batch(sql)?;
        tx.pragma_update(None, "user_version", index as i64 + 1)?;
        tx.commit()?;
    }
    Ok(())
}

/// Records the embedding dimension on first open and refuses a different one later.
pub(crate) fn check_dimension(conn: &Connection, dim: usize) -> Result<()> {
    let stored: Option<String> =
        conn.query_row("SELECT value FROM meta WHERE key = 'embed_dim'", [], |r| r.get(0)).optional()?;
    match stored {
        None => {
            conn.execute("INSERT INTO meta(key, value) VALUES ('embed_dim', ?1)", [dim.to_string()])?;
            Ok(())
        }
        Some(value) => {
            let stored: usize = value.parse().map_err(|_| Error::Task(format!("corrupt meta.embed_dim: {value:?}")))?;
            if stored == dim { Ok(()) } else { Err(Error::DimensionMismatch { stored, requested: dim }) }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v7_upgrade_adds_unbound_being_memory_without_endorsing_or_rewriting_native_facts() {
        let mut conn = Connection::open_in_memory().unwrap();
        for sql in &MIGRATIONS[..7] {
            conn.execute_batch(sql).unwrap();
        }
        conn.pragma_update(None, "user_version", 7).unwrap();
        conn.execute_batch(
            "INSERT INTO entities(id,name,kind) VALUES(1,'Ada','person');
            INSERT INTO facts(subject_id,predicate,object_value,statement,dedup_key,recorded_first,recorded_last)
            VALUES(1,'prefers','tea','Ada prefers tea.','original',123,123);",
        )
        .unwrap();
        let before: String = conn.query_row("SELECT statement FROM facts", [], |r| r.get(0)).unwrap();
        migrate(&mut conn).unwrap();
        migrate(&mut conn).unwrap();
        let after: String = conn.query_row("SELECT statement FROM facts", [], |r| r.get(0)).unwrap();
        assert_eq!(before, after);
        for table in ["being_owner", "being_events", "being_claims", "being_observations"] {
            let count: i64 = conn.query_row(&format!("SELECT count(*) FROM {table}"), [], |r| r.get(0)).unwrap();
            assert_eq!(count, 0, "existing assertions must not receive a guessed owner or endorsement");
        }
        assert_eq!(
            conn.pragma_query_value::<i64, _>(None, "user_version", |r| r.get(0)).unwrap(),
            MIGRATIONS.len() as i64
        );
    }

    #[test]
    fn v5_transition_upgrade_preserves_legacy_identity_and_is_idempotent() {
        let mut conn = Connection::open_in_memory().unwrap();
        for sql in &MIGRATIONS[..5] {
            conn.execute_batch(sql).unwrap();
        }
        conn.pragma_update(None, "user_version", 5).unwrap();
        conn.execute_batch("INSERT INTO entities(id,name,kind) VALUES(1,'Ada','person');
            INSERT INTO sessions(id,ext_id) VALUES(1,'legacy');
            INSERT INTO turns(id,session_id,seq,speaker,role,text) VALUES(1,1,0,'Ada','user','I moved.');
            INSERT INTO facts(id,subject_id,predicate,object_value,statement,dedup_key,recorded_first,recorded_last) VALUES(1,1,'lives in','Bergen','Ada lives in Bergen.','next',123,123);
            INSERT INTO fact_updates(previous_key,next_fact_id,effective_at,effective_basis,source_turn_id,evidence_quote,observed_at,application_id) VALUES('old',1,123,'known_by_source_date',1,'I moved.',456,'legacy');").unwrap();
        migrate(&mut conn).unwrap();
        migrate(&mut conn).unwrap();
        let row: (String, i64, Option<i64>, Option<String>) = conn
            .query_row("SELECT previous_key,effective_at,subject_id,previous_object FROM fact_updates", [], |r| {
                Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?))
            })
            .unwrap();
        assert_eq!(row, ("old".into(), 123, None, None));
        assert_eq!(
            conn.pragma_query_value::<i64, _>(None, "user_version", |r| r.get(0)).unwrap(),
            MIGRATIONS.len() as i64
        );
    }

    #[test]
    fn source_lookup_upgrade_preserves_duplicate_ids_and_uses_an_index() {
        let mut conn = Connection::open_in_memory().unwrap();
        for sql in &MIGRATIONS[..6] {
            conn.execute_batch(sql).unwrap();
        }
        conn.pragma_update(None, "user_version", 6).unwrap();
        conn.execute_batch(
            "INSERT INTO sessions(id,ext_id) VALUES(1,'a'),(2,'b');
            INSERT INTO turns(session_id,seq,speaker,role,text,ext_id) VALUES
            (1,0,'A','user','first','same'),(2,0,'B','participant','second','same');",
        )
        .unwrap();
        migrate(&mut conn).unwrap();
        migrate(&mut conn).unwrap();
        let count: i64 = conn.query_row("SELECT count(*) FROM turns WHERE ext_id='same'", [], |r| r.get(0)).unwrap();
        assert_eq!(count, 2, "index must not impose new source uniqueness or lose evidence");
        let plan: String =
            conn.query_row("EXPLAIN QUERY PLAN SELECT id FROM turns WHERE ext_id=?1", ["same"], |r| r.get(3)).unwrap();
        assert!(plan.contains("turns_ext_id"), "{plan}");
    }
}
