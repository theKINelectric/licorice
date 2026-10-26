use super::*;
use rusqlite::{Connection, OpenFlags, OptionalExtension, params};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::path::Path;

#[derive(Serialize)]
struct LegacyRow {
    seq: i64,
    body: String,
    digest: String,
    recorded_at: i64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LegacyRequest {
    request_id: String,
    kind: String,
    payload: Value,
    expected_revision: Option<i64>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LegacyObservation {
    origin: Origin,
    subject: String,
    text: String,
    relationship_id: Option<String>,
    source_ref: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LegacyBoundary {
    session_id: String,
    turn_id: String,
    phase: BoundaryPhase,
}

fn read(path: &Path, agent: &str) -> Result<Vec<LegacyRow>> {
    let conn = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    let tx = conn.unchecked_transaction()?;
    let app: i64 = tx.pragma_query_value(None, "application_id", |r| r.get(0))?;
    let schema: i64 = tx.pragma_query_value(None, "user_version", |r| r.get(0))?;
    if app != 0x4245494e || schema != 1 {
        return Err(invalid("not a BEING prototype sidecar"));
    }
    let owners = tx
        .prepare("SELECT agent FROM owner")?
        .query_map([], |r| r.get::<_, String>(0))?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    if owners != [agent.to_owned()] {
        return Err(invalid("sidecar agent identity mismatch"));
    }
    let mut stmt = tx.prepare("SELECT seq,request_id,body,digest,recorded_at FROM events ORDER BY seq LIMIT 4097")?;
    let mut rows = Vec::new();
    let mut requests = std::collections::HashSet::new();
    for row in stmt.query_map([], |r| {
        Ok((
            r.get::<_, i64>(0)?,
            r.get::<_, String>(1)?,
            r.get::<_, String>(2)?,
            r.get::<_, String>(3)?,
            r.get::<_, i64>(4)?,
        ))
    })? {
        let (seq, request_id, body, digest, recorded_at) = row?;
        if seq != rows.len() as i64 + 1
            || rows.len() >= 4096
            || body.len() > 32768
            || format!("{:x}", Sha256::digest(body.as_bytes())) != digest
        {
            return Err(invalid("sidecar sequence, size or original SHA256 receipt is invalid"));
        }
        let request: LegacyRequest = kernel::decode(&body)?;
        if request.request_id != request_id || !requests.insert(request_id) {
            return Err(invalid("sidecar request identity mismatch"));
        }
        rows.push(LegacyRow { seq, body, digest, recorded_at });
    }
    drop(stmt);
    tx.commit()?;
    Ok(rows)
}

fn convert(body: &str) -> Result<Request> {
    let old: LegacyRequest = kernel::decode(body)?;
    let operation = match old.kind.as_str() {
        "observe" => {
            let o: LegacyObservation =
                serde_json::from_value(old.payload).map_err(|_| invalid("invalid legacy observation"))?;
            Operation::Observe(Observation {
                origin: o.origin,
                subject: o.subject,
                text: o.text,
                relationship_id: o.relationship_id,
                occurred_at: None,
                source: Source::Reported { reference: o.source_ref },
            })
        }
        "propose" => {
            Operation::Propose(serde_json::from_value(old.payload).map_err(|_| invalid("invalid legacy proposal"))?)
        }
        "decide" => {
            Operation::Decide(serde_json::from_value(old.payload).map_err(|_| invalid("invalid legacy decision"))?)
        }
        "boundary" => {
            let b: LegacyBoundary =
                serde_json::from_value(old.payload).map_err(|_| invalid("invalid legacy boundary"))?;
            kernel::identifier(&b.session_id)?;
            Operation::Boundary { turn: b.turn_id, phase: b.phase }
        }
        _ => return Err(invalid("unsupported legacy event")),
    };
    Ok(Request { request_id: old.request_id, expected_revision: old.expected_revision, operation })
}

impl Memory {
    /// Atomic import into an empty native journal. Original bytes, timestamps and
    /// SHA256 receipts are retained; imported source assurance is never promoted.
    pub async fn import_sidecar(&self, actor: Actor, import_id: &str, path: impl AsRef<Path>) -> Result<ImportReceipt> {
        kernel::identifier(import_id)?;
        kernel::identifier(&actor.principal)?;
        kernel::identifier(&actor.session)?;
        if actor.role != ActorRole::Importer {
            return Err(invalid("sidecar import requires an importer transport actor"));
        }
        let path = path.as_ref().to_path_buf();
        let agent = self.agent.clone();
        let rows = tokio::task::spawn_blocking(move || read(&path, &agent))
            .await
            .map_err(|e| Error::Task(e.to_string()))??;
        let digest = blake3::hash(kernel::encode(&rows)?.as_bytes()).to_hex().to_string();
        let import_id = import_id.to_owned();
        let agent = self.agent.clone();
        self.store
            .write(move |tx| {
                let conn = tx.conn();
                kernel::owner(conn, &agent)?;
                let prior: Option<(String, String)> = conn
                    .query_row("SELECT digest,receipt FROM being_imports WHERE import_id=?1", [&import_id], |r| {
                        Ok((r.get(0)?, r.get(1)?))
                    })
                    .optional()?;
                if let Some((old, receipt)) = prior {
                    if old != digest {
                        return Err(Error::ApplicationConflict { id: import_id });
                    }
                    let mut r: ImportReceipt = kernel::decode(&receipt)?;
                    r.replayed = true;
                    return Ok(r);
                }
                if kernel::revision(conn)? != 0 {
                    return Err(invalid("sidecar import requires an empty native BEING journal"));
                }
                // The placeholder and every imported event/projection share this transaction.
                conn.execute(
                    "INSERT INTO being_imports VALUES(?1,?2,?3,'{}')",
                    params![import_id, digest, rows.len() as i64],
                )?;
                for row in &rows {
                    let request = convert(&row.body)?;
                    let receipt = kernel::apply(conn, &agent, &actor, &request, Some(&import_id), row.recorded_at)?;
                    if receipt.replayed || receipt.revision != row.seq {
                        return Err(invalid("legacy/native event mapping is not one-to-one"));
                    }
                    conn.execute(
                        "INSERT INTO being_import_rows VALUES(?1,?2,?3,?4,?5,?6)",
                        params![import_id, row.seq, row.body, row.digest, row.recorded_at, receipt.revision],
                    )?;
                }
                let receipt = ImportReceipt {
                    import_id: import_id.clone(),
                    events: rows.len(),
                    final_revision: kernel::revision(conn)?,
                    replayed: false,
                };
                conn.execute(
                    "UPDATE being_imports SET receipt=?1 WHERE import_id=?2",
                    params![kernel::encode(&receipt)?, import_id],
                )?;
                Ok(receipt)
            })
            .await
    }
}
