//! A receipt is also the replay guard. Check again inside the writer transaction
//! because two callers may finish preparing the same request concurrently.

use rusqlite::{Connection, OptionalExtension, params};
use serde_json::Value;

use crate::store::now_ms;
use crate::{Error, Result, Store};

#[derive(Clone)]
pub(super) struct Application {
    id: String,
    kind: &'static str,
    session: String,
    input_hash: Vec<u8>,
    output_hash: Vec<u8>,
}

impl Application {
    pub(super) fn new(id: &str, kind: &'static str, session: &str, input: &Value, output: &Value) -> Self {
        Self {
            id: id.into(),
            kind,
            session: session.into(),
            input_hash: blake3::hash(input.to_string().as_bytes()).as_bytes().to_vec(),
            output_hash: blake3::hash(output.to_string().as_bytes()).as_bytes().to_vec(),
        }
    }

    pub(super) async fn lookup(&self, store: &Store) -> Result<Option<Value>> {
        let this = self.clone();
        store.read(move |conn| this.check(conn)).await
    }

    pub(super) fn check(&self, conn: &Connection) -> Result<Option<Value>> {
        let stored: Option<(Vec<u8>, Vec<u8>, String)> = conn
            .query_row("SELECT input_hash,output_hash,report FROM applied_outputs WHERE id=?1", [&self.id], |r| {
                Ok((r.get(0)?, r.get(1)?, r.get(2)?))
            })
            .optional()?;
        match stored {
            None => {
                let legacy:bool=conn.query_row(
                    "SELECT CASE WHEN ?1='extract' THEN EXISTS(SELECT 1 FROM facts f JOIN fact_evidence fe ON fe.fact_id=f.id \
                     JOIN turns t ON t.id=fe.turn_id JOIN sessions s ON s.id=t.session_id WHERE s.ext_id=?2 AND f.identity_version=1) \
                     ELSE EXISTS(SELECT 1 FROM summaries sm JOIN sessions s ON s.id=sm.session_id WHERE s.ext_id=?2) END",
                     params![self.kind,self.session],|r|r.get(0))?;
                if legacy {
                    return Err(Error::LegacyApplication { id: self.id.clone() });
                }
                Ok(None)
            }
            Some((input, output, report)) if input == self.input_hash && output == self.output_hash => {
                serde_json::from_str(&report)
                    .map(Some)
                    .map_err(|e| Error::Task(format!("invalid application receipt: {e}")))
            }
            Some(_) => Err(Error::ApplicationConflict { id: self.id.clone() }),
        }
    }

    pub(super) fn finish(&self, conn: &Connection, report: &Value) -> Result<()> {
        conn.execute(
            "INSERT INTO applied_outputs(id,kind,session_id,input_hash,output_hash,report,created_at) \
            VALUES (?1,?2,(SELECT id FROM sessions WHERE ext_id=?3),?4,?5,?6,?7)",
            params![self.id, self.kind, self.session, self.input_hash, self.output_hash, report.to_string(), now_ms()],
        )?;
        Ok(())
    }
}
