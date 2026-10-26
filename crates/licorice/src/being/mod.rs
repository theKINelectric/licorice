//! Deliberate, attributed memory. A factual assertion and an agent's endorsement
//! are distinct records. All writes use the existing Store transaction/receipt seam.
pub mod context;
mod import;
mod kernel;
pub mod ontology;
pub mod reflection;
mod relationships;
mod types;

pub use types::*;

use crate::store::now_ms;
use crate::{Error, Result, Store};
use rusqlite::{OptionalExtension, params};

#[derive(Clone)]
pub struct Memory {
    pub(crate) store: Store,
    pub(crate) agent: String,
}

impl Memory {
    /// Explicitly bind a memory space. Existing native facts are never adopted.
    pub async fn initialize(store: &Store, agent: &str) -> Result<Self> {
        kernel::identifier(agent)?;
        let agent_owned = agent.to_owned();
        store
            .write(move |tx| {
                tx.conn().execute(
                    "INSERT OR IGNORE INTO being_owner(singleton,agent_id,created_at) VALUES(1,?1,?2)",
                    params![agent_owned, now_ms()],
                )?;
                kernel::owner(tx.conn(), &agent_owned)
            })
            .await?;
        Ok(Self { store: store.clone(), agent: agent.into() })
    }

    /// Access an already bound space, without implicitly creating an identity.
    pub async fn attach(store: &Store, agent: &str) -> Result<Self> {
        kernel::identifier(agent)?;
        let agent_owned = agent.to_owned();
        store.read(move |conn| kernel::owner(conn, &agent_owned)).await?;
        Ok(Self { store: store.clone(), agent: agent.into() })
    }

    pub async fn apply(&self, actor: Actor, request: Request) -> Result<Receipt> {
        let agent = self.agent.clone();
        self.store.write(move |tx| kernel::apply(tx.conn(), &agent, &actor, &request, None, now_ms())).await
    }

    /// Scoped projection at one SQLite read snapshot. None means global only.
    pub async fn snapshot(&self, relationship: Option<&str>) -> Result<Snapshot> {
        if let Some(id) = relationship {
            kernel::identifier(id)?;
        }
        let agent = self.agent.clone();
        let relationship = relationship.map(str::to_owned);
        self.store
            .read(move |conn| {
                let tx = conn.unchecked_transaction()?;
                kernel::owner(&tx, &agent)?;
                let result = kernel::snapshot(&tx, &agent, relationship.as_deref())?;
                tx.commit()?;
                Ok(result)
            })
            .await
    }

    pub async fn history(&self) -> Result<Vec<Event>> {
        let agent = self.agent.clone();
        self.store
            .read(move |conn| {
                let tx = conn.unchecked_transaction()?;
                kernel::owner(&tx, &agent)?;
                let mut stmt = tx.prepare(
                    "SELECT seq,actor,request,digest,recorded_at,imported_from FROM being_events ORDER BY seq",
                )?;
                let rows = stmt.query_map([], |r| {
                    Ok((
                        r.get::<_, i64>(0)?,
                        r.get::<_, String>(1)?,
                        r.get::<_, String>(2)?,
                        r.get::<_, String>(3)?,
                        r.get::<_, i64>(4)?,
                        r.get::<_, Option<String>>(5)?,
                    ))
                })?;
                let mut events = Vec::new();
                for row in rows {
                    let (revision, actor, request, digest, recorded_at, imported_from) = row?;
                    events.push(Event {
                        revision,
                        actor: kernel::decode(&actor)?,
                        request: kernel::decode(&request)?,
                        digest,
                        recorded_at,
                        imported_from,
                    });
                }
                drop(stmt);
                tx.commit()?;
                Ok(events)
            })
            .await
    }

    pub async fn revision(&self) -> Result<i64> {
        let agent = self.agent.clone();
        self.store
            .read(move |conn| {
                kernel::owner(conn, &agent)?;
                kernel::revision(conn)
            })
            .await
    }

    pub async fn is_initialized(store: &Store) -> Result<bool> {
        store
            .read(|conn| {
                Ok(conn
                    .query_row("SELECT agent_id FROM being_owner WHERE singleton=1", [], |r| r.get::<_, String>(0))
                    .optional()?
                    .is_some())
            })
            .await
    }
}

pub(crate) fn invalid(message: &str) -> Error {
    Error::InvalidInput(message.into())
}
