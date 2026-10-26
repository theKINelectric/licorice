use crate::*;
use being::{Actor, ActorRole, CommitmentStatus, Memory, Observation, Operation, Origin, Request, Source};
use licorice::{Store, StoreConfig};
use licorice_providers::tokens::{Cl100kCounter, TokenCounter};
use std::collections::BTreeMap;

struct Character {
    memory: Memory,
    store: Store,
}

pub struct GameMemory {
    state: Checkpoint,
    characters: BTreeMap<String, Character>,
    receipts: BTreeMap<String, (Action, Receipt)>,
    runtime: String,
    needs_rebuild: bool,
    // Last field: keep cache files alive until all their store handles drop.
    cache: tempfile::TempDir,
}

impl GameMemory {
    pub async fn new(world: &str, timeline: &str, mut characters: Vec<String>) -> Result<Self> {
        characters.sort();
        let mut state = Checkpoint {
            version: 1,
            world: world.into(),
            characters,
            timelines: vec![Timeline { id: timeline.into(), at_entry: 0, at_tick: 0 }],
            tick: 0,
            entries: vec![],
            checksum: String::new(),
        };
        state.seal()?;
        Self::restore(&state).await
    }

    /// Rebuild a fresh, private projection. No old database path is reopened.
    pub async fn restore(checkpoint: &Checkpoint) -> Result<Self> {
        checkpoint.validate()?;
        let cache = tempfile::Builder::new().prefix("licorice-game-").tempdir()?;
        let runtime = blake3::hash(cache.path().to_string_lossy().as_bytes()).to_hex().to_string();
        let mut memory = Self {
            state: checkpoint.clone(),
            characters: BTreeMap::new(),
            receipts: BTreeMap::new(),
            runtime,
            needs_rebuild: false,
            cache,
        };
        memory.state.entries.clear();
        memory.state.tick = 0;
        for id in &checkpoint.characters {
            let mut config = StoreConfig::new(4);
            config.readers = 1;
            let file = format!("{}.sqlite", blake3::hash(id.as_bytes()).to_hex());
            let store = Store::open(memory.cache.path().join(file), config).await?;
            let native = Memory::initialize(&store, id).await?;
            memory.characters.insert(id.clone(), Character { memory: native, store });
        }
        for entry in &checkpoint.entries {
            memory.advance(entry.tick)?;
            let receipt = memory.execute(entry.clone()).await?;
            if receipt.replayed {
                return Err(invalid("checkpoint contains duplicate journal operations"));
            }
        }
        memory.advance(checkpoint.tick)?;
        memory.state.seal()?;
        Ok(memory)
    }

    /// Only replace the live world after every restored entry passes admission.
    pub async fn restore_into(&mut self, checkpoint: &Checkpoint) -> Result<()> {
        let replacement = Self::restore(checkpoint).await?;
        *self = replacement;
        Ok(())
    }
    pub async fn fork(&self, timeline: &str) -> Result<Self> {
        identifier(timeline)?;
        let mut checkpoint = self.checkpoint()?;
        checkpoint.timelines.push(Timeline {
            id: timeline.into(),
            at_entry: checkpoint.entries.len(),
            at_tick: checkpoint.tick,
        });
        checkpoint.seal()?;
        Self::restore(&checkpoint).await
    }
    pub fn world(&self) -> &str {
        &self.state.world
    }
    pub fn timeline(&self) -> &str {
        self.state.timeline()
    }
    pub fn tick(&self) -> u64 {
        self.state.tick
    }
    pub fn advance(&mut self, tick: u64) -> Result<()> {
        if tick < self.state.tick {
            return Err(invalid("clock cannot rewind; restore or fork a checkpoint"));
        }
        self.state.tick = tick;
        Ok(())
    }
    pub fn checkpoint(&self) -> Result<Checkpoint> {
        let mut checkpoint = self.state.clone();
        checkpoint.seal()?;
        checkpoint.validate()?;
        Ok(checkpoint)
    }
    fn character(&self, id: &str) -> Result<&Character> {
        self.characters.get(id).ok_or_else(|| invalid("character is outside this world's roster"))
    }
    pub(crate) fn check_roster(&self, id: &str) -> Result<()> {
        self.character(id).map(|_| ())
    }
    pub(crate) fn ensure_ready(&self) -> Result<()> {
        if self.needs_rebuild {
            return Err(invalid("cancelled write requires rebuilding from checkpoint"));
        }
        Ok(())
    }
    pub async fn revision(&self, character: &str) -> Result<i64> {
        self.ensure_ready()?;
        Ok(self.character(character)?.memory.revision().await?)
    }
    pub async fn snapshot(&self, character: &str, relationship: Option<&str>) -> Result<being::Snapshot> {
        self.ensure_ready()?;
        Ok(self.character(character)?.memory.snapshot_with_episodes(relationship).await?.0)
    }
    pub async fn history(&self, character: &str) -> Result<Vec<being::Event>> {
        self.ensure_ready()?;
        Ok(self.character(character)?.memory.history().await?)
    }
    /// Complete accepted perceptions within this bounded saved journal. Unlike
    /// model recall, this typed view has no token-based omissions. The channel
    /// remains available so reports need not become learned witness cues.
    pub fn perceptions<'a>(&'a self, character: &'a str) -> Result<impl Iterator<Item = &'a Perception> + 'a> {
        self.ensure_ready()?;
        self.check_roster(character)?;
        Ok(self.perceived(character))
    }

    fn native_id(&self, timeline: &str, action: &Action) -> Result<String> {
        let (kind, character, id) = match action {
            Action::Perceive { character, perception } => ("event", character, &perception.event_id),
            Action::Deliberate { character, request_id, .. } => ("choice", character, request_id),
        };
        identifier(id)?;
        let bytes = serde_json::to_vec(&(self.world(), timeline, character, id))?;
        Ok(format!("{kind}-{}", blake3::hash(&bytes).to_hex()))
    }

    /// The host has already checked sight/radio access; the library does not
    /// reconstruct geometry. Call once per actual observer, never broadcast truth.
    pub async fn perceive(&mut self, character: &str, perception: Perception) -> Result<Receipt> {
        self.execute(Entry {
            timeline: self.timeline().into(),
            tick: self.tick(),
            action: Action::Perceive { character: character.into(), perception },
        })
        .await
    }
    /// This is an explicitly agent-authored controller decision, not an automatic
    /// side effect of observation or a general untrusted command endpoint.
    pub async fn deliberate(
        &mut self,
        character: &str,
        request_id: &str,
        expected_revision: i64,
        operation: Operation,
    ) -> Result<Receipt> {
        self.execute(Entry {
            timeline: self.timeline().into(),
            tick: self.tick(),
            action: Action::Deliberate {
                character: character.into(),
                request_id: request_id.into(),
                expected_revision,
                operation,
            },
        })
        .await
    }

    fn validate_perception(&self, p: &Perception, tick: u64) -> Result<()> {
        identifier(&p.event_id)?;
        identifier(&p.subject)?;
        if p.tick > tick
            || p.place.trim().is_empty()
            || p.place.len() > 256
            || p.text.trim().is_empty()
            || p.text.len() > 8192
        {
            return Err(invalid("perception has future time, empty or oversized content"));
        }
        for id in [&p.cue, &p.relationship_id, &p.fulfills].into_iter().flatten() {
            identifier(id)?;
        }
        if let Channel::Report { speaker } = &p.channel {
            identifier(speaker)?;
            if p.fulfills.is_some() {
                return Err(invalid("a report cannot certify a commitment outcome"));
            }
        }
        Ok(())
    }

    fn perception_for(&self, character: &str, source: &str) -> Option<&Perception> {
        self.state.entries.iter().find_map(|entry| match &entry.action {
            Action::Perceive { character: c, perception }
                if c == character && self.native_id(&entry.timeline, &entry.action).ok().as_deref() == Some(source) =>
            {
                Some(perception)
            }
            _ => None,
        })
    }
    fn validate_operation(&self, character: &str, op: &Operation) -> Result<()> {
        let finite = |n: f64| n.is_finite() && (0.0..=1.0).contains(&n);
        match op {
            Operation::Observe(_) | Operation::BindSession { .. } | Operation::Boundary { .. } => {
                return Err(invalid("game observations require the trusted perception boundary"));
            }
            Operation::Propose(p) => {
                if !finite(p.importance) || p.credence.is_some_and(|n| !finite(n)) {
                    return Err(invalid("nonfinite or invalid weight"));
                }
                if p.category == being::Category::Autobiographical
                    && p.source_ids
                        .iter()
                        .any(|id| self.perception_for(character, id).is_none_or(|p| p.channel != Channel::Witnessed))
                {
                    return Err(invalid("a report is not this character's autobiography"));
                }
            }
            Operation::Decide(d) if d.credence.is_some_and(|n| !finite(n)) => {
                return Err(invalid("nonfinite or invalid weight"));
            }
            Operation::DefineHypotheses(h) => {
                if h.alternatives.iter().filter_map(|a| a.weight).chain(h.unknown_weight).any(|n| !finite(n)) {
                    return Err(invalid("nonfinite or invalid hypothesis weight"));
                }
            }
            Operation::RegisterCommitment(c) if c.due_at.is_some() => {
                return Err(invalid("keep simulation deadlines in the engine; native due_at is wall time"));
            }
            Operation::TransitionCommitment(c) if c.status == CommitmentStatus::Released => {
                return Err(invalid("counterparty release is not yet mapped by the game adapter"));
            }
            _ => {}
        }
        Ok(())
    }

    fn active_commitment(&self, character: &str, claim: &str) -> Option<(usize, u64)> {
        self.state
            .entries
            .iter()
            .enumerate()
            .rev()
            .find_map(|(i, entry)| {
                let Action::Deliberate { character: author, operation, .. } = &entry.action else { return None };
                if author != character {
                    return None;
                }
                match operation {
                    Operation::RegisterCommitment(c) if c.claim_id == claim => Some(Some((i, entry.tick))),
                    Operation::TransitionCommitment(c) if c.claim_id == claim => {
                        Some((c.status == CommitmentStatus::Active).then_some((i, entry.tick)))
                    }
                    _ => None,
                }
            })
            .flatten()
    }

    fn validate_outcome(&self, action: &Action) -> Result<()> {
        match action {
            Action::Perceive { character, perception } => {
                if let Some(claim) = &perception.fulfills {
                    let (_, since) = self
                        .active_commitment(character, claim)
                        .ok_or_else(|| invalid("outcome receipt needs an active registered commitment"))?;
                    if perception.tick < since {
                        return Err(invalid("outcome predates the active promise lifecycle"));
                    }
                }
            }
            Action::Deliberate { character, operation: Operation::TransitionCommitment(c), .. }
                if c.status == CommitmentStatus::Fulfilled =>
            {
                let (activation, since) = self
                    .active_commitment(character, &c.claim_id)
                    .ok_or_else(|| invalid("fulfillment needs an active commitment"))?;
                let supported = self.state.entries.iter().enumerate().any(|(i, entry)| {
                    let Action::Perceive { character: observer, perception: p } = &entry.action else { return false };
                    i > activation
                        && observer == character
                        && p.tick >= since
                        && p.channel == Channel::Witnessed
                        && p.fulfills.as_deref() == Some(&c.claim_id)
                        && self.native_id(&entry.timeline, &entry.action).is_ok_and(|id| c.source_ids.contains(&id))
                });
                if !supported {
                    return Err(invalid("fulfillment needs a matching outcome from the current lifecycle"));
                }
            }
            _ => {}
        }
        Ok(())
    }

    async fn execute(&mut self, entry: Entry) -> Result<Receipt> {
        self.ensure_ready()?;
        let character = entry.action.character();
        self.character(character)?;
        let (role, operation, expected_revision) = match &entry.action {
            Action::Perceive { perception, .. } => {
                self.validate_perception(perception, entry.tick)?;
                let text = serde_json::to_string(perception)?;
                let id = self.native_id(&entry.timeline, &entry.action)?;
                (
                    ActorRole::Harness,
                    Operation::Observe(Observation {
                        origin: Origin::ToolResult,
                        subject: perception.subject.clone(),
                        relationship_id: perception.relationship_id.clone(),
                        occurred_at: None,
                        source: Source::HostReceipt {
                            receipt_id: id,
                            body_blake3: blake3::hash(text.as_bytes()).to_hex().to_string(),
                        },
                        text,
                    }),
                    None,
                )
            }
            Action::Deliberate { operation, expected_revision, .. } => {
                self.validate_operation(character, operation)?;
                (ActorRole::Agent, operation.clone(), Some(*expected_revision))
            }
        };
        let id = self.native_id(&entry.timeline, &entry.action)?;
        if let Some((old, receipt)) = self.receipts.get(&id) {
            if old != &entry.action {
                return Err(invalid("event/request ID conflicts with its original payload"));
            }
            return Ok(Receipt { replayed: true, ..receipt.clone() });
        }
        // Exact historical replay remains idempotent after lifecycle changes;
        // only a new operation must satisfy today's active-outcome precondition.
        self.validate_outcome(&entry.action)?;
        if self.state.entries.len() >= MAX_OPERATIONS || serde_json::to_vec(&entry)?.len() > 65536 {
            return Err(invalid("game journal or operation limit reached"));
        }
        let mut prospective = self.state.clone();
        prospective.entries.push(entry.clone());
        // Reserve exact serialized size, including a full 64-byte checksum.
        prospective.checksum = "0".repeat(64);
        if serde_json::to_vec(&prospective)?.len() > MAX_SAVE_BYTES {
            return Err(invalid("game journal exceeds 8 MiB"));
        }
        let request = Request { request_id: id.clone(), expected_revision, operation };
        let actor = Actor {
            principal: if role == ActorRole::Harness { "game-host".into() } else { character.into() },
            role,
            session: entry.timeline.clone(),
        };
        // Cancellation can drop this future after SQLite committed but before
        // the journal append. Refuse further cache use until a clean rebuild.
        self.needs_rebuild = true;
        let result = self.characters[character].memory.apply(actor, request).await;
        self.needs_rebuild = false;
        let native = result?;
        let receipt = Receipt {
            source_id: matches!(entry.action, Action::Perceive { .. }).then(|| id.clone()),
            revision: native.revision,
            replayed: native.replayed,
        };
        self.receipts.insert(id, (entry.action.clone(), receipt.clone()));
        self.state.entries.push(entry);
        Ok(receipt)
    }

    pub async fn recall(
        &self,
        character: &str,
        relationship: Option<&str>,
        query: &str,
        budget_tokens: usize,
        policy_revision: u64,
        lifetime_ticks: u64,
    ) -> Result<Recall> {
        self.ensure_ready()?;
        if !(128..=MAX_MEMORY_TOKENS).contains(&budget_tokens) || lifetime_ticks == 0 {
            return Err(invalid("recall requires 128..2000 tokens and a positive simulation lifetime"));
        }
        let (snapshot, episodes) = self.character(character)?.memory.snapshot_with_episodes(relationship).await?;
        let prefix = format!(
            "Game memory: world {}, timeline {}, character {}, simulation tick {}. Reports retain their speakers; beliefs do not certify world truth.\n",
            self.world(),
            self.timeline(),
            character,
            self.tick()
        );
        let counter = Cl100kCounter;
        let prefix_tokens = counter.count(&prefix).map_err(|e| invalid(&e.to_string()))?;
        let mut available = budget_tokens.saturating_sub(prefix_tokens + 8);
        let (packet, text, tokens) = loop {
            let packet = being::context::continuity(&snapshot, query, &counter, available)?;
            let text = format!("{prefix}{}", packet.text);
            let tokens = counter.count(&text).map_err(|e| invalid(&e.to_string()))?;
            if tokens <= budget_tokens {
                break (packet, text, tokens);
            }
            available = available
                .checked_sub(tokens - budget_tokens + 8)
                .ok_or_else(|| invalid("memory allowance cannot fit attribution"))?;
        };
        let expires_at_tick =
            self.tick().checked_add(lifetime_ticks).ok_or_else(|| invalid("callback lifetime overflow"))?;
        Ok(Recall {
            text,
            tokens,
            budget_tokens,
            tokenizer: counter.id().into(),
            source_ids: packet.source_ids,
            episode_candidates_saturated: episodes.candidate_limit_reached,
            stamp: DecisionStamp {
                world: self.world().into(),
                timeline: self.timeline().into(),
                runtime: self.runtime.clone(),
                character: character.into(),
                revision: snapshot.revision,
                policy_revision,
                based_on_tick: self.tick(),
                expires_at_tick,
            },
        })
    }

    pub async fn validate_reply(&self, stamp: &DecisionStamp, policy_revision: u64) -> Result<()> {
        if stamp.world != self.world()
            || stamp.timeline != self.timeline()
            || stamp.runtime != self.runtime
            || stamp.policy_revision != policy_revision
            || stamp.based_on_tick > self.tick()
            || stamp.expires_at_tick < self.tick()
            || stamp.expires_at_tick <= stamp.based_on_tick
            || stamp.revision != self.revision(&stamp.character).await?
        {
            return Err(invalid("callback belongs to a stale world, timeline, memory, policy or tick"));
        }
        Ok(())
    }
    pub async fn apply_reply(
        &mut self,
        stamp: &DecisionStamp,
        policy_revision: u64,
        request_id: &str,
        operation: Operation,
    ) -> Result<Receipt> {
        self.validate_reply(stamp, policy_revision).await?;
        self.deliberate(&stamp.character, request_id, stamp.revision, operation).await
    }

    pub(crate) fn perceived<'a>(&'a self, character: &'a str) -> impl Iterator<Item = &'a Perception> + 'a {
        self.state.entries.iter().filter_map(move |entry| match &entry.action {
            Action::Perceive { character: c, perception } if c == character => Some(perception),
            _ => None,
        })
    }
    pub async fn close(mut self) -> Result<()> {
        let mut error = None;
        for (_, Character { memory, store }) in std::mem::take(&mut self.characters) {
            drop(memory);
            if let Err(e) = store.close().await {
                error = Some(e);
            }
        }
        match error {
            Some(e) => Err(e.into()),
            None => Ok(()),
        }
    }
}
