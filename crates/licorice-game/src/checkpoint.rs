use crate::*;
use std::collections::BTreeSet;

/// One linear fork lineage. An engine creates separate checkpoints for sibling
/// worlds. No merge or knowledge transfer between siblings is implicit.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Timeline {
    pub id: String,
    pub at_entry: usize,
    pub at_tick: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Checkpoint {
    pub(crate) version: u32,
    pub(crate) world: String,
    pub(crate) characters: Vec<String>,
    pub(crate) timelines: Vec<Timeline>,
    pub(crate) tick: u64,
    pub(crate) entries: Vec<Entry>,
    pub(crate) checksum: String,
}

impl Checkpoint {
    pub fn world(&self) -> &str {
        &self.world
    }
    pub fn timeline(&self) -> &str {
        self.timelines.last().map_or("", |t| t.id.as_str())
    }
    pub fn tick(&self) -> u64 {
        self.tick
    }
    pub fn operation_count(&self) -> usize {
        self.entries.len()
    }
    pub fn characters(&self) -> &[String] {
        &self.characters
    }

    /// Call against the containing engine save before restoring this component.
    /// The library cannot verify a world snapshot the host has not supplied.
    pub fn check_world(&self, world: &str, timeline: &str, tick: u64) -> Result<()> {
        self.validate()?;
        if self.world != world || self.timeline() != timeline || self.tick != tick {
            return Err(invalid("memory checkpoint does not match the containing world save"));
        }
        Ok(())
    }

    pub(crate) fn digest(&self) -> Result<String> {
        let bytes = serde_json::to_vec(&(
            self.version,
            &self.world,
            &self.characters,
            &self.timelines,
            self.tick,
            &self.entries,
        ))?;
        Ok(blake3::hash(&bytes).to_hex().to_string())
    }
    pub(crate) fn seal(&mut self) -> Result<()> {
        self.checksum = self.digest()?;
        Ok(())
    }
    pub fn to_json(&self) -> Result<String> {
        self.validate()?;
        let text = serde_json::to_string(self)?;
        if text.len() > MAX_SAVE_BYTES {
            return Err(invalid("checkpoint exceeds 8 MiB"));
        }
        Ok(text)
    }
    pub fn from_json(text: &str) -> Result<Self> {
        if text.len() > MAX_SAVE_BYTES {
            return Err(invalid("checkpoint exceeds 8 MiB"));
        }
        let checkpoint: Self = serde_json::from_str(text)?;
        checkpoint.validate()?;
        Ok(checkpoint)
    }
    pub(crate) fn validate(&self) -> Result<()> {
        identifier(&self.world)?;
        if self.version != 1
            || self.characters.is_empty()
            || self.characters.len() > MAX_CHARACTERS
            || self.characters.windows(2).any(|p| p[0] >= p[1])
            || self.entries.len() > MAX_OPERATIONS
            || self.timelines.is_empty()
            || self.timelines.len() > 64
            || self.checksum != self.digest()?
        {
            return Err(invalid("unsupported, inconsistent or corrupt checkpoint"));
        }
        for character in &self.characters {
            identifier(character)?;
        }
        let mut names = BTreeSet::new();
        for (i, timeline) in self.timelines.iter().enumerate() {
            identifier(&timeline.id)?;
            if !names.insert(&timeline.id)
                || timeline.at_entry > self.entries.len()
                || timeline.at_tick > self.tick
                || (i == 0 && (timeline.at_entry != 0 || timeline.at_tick != 0))
                || (i > 0
                    && (timeline.at_entry < self.timelines[i - 1].at_entry
                        || timeline.at_tick < self.timelines[i - 1].at_tick))
                || (timeline.at_entry > 0 && self.entries[timeline.at_entry - 1].tick > timeline.at_tick)
            {
                return Err(invalid("invalid timeline lineage"));
            }
        }
        let mut previous_tick = 0;
        for (i, entry) in self.entries.iter().enumerate() {
            let timeline = self.timelines.iter().rev().find(|t| t.at_entry <= i).unwrap();
            if entry.timeline != timeline.id
                || entry.tick < previous_tick
                || entry.tick < timeline.at_tick
                || entry.tick > self.tick
                || self.characters.binary_search(&entry.action.character().to_string()).is_err()
            {
                return Err(invalid("journal entry crosses timeline, clock or character boundary"));
            }
            previous_tick = entry.tick;
        }
        if serde_json::to_vec(self)?.len() > MAX_SAVE_BYTES {
            return Err(invalid("checkpoint exceeds 8 MiB"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn semantically_forged_but_checksummed_journal_does_not_replace_live_memory() {
        let mut memory = GameMemory::new("world", "main", vec!["buddy".into()]).await.unwrap();
        let before = memory.checkpoint().unwrap();
        let mut forged = before.clone();
        forged.entries.push(Entry {
            timeline: "main".into(),
            tick: 0,
            action: Action::Deliberate {
                character: "buddy".into(),
                request_id: "forged".into(),
                expected_revision: 0,
                operation: being::Operation::Decide(being::Decision {
                    claim_id: "nonexistent".into(),
                    stance: being::Stance::Adopted,
                    credence: None,
                    reason: "Attempt to bypass semantic validation.".into(),
                    source_ids: vec![],
                }),
            },
        });
        forged.seal().unwrap();
        assert!(forged.validate().is_ok()); // Structure/checksum alone is insufficient.
        assert!(memory.restore_into(&forged).await.is_err());
        assert_eq!(before.to_json().unwrap(), memory.checkpoint().unwrap().to_json().unwrap());
        memory.close().await.unwrap();
    }

    #[tokio::test]
    async fn checksummed_future_cross_branch_and_duplicate_entries_are_refused() {
        let mut memory = GameMemory::new("world", "main", vec!["buddy".into()]).await.unwrap();
        memory
            .perceive(
                "buddy",
                Perception {
                    event_id: "one".into(),
                    tick: 0,
                    place: "garden".into(),
                    subject: "buddy".into(),
                    text: "Actual incident.".into(),
                    channel: Channel::Witnessed,
                    cue: None,
                    relationship_id: None,
                    fulfills: None,
                },
            )
            .await
            .unwrap();
        let before = memory.checkpoint().unwrap();
        for probe in 0..4 {
            let mut forged = before.clone();
            match probe {
                0 => forged.entries[0].tick = 1,
                1 => forged.entries[0].timeline = "sibling".into(),
                2 => forged.entries.push(forged.entries[0].clone()),
                _ => {
                    if let Action::Perceive { perception, .. } = &mut forged.entries[0].action {
                        perception.tick = 1;
                    }
                }
            }
            forged.seal().unwrap();
            assert!(memory.restore_into(&forged).await.is_err(), "probe={probe}");
            assert_eq!(before.to_json().unwrap(), memory.checkpoint().unwrap().to_json().unwrap());
        }
        memory.close().await.unwrap();
    }
}
