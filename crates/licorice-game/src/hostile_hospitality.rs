//! Review adapter for HH's existing five lesson kinds. No dependency on, or
//! mutation of, the game. The host preserves its perception tests and authority.
//! This is a source-derived compatibility projection, not the live HH controller.
use crate::*;

pub const BUDDY: &str = "buddy";
pub const SARGE: &str = "sarge";

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum LessonKind {
    PowerInterrupted,
    PlantBitHose,
    WorkSucceeded,
    WorkInterrupted,
    PartnerHelped,
}
impl LessonKind {
    pub fn cue(self) -> &'static str {
        match self {
            Self::PowerInterrupted => "hh.power_interrupted",
            Self::PlantBitHose => "hh.plant_bit_hose",
            Self::WorkSucceeded => "hh.work_succeeded",
            Self::WorkInterrupted => "hh.work_interrupted",
            Self::PartnerHelped => "hh.partner_helped",
        }
    }
    fn from_cue(cue: &str) -> Option<Self> {
        [Self::PowerInterrupted, Self::PlantBitHose, Self::WorkSucceeded, Self::WorkInterrupted, Self::PartnerHelped]
            .into_iter()
            .find(|kind| kind.cue() == cue)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Incident {
    pub event_id: u64,
    pub tick: u64,
    pub region: [i32; 2],
    pub subject: String,
    pub text: String,
    pub kind: LessonKind,
}
impl Incident {
    /// Call only for a helper admitted by the game's actual witness rule.
    pub fn witnessed(&self) -> Result<Perception> {
        if self.event_id == 0 {
            return Err(invalid("HH event IDs start at one"));
        }
        Ok(Perception {
            event_id: format!("hh.event.{}", self.event_id),
            tick: self.tick,
            place: format!("region:{},{}", self.region[0], self.region[1]),
            subject: self.subject.clone(),
            text: self.text.clone(),
            channel: Channel::Witnessed,
            cue: Some(self.kind.cue().into()),
            relationship_id: None,
            fulfills: None,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LessonSummary {
    pub kind: LessonKind,
    pub first_event: u64,
    pub last_event: u64,
    pub first_tick: u64,
    pub last_tick: u64,
    pub occurrences: u64,
}

impl GameMemory {
    /// Exact presence/count projection of admitted HH witness inputs. Uses/job
    /// counters remain owned by the game and are not synthesized by this view.
    pub fn hh_lessons(&self, character: &str) -> Result<Vec<LessonSummary>> {
        let mut summaries = std::collections::BTreeMap::<LessonKind, LessonSummary>::new();
        for p in self.perceptions(character)?.filter(|p| p.channel == Channel::Witnessed) {
            let Some(kind) = p.cue.as_deref().and_then(LessonKind::from_cue) else { continue };
            let event = p
                .event_id
                .strip_prefix("hh.event.")
                .and_then(|v| v.parse::<u64>().ok())
                .filter(|v| *v > 0)
                .ok_or_else(|| invalid("HH cue requires its original numeric event ID"))?;
            let entry = summaries.entry(kind).or_insert(LessonSummary {
                kind,
                first_event: event,
                last_event: event,
                first_tick: p.tick,
                last_tick: p.tick,
                occurrences: 0,
            });
            // The legacy event stream is ordered; delayed observations remain
            // supported generically but cannot silently mimic this controller.
            if event < entry.last_event || p.tick < entry.last_tick {
                return Err(invalid("HH compatibility requires ordered witnessed incidents"));
            }
            entry.last_event = event;
            entry.last_tick = p.tick;
            entry.occurrences += 1;
        }
        Ok(summaries.into_values().collect())
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Policy {
    pub caretaker: bool,
    pub allow_bait: bool,
    pub maintenance: bool,
    pub irrigation_connected: bool,
    pub hatch_open: bool,
    pub gripped: bool,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Behavior {
    pub prepare_bait: bool,
    pub secure_hatch: bool,
}
#[derive(Clone, Debug, Serialize)]
pub struct Comparison {
    /// Passed in by the host from its current system. This function never
    /// executes an engine command or silently promotes the candidate.
    pub authoritative: Behavior,
    pub candidate: Behavior,
    pub matches: bool,
}
pub fn compare(memory: &GameMemory, policy: Policy, actual_baseline: Behavior) -> Result<Comparison> {
    let buddy = memory.hh_lessons(BUDDY)?;
    let sarge = memory.hh_lessons(SARGE)?;
    let candidate = Behavior {
        prepare_bait: policy.caretaker
            && policy.allow_bait
            && buddy.iter().any(|m| m.kind == LessonKind::PowerInterrupted),
        secure_hatch: policy.caretaker
            && policy.maintenance
            && policy.irrigation_connected
            && policy.hatch_open
            && !policy.gripped
            && sarge.iter().any(|m| m.kind == LessonKind::PlantBitHose),
    };
    Ok(Comparison { authoritative: actual_baseline, candidate, matches: actual_baseline == candidate })
}
