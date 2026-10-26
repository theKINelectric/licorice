//! Offline source-derived fixture, not an actual HH simulation or model trial.
use licorice_game::{hostile_hospitality as hh, *};
use serde_json::json;

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    let mut memory = GameMemory::new("hh-fixture", "park-main", vec!["buddy".into(), "sarge".into()]).await?;
    memory.advance(120)?;
    for (actor, id, kind, text) in [
        (hh::BUDDY, 101, hh::LessonKind::PowerInterrupted, "The feeder outage interrupted Buddy's water supply."),
        (hh::SARGE, 102, hh::LessonKind::PlantBitHose, "Sarge witnessed the plant grip the exposed hose."),
    ] {
        memory
            .perceive(
                actor,
                hh::Incident {
                    event_id: id,
                    tick: 120,
                    region: [0, 0],
                    subject: actor.into(),
                    text: text.into(),
                    kind,
                }
                .witnessed()?,
            )
            .await?;
    }
    let mut matches = 0;
    for bits in 0..64 {
        let policy = hh::Policy {
            caretaker: bits & 1 != 0,
            allow_bait: bits & 2 != 0,
            maintenance: bits & 4 != 0,
            irrigation_connected: bits & 8 != 0,
            hatch_open: bits & 16 != 0,
            gripped: bits & 32 != 0,
        };
        let source_reference = hh::Behavior { prepare_bait: bits & 3 == 3, secure_hatch: bits & 61 == 29 };
        matches += usize::from(hh::compare(&memory, policy, source_reference)?.matches);
    }
    let buddy = memory.recall(hh::BUDDY, None, "What interrupted my water work?", 2000, 1, 180).await?;
    let sarge = memory.recall(hh::SARGE, None, "Why secure the hose?", 2000, 1, 180).await?;
    let checkpoint = memory.checkpoint()?;
    let restored = GameMemory::restore(&Checkpoint::from_json(&checkpoint.to_json()?)?).await?;
    let exact_recovery = restored.checkpoint()?.to_json()? == checkpoint.to_json()?;
    let out = json!({"protocol":"licorice-game-fixture/1", "baseline":"source_derived_boolean_reference",
        "live_game_execution":false, "model_calls":0, "policy_cases":64, "matched":matches,
        "exact_recovery":exact_recovery, "buddy":buddy, "sarge":sarge,
        "legacy_buddy_lessons":memory.hh_lessons(hh::BUDDY)?, "legacy_sarge_lessons":memory.hh_lessons(hh::SARGE)?});
    println!("{}", serde_json::to_string_pretty(&out)?);
    memory.close().await?;
    restored.close().await?;
    if matches != 64 || !exact_recovery {
        return Err(Error::Invalid("comparison gate failed".into()));
    }
    Ok(())
}
