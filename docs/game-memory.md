# Licorice game memory — native review component

`crates/licorice-game` is an engine-independent Rust library over Licorice's native
BEING kernel. It is intended for a small continuing cast. The first adapter is
shaped around Hostile Hospitality's existing helper lessons. The game checkout
and its save schema are not modified by this package.

The host owns perception, world truth, action admission, model execution and
atomic saving. The library preserves attributed experience and explicit
character-authored judgments. It starts no daemon, model, network request or
reflection loop. Its dependencies include the existing native SQLite/vector
kernel and providers crate; absence of runtime model calls is not a claim of a
minimal dependency graph. Native macOS is the initial qualification platform;
WASM, mobile, Unity/Unreal/Godot bindings and fleet scale remain future work.

## Integrating a Rust engine

```toml
[dependencies]
licorice-game = { path = "/absolute/licorice/crates/licorice-game" }
```

```rust,no_run
use licorice_game::{Channel, GameMemory, Perception};
# async fn example() -> licorice_game::Result<()> {
let mut memory = GameMemory::new(
    "world-1618", "main", vec!["buddy".into(), "sarge".into()]
).await?;
memory.advance(120)?;
let receipt = memory.perceive("buddy", Perception {
    event_id: "supply-line-bite-17".into(), tick: 120,
    place: "garden-east".into(), subject: "buddy".into(),
    text: "Buddy witnessed a creature sever the live supply line.".into(),
    channel: Channel::Witnessed, cue: Some("supply_interrupted".into()),
    relationship_id: None, fulfills: None,
}).await?;
let recall = memory.recall("buddy", None, "How should I tend this line?", 2000, 4, 180).await?;
assert!(recall.tokens <= 2000);
// Inject recall.text only. Preserve recall.stamp in trusted host request metadata.
let save = memory.checkpoint()?.to_json()?;
// Save `save` in the same atomic envelope as the world at tick 120.
memory.close().await?;
# let _ = (receipt, save);
# Ok(()) }
```

Queue these async operations in the engine's existing bounded worker. SQLite
opening, save serialization, tokenization, retrieval and journal replay must not
run inside the fixed physics step. The component does not supply a hidden worker
or a hitch-free performance guarantee. A worker must preserve event order,
apply backpressure and report saturation rather than silently lose perceptions.
For rendering, use a cached read-only result delivered back to the engine.

Only send observations actually available to that character. A radio message
uses `Channel::Report { speaker }`; this attests the utterance, not its truth.
Geometry, radio reach, membership and appropriate visibility are engine duties.
Neither local caller metadata nor `DecisionStamp` is remote authentication.

## Deliberate judgment and commitments

The crate reexports native BEING types as `licorice_game::being`. Use
`deliberate(character, request_id, current_revision, Operation::Propose(...))`
to propose a belief/method/value. It begins considering. A separate
`Operation::Decide(...)` adopts, suspends, declines or retracts it. Reading an
incident or receiving a report never adopts a stance. Changed meanings use new
claim IDs; reasons and counterexamples remain in the journal.

Native relationships, concepts, hypotheses and dependency links retain their
existing source/revision rules. Evidence from another character cannot resolve
inside this character's store. A report cannot certify autobiography. Native
agent identity is the roster ID within its own disposable world projection;
do not bind these fictional characters to real Codex/Claude stores.

A structured promise uses an adopted relationship-scoped commitment claim and
`RegisterCommitment`. A `Fulfilled` transition additionally requires a witnessed
host perception with `fulfills: Some(the_same_claim_id)`. Ordinary observation,
another promise's outcome, reports and model assertions cannot fulfill it. The
host sets this field after actual action completion, never from model output.
Outcome receipts require an active registered promise and cannot predate its
current activation. Reopening a promise requires a new outcome; an earlier
completion cannot be recycled to close the new lifecycle.
This validates the host's declaration; the library cannot independently inspect
the physics that produced it.

Simulation deadlines remain engine-owned. Native `due_at` is wall time and is
refused here. Counterparty `Released` transitions are deliberately refused until
there is a correct game-character report mapping; Active/Fulfilled/Broken retain
native admission. No native wall-clock reflection jobs execute in this package.
An optional model/controller can propose revisions through `apply_reply`; there
is no automatic reflexive adoption or new authority to perform game actions.

## Context and callback lifetime

`recall` renders the world, timeline, character and simulation tick alongside
source-closed native continuity and bounded episodes. All this memory text shares
one 128..2,000 ordinary-cl100k token allowance. This is not the answering model's
whole request budget. Reserve scene/system/tool/output framing separately.

The latest 128 scoped observations are episode candidates; standing claims keep
their source closure even if the source is older. Saturation is reported, and
omission never establishes global absence. `snapshot` exposes that same bounded
episode view; `history` and `Checkpoint` retain the accepted journal. Context
selection currently uses native lexical overlap and significance, not an
embedding or a new model call. Compatibility decisions inspect accepted cue
records directly, so a context omission does not erase a learned lesson.
Other engines can use `perceptions(character)` for that complete typed view of
the bounded journal, preserving each witness/report channel without model calls.

Keep `DecisionStamp` in trusted request metadata. Before using an external
response, call `validate_reply(stamp, current_policy_revision)` or
`apply_reply(...)`. World, timeline, live runtime, character revision, policy
revision and simulation deadline must still match. Restoring the exact same save
creates a fresh runtime token, so an old in-flight response is invalid even if
the world later reaches the same tick again. The game must still recheck action
availability and resources when admitting a command.

## Saves, forks and recovery

The authoritative data is a versioned replay journal with its complete roster,
linear fork lineage, simulation tick and checksum. Per-character SQLite databases
are private disposable projections in a temporary directory. Save the journal
and simulation in one engine-owned atomic envelope; a separately saved newest
database must never be attached to an older park.

`Checkpoint::from_json` validates schema, checksum, limits and lineage.
`GameMemory::restore` semantically replays all records into fresh projections.
Call `checkpoint.check_world(world_id, timeline_id, world_tick)` against the
containing engine save before restoring, so a mismatched memory envelope is
refused before attachment.
`restore_into` replaces a live component only after successful replay. Loading
an old checkpoint excludes the abandoned future. `fork(new_timeline)` inherits
the exact current prefix, then records a distinct branch. Siblings do not merge.
Event/request IDs are scoped by world, originating timeline and character.
Exact replay is idempotent; the same ID with a different payload is refused.
The host resumes its source-event cursor with the world and only forwards new
events; inherited pre-fork events should not be reissued as new branch events.

The checksum detects accidental corruption, not a malicious editor with save
access. Save authentication and multiplayer authority belong to the host.
Kernel timestamps remain diagnostic wall time and are not used as game time;
recovered semantic state and the game journal do not depend on those timestamps.

Cancelling an in-flight write may interrupt the handoff after SQLite committed.
The component then refuses projection reads/writes. Its last acknowledged journal
is still available via `checkpoint`; rebuild with `restore_into` before resuming.
Never keep using a possibly divergent cache after cancelling a write future.

Initial diagnostic bounds are 16 roster characters, 4,096 accepted operations,
64 KiB per operation, 8 MiB per save and 64 lineage branches. The roster is fixed
within a checkpoint; dynamic NPC creation, journal compaction/migration and
cross-world character transfer are not implemented. Reaching a bound is an
explicit error. These are implementation defaults, not intended game-design laws.
Replay is linear in history and opens one native store per roster member; larger
cast/long-play use requires a measured storage and compaction design first.

## Hostile Hospitality adoption and comparison

The `hostile_hospitality` module contains owned DTOs for the existing five
`MemoryKind` categories and a read-only `hh_lessons` projection. The adapter does
not import the game's crate, execute game actions or reinterpret an arbitrary
shared expedition fact as every helper's witnessed experience.

The inspected game has two different layers: the retained ecological helper
controller (`CrewMindSnapshot`) and the current expedition's shared bounded
facts/dialogue. Initial compatibility covers the former's lesson presence,
occurrences, first/last event and tick, plus `prepare_bait` and `secure_hatch`
conditions. It does not replace the expedition crew, whole caretaker planner,
job scheduler, use counters, policy/proposal budgets or saga trust system.

Suggested staged adoption, for the game team to decide:

1. **Legacy:** retain existing saves, memory and decisions as authoritative.
   Export the actual witness/lesson admissions and current baseline decisions.
   Store a new optional component envelope without changing old save meaning.
2. **Shadow:** feed both systems the same admitted incidents. Pass actual legacy
   decisions into `hostile_hospitality::compare`. Its `authoritative` field always
   contains the supplied legacy result, including on mismatch. Record divergence,
   tick, policy, source IDs, query packet size and latency. Start new candidate
   memory at an explicit point or import only backed-up historical evidence;
   aggregate legacy counters cannot reconstruct lost individual episodes.
3. **Compatibility candidate:** behind an explicit game feature/mode switch,
   substitute only the learned-lesson read seam after parity and recovery pass.
   Keep command admission, job execution and existing counters in the game.
   Retain the legacy state for rollback, with a declared save-format policy.
4. **Richer behaviour:** compare perspective-aware recall and explicit revised
   methods under matched incidents, reader/model, packet cap and action menu.
   Record model proposals and admission ticks for deterministic replay; do not
   re-call a model during replay. Old/new systems must not both execute actions.
5. **Player observation:** determine whether players notice useful continuity,
   change their decisions and find repair/reconciliation understandable. Tests
   cannot establish fun or sustained individuality.

The included example is an offline source-derived fixture, clearly labelled:

```sh
cargo test --offline -p licorice-game -- --test-threads=1
cargo run --offline -p licorice-game --example hh_shadow
```

It compares Boolean conditions and demonstrates distinct retained episodes,
bounded packets and exact journal recovery. It is not a live game A/B, a physical
simulation, a model-quality evaluation or evidence of better player experience.
The receipt records hashes of the inspected HH source. Recheck that source before
an integration; current development may change the seam.

The game team owns the swap, actual save-envelope integration, real simulation
replays, resource/latency measurement and playtesting. C ABI/engine plugins should
follow that first proved integration rather than freeze an untested interface.
