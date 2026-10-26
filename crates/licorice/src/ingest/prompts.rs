//! Prompts and output schemas for ingestion (SPEC §8.2, §8.5). Written for Licorice;
//! LiCoMemory's prompts carry no licence and are not reused (SPEC §14).

use serde_json::{Value, json};

pub(crate) const EXTRACT_SYSTEM: &str = "You extract long-term memory from conversation transcripts. \
You return only JSON that matches the given schema, with no commentary.";

pub(crate) const EXTRACT_INSTRUCTIONS: &str = r#"Extract what someone would need to remember from the turns under "Turns to extract from".

entities: the people, organizations, places, events, concepts and objects those turns mention that matter for later questions about the speakers.
- Use each speaker's name exactly as listed under "Speakers"; "I", "me" and "my" mean the speaker of that turn.
- kind is one of: person, organization, location, event, concept, object.
- aliases: other names or spellings used for the same entity in these turns.

facts: specific, durable statements about the speakers or the entities they mention — things that happened, plans, preferences, opinions, relationships, possessions, identities, habits and feelings.
- One fact per statement. Write "statement" as a standalone sentence that names its subject (no pronouns) and keeps concrete details: names, numbers, places, titles.
- subject: the entity the fact is about, as named in entities. object: an entity name, or a literal value such as "45 minutes"; set object_is_entity accordingly.
- negated: true if the fact says something is not or is no longer the case.
- salience: 1 (trivial) to 10 (central to who the person is).
- time.expression: the time words from the turn ("yesterday", "last Saturday", "two years ago", "next month", "on 7 May"), or null when the turn gives none. Copy the words; do not compute dates.
- time.kind: instant, interval, ongoing or unknown.
- evidence_turns: the ids, in square brackets in the transcript, of the turns that support the fact.

Rules:
- Use only what the turns under "Turns to extract from" state or clearly imply. Earlier turns are context only.
- A photo a speaker shares, written as [shares a photo: ...], is part of that turn.
- Skip greetings, thanks and small talk that says nothing about anyone."#;

pub(crate) fn extract_schema() -> Value {
    json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["entities", "facts"],
        "properties": {
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["name", "kind", "aliases"],
                    "properties": {
                        "name": {"type": "string"},
                        "kind": {"type": "string", "enum": ["person", "organization", "location", "event", "concept", "object"]},
                        "aliases": {"type": "array", "items": {"type": "string"}},
                        "description": {"type": ["string", "null"]}
                    }
                }
            },
            "facts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["subject", "predicate", "object", "object_is_entity", "statement", "negated", "salience", "time", "evidence_turns"],
                    "properties": {
                        "subject": {"type": "string"},
                        "predicate": {"type": "string"},
                        "object": {"type": ["string", "null"]},
                        "object_is_entity": {"type": "boolean"},
                        "statement": {"type": "string"},
                        "negated": {"type": "boolean"},
                        "salience": {"type": "integer", "minimum": 1, "maximum": 10},
                        "time": {
                            "type": "object",
                            "additionalProperties": false,
                            "required": ["expression", "kind"],
                            "properties": {
                                "expression": {"type": ["string", "null"]},
                                "kind": {"type": "string", "enum": ["instant", "interval", "ongoing", "unknown"]}
                            }
                        },
                        "evidence_turns": {"type": "array", "items": {"type": "string"}}
                    }
                }
            }
        }
    })
}

pub(crate) const SUMMARY_SYSTEM: &str = "You summarize conversation sessions for long-term memory. \
You return only JSON that matches the given schema, with no commentary.";

pub(crate) const SUMMARY_INSTRUCTIONS: &str = r#"Summarize the session above for someone who will later answer questions about these people.

keys: up to 5 short key phrases (names, places, events, topics) that best identify this session. Include the session date as one key, written like "8 May 2023".
themes: up to 5 distinct topics of the session. For each, a short title and a summary of at most 60 words: what each speaker said, did, plans or prefers, with concrete details (names, numbers, places, dates). Write in the third person, using the speakers' names. Use only what the transcript says."#;

pub(crate) fn summary_schema() -> Value {
    json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["keys", "themes"],
        "properties": {
            "keys": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
            "themes": {
                "type": "array",
                "maxItems": 5,
                "items": {
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["title", "summary"],
                    "properties": {"title": {"type": "string"}, "summary": {"type": "string"}}
                }
            }
        }
    })
}

pub(crate) const TEMPORAL_INSTRUCTIONS: &str = r#"
Temporal classification is about the underlying fact, not the act of saying it:
- ongoing: a continuing state, including where someone lives, employment,
  family relationships, preferences and habits. 'Mira lives in Oslo' is ongoing.
- instant: a one-time action/event, including a hike, visit, birth or move.
- interval: a fact explicitly bounded to a past or future period.
- unknown: the source does not establish which kind applies.
Use stable relation names across tenses. Residence ALWAYS uses predicate 'lives in',
including 'lived in'; dates encode tense. Represent named places as entities.
Residence facts MUST put the named place in object, never null. A residence state
is ongoing (or interval for an explicitly bounded past period), never instant.
An explicit move/replacement must produce the new ongoing state with its update;
optionally also retain the separate move event. Example: 'Mira moved from Oslo to
Bergen on 1 February 2023 and lives in Bergen now' produces Mira/lives in/Bergen,
time.kind ongoing, expression '1 February 2023', update.previous_object 'Oslo',
update.evidence_quote 'moved from Oslo to Bergen', and update.effective_expression
'1 February 2023'. A historical mention of Oslo
without a new move does not replace Bergen.
The replacement already ends the old residence. Do not add a redundant negated
old-home assertion from a positive move alone; retain a separate negative fact
only when the source independently states that denial.
Each turn has its own UTC date. Copy relative time words verbatim, and set
 time.anchor_turn to the supporting turn whose date anchors those words.
update.kind is 'replace' for a genuine change of state. Use 'correction' only
when the source explicitly says the earlier assertion was wrong; corrections
retract that erroneous assertion rather than claiming it used to be true.
A correction's evidence_turns and time.anchor_turn MUST cite the CURRENT turn
that corrects the claim, never the earlier mistaken claim shown only as context.
One turn may both correct a false old claim and report a genuine move. Preserve
both operations in separate assertions when needed: an explicit 'I never lived
in X; that claim was wrong' produces a negated ongoing residence at X, with
update.kind correction and previous_object X. A genuine move from Y to Z produces
the positive ongoing residence at Z, with update.kind replace and previous_object Y.
Each operation needs its own literal supporting quote; do not turn a never-true
claim into a formerly true state, or date the residence as an instant move event.
update must be null unless the source explicitly replaces a previous ongoing
state or explicitly corrects an earlier assertion. A new value alone is not a
replacement: people can have multiple jobs, homes, interests, and relationships.
For an explicit replacement, use time.kind=ongoing, copy previous_object exactly,
set update.effective_expression to the exact source time words for when the new
state took effect (for example '1 February 2023'), or null if the source gives no
effective date. This is separate from the date the speaker reports the change.
Include the date in time.expression too when it applies to the new state.
Give evidence_quote as an exact substring of a cited turn that establishes
the change (for example 'moved from Oslo to Bergen'). Never invent a previous
value or infer a replacement solely from a more recent message. For a correction,
copy the explicit corrected value and use the date on which the correction takes
effect, if given. Preserve uncertainty, hypotheticals and speaker attribution.
"#;

pub(crate) fn temporal_schema() -> Value {
    let mut schema = extract_schema();
    let fact = &mut schema["properties"]["facts"]["items"];
    fact["required"].as_array_mut().unwrap().push(json!("update"));
    fact["properties"]["time"]["required"].as_array_mut().unwrap().push(json!("anchor_turn"));
    fact["properties"]["time"]["properties"]["anchor_turn"] = json!({"type":"string"});
    fact["properties"]["update"] = json!({"anyOf":[{"type":"null"},{"type":"object",
       "additionalProperties":false,"required":["kind","previous_object","evidence_quote","effective_expression"],"properties":{
        "kind":{"type":"string","enum":["replace","correction"]},
        "previous_object":{"type":"string"},"evidence_quote":{"type":"string"},"effective_expression":{"type":["string","null"]}}}]});
    schema
}
