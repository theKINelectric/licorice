use serde::{Deserialize, Serialize};

/// Assigned by the trusted local transport, never taken from an event payload.
/// This is attribution within the OS/file-access boundary, not remote authentication.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Actor {
    pub principal: String,
    pub role: ActorRole,
    pub session: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActorRole {
    Agent,
    Human,
    Harness,
    Importer,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub request_id: String,
    pub expected_revision: Option<i64>,
    pub operation: Operation,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "payload", rename_all = "snake_case", deny_unknown_fields)]
pub enum Operation {
    Observe(Observation),
    Propose(Proposal),
    Decide(Decision),
    DefineRelationship(RelationshipDefinition),
    RegisterCommitment(CommitmentDefinition),
    TransitionCommitment(CommitmentTransition),
    DefineConcept(ConceptDefinition),
    DefineHypotheses(HypothesisDefinition),
    LinkPositions(PositionLinkDefinition),
    RetireLink { link_id: String, reason: String },
    BindSession { session: String },
    Boundary { turn: String, phase: BoundaryPhase },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BoundaryPhase {
    Completed,
    Compacting,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Observation {
    pub origin: Origin,
    pub subject: String,
    pub text: String,
    pub relationship_id: Option<String>,
    pub occurred_at: Option<i64>,
    pub source: Source,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Origin {
    HumanReport,
    ToolResult,
    ImportedDocument,
    ModelDefaultReport,
    AgentReflection,
    DirectInteraction,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Source {
    Reported {
        reference: Option<String>,
    },
    NativeTurn {
        session: String,
        turn: String,
    },
    /// Digest of the exact observation text, attested by the local harness.
    HostReceipt {
        receipt_id: String,
        body_blake3: String,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Assurance {
    Reported,
    CanonicalTurn,
    HostReceipt,
    LegacyImport,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StoredObservation {
    pub source_id: String,
    pub observation: Observation,
    pub assurance: Assurance,
    /// Present in the bounded episode view; not an assertion of when it occurred.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recorded_revision: Option<i64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Category {
    Belief,
    Value,
    Method,
    Commitment,
    SelfDescription,
    Relationship,
    Autobiographical,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Stance {
    Considering,
    Adopted,
    Suspended,
    Declined,
    Retracted,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Proposal {
    pub claim_id: String,
    pub category: Category,
    pub subject: String,
    pub statement: String,
    pub source_ids: Vec<String>,
    pub credence: Option<f64>,
    pub importance: f64,
    pub reason: String,
    pub relationship_id: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Decision {
    pub claim_id: String,
    pub stance: Stance,
    pub credence: Option<f64>,
    pub reason: String,
    pub source_ids: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Claim {
    pub proposal: Proposal,
    pub stance: Stance,
    pub credence: Option<f64>,
    pub reason: String,
    pub revision: i64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Snapshot {
    pub agent_id: String,
    pub revision: i64,
    pub observations: Vec<StoredObservation>,
    pub claims: Vec<Claim>,
    #[serde(default)]
    pub relationships: Vec<Relationship>,
    #[serde(default)]
    pub commitments: Vec<Commitment>,
    #[serde(default)]
    pub concepts: Vec<Concept>,
    #[serde(default)]
    pub hypotheses: Vec<HypothesisSet>,
    #[serde(default)]
    pub links: Vec<PositionLink>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum PositionRef {
    Claim { claim_id: String },
    Concept { concept_id: String, version: u32 },
    Hypotheses { set_id: String, version: u32 },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConceptDefinition {
    pub concept_id: String,
    pub version: u32,
    pub label: String,
    pub definition: String,
    pub relationship_id: Option<String>,
    /// Split/merge/redefinition lineage; referenced meanings remain immutable.
    pub supersedes: Vec<PositionRef>,
    pub source_ids: Vec<String>,
    pub reason: String,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Concept {
    pub definition: ConceptDefinition,
    pub revision: i64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HypothesisMode {
    Exclusive,
    Compatible,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Alternative {
    pub claim_id: String,
    pub weight: Option<f64>,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HypothesisDefinition {
    pub set_id: String,
    pub version: u32,
    pub label: String,
    pub mode: HypothesisMode,
    pub alternatives: Vec<Alternative>,
    pub unknown_weight: Option<f64>,
    pub relationship_id: Option<String>,
    pub source_ids: Vec<String>,
    pub reason: String,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HypothesisSet {
    pub definition: HypothesisDefinition,
    pub revision: i64,
    pub calibration: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PositionRelation {
    DependsOn,
    Supports,
    ObjectsTo,
    Contradicts,
    UsesMeaning,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PositionLinkDefinition {
    pub link_id: String,
    /// Dependent/position first; dependency/evidence position second.
    pub from: PositionRef,
    pub to: PositionRef,
    pub relation: PositionRelation,
    pub relationship_id: Option<String>,
    pub source_ids: Vec<String>,
    pub reason: String,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PositionLink {
    pub definition: PositionLinkDefinition,
    pub revision: i64,
    pub from_revision: i64,
    pub to_revision: i64,
    pub retired_at: Option<i64>,
    pub retirement_reason: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RelationshipDefinition {
    pub relationship_id: String,
    pub version: u32,
    pub members: Vec<String>,
    /// The owning agent's description, not a declaration of reciprocal consent.
    pub description: String,
    pub boundaries: Vec<String>,
    pub source_ids: Vec<String>,
    pub reason: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Relationship {
    pub definition: RelationshipDefinition,
    pub revision: i64,
    pub perspective: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CommitmentDefinition {
    pub claim_id: String,
    pub counterparty: String,
    pub next_action: String,
    pub due_at: Option<i64>,
    pub acknowledgement_source: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CommitmentStatus {
    Active,
    Fulfilled,
    Released,
    Broken,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CommitmentTransition {
    pub claim_id: String,
    pub status: CommitmentStatus,
    pub reason: String,
    pub source_ids: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Commitment {
    pub definition: CommitmentDefinition,
    pub status: CommitmentStatus,
    pub reason: String,
    pub source_ids: Vec<String>,
    pub revision: i64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Receipt {
    pub revision: i64,
    pub replayed: bool,
    pub request_digest: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Event {
    pub revision: i64,
    pub actor: Actor,
    pub request: Request,
    pub digest: String,
    pub recorded_at: i64,
    pub imported_from: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ImportReceipt {
    pub import_id: String,
    pub events: usize,
    pub final_revision: i64,
    pub replayed: bool,
}
