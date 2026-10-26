-- Structured obligations are projections of explicit events, never inferred
-- from a relationship label or a high-confidence factual assertion.
CREATE TABLE being_relationships(
  relationship_id TEXT PRIMARY KEY,
  body TEXT NOT NULL,
  revision INTEGER NOT NULL REFERENCES being_events(seq)
);
CREATE TABLE being_commitments(
  claim_id TEXT PRIMARY KEY REFERENCES being_claims(claim_id),
  body TEXT NOT NULL,
  revision INTEGER NOT NULL REFERENCES being_events(seq)
);
