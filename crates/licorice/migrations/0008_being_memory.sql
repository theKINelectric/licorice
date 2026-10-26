-- Deliberate agent stance is separate from factual assertion lifecycle.
-- Existing spaces remain unbound; no facts receive inferred endorsement.
CREATE TABLE being_owner(
  singleton INTEGER PRIMARY KEY CHECK(singleton=1),
  agent_id TEXT NOT NULL UNIQUE,
  created_at INTEGER NOT NULL
);
CREATE TABLE being_events(
  seq INTEGER PRIMARY KEY AUTOINCREMENT,
  request_id TEXT NOT NULL UNIQUE,
  actor TEXT NOT NULL,
  request TEXT NOT NULL,
  digest TEXT NOT NULL,
  recorded_at INTEGER NOT NULL,
  imported_from TEXT
);
CREATE TABLE being_observations(
  source_id TEXT PRIMARY KEY,
  event_seq INTEGER NOT NULL REFERENCES being_events(seq),
  body TEXT NOT NULL,
  assurance TEXT NOT NULL
);
CREATE TABLE being_claims(
  claim_id TEXT PRIMARY KEY,
  body TEXT NOT NULL,
  revision INTEGER NOT NULL REFERENCES being_events(seq)
);
CREATE TABLE being_claim_sources(
  claim_id TEXT NOT NULL REFERENCES being_claims(claim_id),
  source_id TEXT NOT NULL REFERENCES being_observations(source_id),
  first_revision INTEGER NOT NULL REFERENCES being_events(seq),
  PRIMARY KEY(claim_id,source_id)
);
CREATE TABLE being_session_bindings(
  session_id INTEGER PRIMARY KEY REFERENCES sessions(id),
  event_seq INTEGER NOT NULL REFERENCES being_events(seq)
);
CREATE TABLE being_imports(
  import_id TEXT PRIMARY KEY,
  digest TEXT NOT NULL,
  event_count INTEGER NOT NULL,
  receipt TEXT NOT NULL
);
CREATE TABLE being_import_rows(
  import_id TEXT NOT NULL REFERENCES being_imports(import_id),
  original_seq INTEGER NOT NULL,
  original_body TEXT NOT NULL,
  original_sha256 TEXT NOT NULL,
  original_recorded_at INTEGER NOT NULL,
  native_event_seq INTEGER NOT NULL REFERENCES being_events(seq),
  PRIMARY KEY(import_id,original_seq)
);
