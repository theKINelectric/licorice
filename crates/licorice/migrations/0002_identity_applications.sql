-- Existing facts may already contain lossy merges. Preserve them as v1 rather
-- than inventing event boundaries; only newly applied facts use v2 identity.
ALTER TABLE facts ADD COLUMN identity_version INTEGER NOT NULL DEFAULT 1;
ALTER TABLE facts ADD COLUMN time_kind TEXT NOT NULL DEFAULT 'unknown'
  CHECK(time_kind IN ('instant','interval','ongoing','unknown'));
CREATE INDEX facts_identity_v2 ON facts(dedup_key, status) WHERE identity_version = 2;

-- Completion and domain changes are committed by the same writer transaction.
CREATE TABLE applied_outputs(
  id          TEXT PRIMARY KEY,
  kind        TEXT NOT NULL CHECK(kind IN ('extract','summarize')),
  session_id  INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  input_hash  BLOB NOT NULL CHECK(length(input_hash) = 32),
  output_hash BLOB NOT NULL CHECK(length(output_hash) = 32),
  report      TEXT NOT NULL,
  created_at  INTEGER NOT NULL
);
