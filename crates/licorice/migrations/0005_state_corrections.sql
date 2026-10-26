-- Replacements end a valid state; corrections retract an erroneous assertion.
ALTER TABLE fact_updates ADD COLUMN kind TEXT NOT NULL DEFAULT 'replace'
  CHECK(kind IN ('replace','correction'));
ALTER TABLE fact_updates ADD COLUMN assertion_at INTEGER;
UPDATE fact_updates SET assertion_at=(SELECT ts FROM turns WHERE id=source_turn_id);
