-- Transaction timestamps are intentionally unknown for pre-migration facts.
ALTER TABLE facts ADD COLUMN transaction_created_at INTEGER;
ALTER TABLE facts ADD COLUMN transaction_invalidated_at INTEGER;
CREATE TABLE fact_updates(
  id INTEGER PRIMARY KEY,
  previous_key TEXT NOT NULL,
  next_fact_id INTEGER NOT NULL REFERENCES facts(id),
  effective_at INTEGER NOT NULL,
  effective_basis TEXT NOT NULL,
  source_turn_id INTEGER NOT NULL REFERENCES turns(id),
  evidence_quote TEXT NOT NULL,
  observed_at INTEGER NOT NULL,
  application_id TEXT NOT NULL,
  UNIQUE(application_id,previous_key,next_fact_id)
);
CREATE INDEX fact_updates_previous ON fact_updates(previous_key,effective_at);
