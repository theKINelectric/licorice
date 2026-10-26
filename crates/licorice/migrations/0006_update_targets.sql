-- Preserve the source's explicit old value independently of how either fact
-- happened to represent it. Legacy transitions retain their exact-key behavior.
ALTER TABLE fact_updates ADD COLUMN subject_id INTEGER REFERENCES entities(id);
ALTER TABLE fact_updates ADD COLUMN predicate_key TEXT;
ALTER TABLE fact_updates ADD COLUMN previous_object TEXT;
ALTER TABLE fact_updates ADD COLUMN previous_negated INTEGER CHECK(previous_negated IN (0,1));
CREATE INDEX fact_updates_target ON fact_updates(subject_id,predicate_key,previous_negated);
