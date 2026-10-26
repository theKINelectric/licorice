-- Source provenance checks need indexed caller-ID lookups. IDs remain nonunique
-- across sessions: ambiguous source identities must be detected, never merged.
CREATE INDEX turns_ext_id ON turns(ext_id);
