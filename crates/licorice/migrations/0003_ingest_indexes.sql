-- Lookups the writer runs inside every extraction and summary transaction. Without
-- these, each is a full table scan and ingest slows quadratically as a space grows.

-- The chunk holding an evidence turn: chunks never span sessions.
CREATE INDEX chunks_session ON chunks(session_id, first_turn);
-- Facts citing a turn (the legacy-application check, and forgetting a session).
CREATE INDEX fact_evidence_turn ON fact_evidence(turn_id);
-- A session's live summary keys, retired when its summary is replaced.
CREATE INDEX summary_keys_session ON summary_keys(session_id) WHERE live = 1;
