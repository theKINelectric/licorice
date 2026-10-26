-- Licorice schema v1 (SPEC §6, §7, §10.2).
-- Engine-portable subset: no WITHOUT ROWID tables, no triggers. FTS rows are
-- written by the store in the same transaction as their source rows.
-- All times are Unix milliseconds UTC.

CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);

CREATE TABLE sessions(
  id         INTEGER PRIMARY KEY,
  ext_id     TEXT NOT NULL UNIQUE,
  started_at INTEGER,
  tz         TEXT,
  closed_at  INTEGER,
  meta       TEXT
);

CREATE TABLE turns(
  id         INTEGER PRIMARY KEY,
  session_id INTEGER NOT NULL REFERENCES sessions(id),
  seq        INTEGER NOT NULL,
  speaker    TEXT NOT NULL,
  role       TEXT NOT NULL CHECK (role IN ('user','assistant','participant','system')),
  ts         INTEGER,
  text       TEXT NOT NULL,
  ext_id     TEXT,
  UNIQUE(session_id, seq)
);

CREATE TABLE chunks(
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id INTEGER NOT NULL REFERENCES sessions(id),
  first_turn INTEGER NOT NULL REFERENCES turns(id),
  last_turn  INTEGER NOT NULL REFERENCES turns(id),
  ts         INTEGER,
  text       TEXT NOT NULL,
  hash       BLOB NOT NULL UNIQUE
);

CREATE TABLE entities(
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  name        TEXT NOT NULL,
  kind        TEXT NOT NULL,
  description TEXT,
  pinned      INTEGER NOT NULL DEFAULT 0,
  merged_into INTEGER REFERENCES entities(id),
  first_seen  INTEGER,
  last_seen   INTEGER,
  mentions    INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE entity_aliases(
  alias     TEXT NOT NULL,
  entity_id INTEGER NOT NULL REFERENCES entities(id),
  PRIMARY KEY (alias, entity_id)
);
CREATE INDEX entity_aliases_entity ON entity_aliases(entity_id);

CREATE TABLE facts(
  id             INTEGER PRIMARY KEY AUTOINCREMENT,
  subject_id     INTEGER NOT NULL REFERENCES entities(id),
  predicate      TEXT NOT NULL,
  object_id      INTEGER REFERENCES entities(id),
  object_value   TEXT,
  statement      TEXT NOT NULL,
  negated        INTEGER NOT NULL DEFAULT 0,
  salience       REAL,
  occurred_from  INTEGER,
  occurred_to    INTEGER,
  time_precision TEXT,
  time_text      TEXT,
  recorded_first INTEGER NOT NULL,
  recorded_last  INTEGER NOT NULL,
  mentions       INTEGER NOT NULL DEFAULT 1,
  status         TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active','superseded','retracted')),
  superseded_by  INTEGER REFERENCES facts(id),
  valid_until    INTEGER,
  dedup_key      TEXT NOT NULL,
  CHECK (object_id IS NOT NULL OR object_value IS NOT NULL)
);
CREATE INDEX facts_subject ON facts(subject_id, status);
CREATE INDEX facts_object  ON facts(object_id, status);
CREATE INDEX facts_time    ON facts(occurred_from);
CREATE INDEX facts_dedup   ON facts(dedup_key);

CREATE TABLE fact_evidence(
  fact_id  INTEGER NOT NULL REFERENCES facts(id),
  turn_id  INTEGER NOT NULL REFERENCES turns(id),
  chunk_id INTEGER REFERENCES chunks(id),
  PRIMARY KEY (fact_id, turn_id)
);

CREATE TABLE summaries(
  session_id INTEGER PRIMARY KEY REFERENCES sessions(id),
  version    INTEGER NOT NULL,
  body       TEXT NOT NULL,
  text       TEXT NOT NULL,
  updated_at INTEGER NOT NULL
);

CREATE TABLE summary_keys(
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id INTEGER NOT NULL REFERENCES sessions(id),
  kind       TEXT NOT NULL CHECK (kind IN ('key','theme')),
  text       TEXT NOT NULL,
  live       INTEGER NOT NULL DEFAULT 1
);

-- Append-only vector logs (§7). `vid` is the rowid sqlite-vector returns; AUTOINCREMENT
-- keeps vids monotonic so the quantization watermark stays meaningful after deletes.
CREATE TABLE vec_fact    (vid INTEGER PRIMARY KEY AUTOINCREMENT, owner INTEGER NOT NULL, live INTEGER NOT NULL DEFAULT 1, v BLOB NOT NULL);
CREATE TABLE vec_entity  (vid INTEGER PRIMARY KEY AUTOINCREMENT, owner INTEGER NOT NULL, live INTEGER NOT NULL DEFAULT 1, v BLOB NOT NULL);
CREATE TABLE vec_chunk   (vid INTEGER PRIMARY KEY AUTOINCREMENT, owner INTEGER NOT NULL, live INTEGER NOT NULL DEFAULT 1, v BLOB NOT NULL);
CREATE TABLE vec_key     (vid INTEGER PRIMARY KEY AUTOINCREMENT, owner INTEGER NOT NULL, live INTEGER NOT NULL DEFAULT 1, v BLOB NOT NULL);
CREATE TABLE vec_summary (vid INTEGER PRIMARY KEY AUTOINCREMENT, owner INTEGER NOT NULL, live INTEGER NOT NULL DEFAULT 1, v BLOB NOT NULL);
CREATE INDEX vec_fact_owner    ON vec_fact(owner)    WHERE live = 1;
CREATE INDEX vec_entity_owner  ON vec_entity(owner)  WHERE live = 1;
CREATE INDEX vec_chunk_owner   ON vec_chunk(owner)   WHERE live = 1;
CREATE INDEX vec_key_owner     ON vec_key(owner)     WHERE live = 1;
CREATE INDEX vec_summary_owner ON vec_summary(owner) WHERE live = 1;

-- One row per vector table. `generation` increments on every re-quantization; reader
-- connections loaded at an older generation are replaced (sqlite-vector reads its
-- quantization parameters only when vector_init runs on a connection).
CREATE TABLE vec_state(
  tbl          TEXT PRIMARY KEY,
  watermark    INTEGER NOT NULL DEFAULT 0,
  generation   INTEGER NOT NULL DEFAULT 0,
  qtype        TEXT,
  quantized_at INTEGER,
  live_rows    INTEGER NOT NULL DEFAULT 0,
  dead_rows    INTEGER NOT NULL DEFAULT 0
);
INSERT INTO vec_state(tbl) VALUES ('vec_fact'), ('vec_entity'), ('vec_chunk'), ('vec_key'), ('vec_summary');

-- Lexical indexes. External content: the store writes and deletes FTS rows itself.
CREATE VIRTUAL TABLE fts_fact  USING fts5(statement, content='facts',  content_rowid='id', tokenize='porter unicode61 remove_diacritics 2');
CREATE VIRTUAL TABLE fts_chunk USING fts5(text,      content='chunks', content_rowid='id', tokenize='porter unicode61 remove_diacritics 2');
CREATE VIRTUAL TABLE fts_alias USING fts5(alias,     content='entity_aliases', content_rowid='rowid', tokenize='trigram');

CREATE TABLE jobs(
  id         INTEGER PRIMARY KEY,
  kind       TEXT NOT NULL,
  session_id INTEGER,
  first_turn INTEGER,
  last_turn  INTEGER,
  state      TEXT NOT NULL,
  attempts   INTEGER NOT NULL DEFAULT 0,
  error      TEXT,
  updated_at INTEGER NOT NULL
);

CREATE TABLE llm_cache(
  key        BLOB PRIMARY KEY,
  model      TEXT NOT NULL,
  response   TEXT NOT NULL,
  created_at INTEGER NOT NULL
);

-- System One judgments (§10.2): every request and answer, for caching, calibration and eval.
CREATE TABLE judgments(
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  point        TEXT NOT NULL,
  model        TEXT NOT NULL,
  request_hash BLOB NOT NULL,
  request      TEXT NOT NULL,
  answers      TEXT NOT NULL,
  latency_ms   INTEGER NOT NULL,
  created_at   INTEGER NOT NULL,
  outcome      TEXT
);
CREATE INDEX judgments_hash ON judgments(model, request_hash);
