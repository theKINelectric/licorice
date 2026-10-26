CREATE TABLE being_reflections(
  job_id TEXT PRIMARY KEY,
  digest TEXT NOT NULL,
  actor TEXT NOT NULL,
  plan TEXT NOT NULL,
  status TEXT NOT NULL CHECK(status IN ('reserved','running','completed','failed')),
  source_groups TEXT NOT NULL,
  reserved_at INTEGER NOT NULL,
  output TEXT,
  receipt TEXT
);
CREATE INDEX being_reflections_reserved_at ON being_reflections(reserved_at);
