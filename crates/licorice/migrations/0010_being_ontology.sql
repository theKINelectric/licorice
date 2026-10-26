CREATE TABLE being_concepts(
  concept_id TEXT NOT NULL,
  version INTEGER NOT NULL,
  body TEXT NOT NULL,
  revision INTEGER NOT NULL REFERENCES being_events(seq),
  PRIMARY KEY(concept_id,version)
);
CREATE TABLE being_hypotheses(
  set_id TEXT NOT NULL,
  version INTEGER NOT NULL,
  body TEXT NOT NULL,
  revision INTEGER NOT NULL REFERENCES being_events(seq),
  PRIMARY KEY(set_id,version)
);
CREATE TABLE being_links(
  link_id TEXT PRIMARY KEY,
  body TEXT NOT NULL,
  revision INTEGER NOT NULL REFERENCES being_events(seq)
);
