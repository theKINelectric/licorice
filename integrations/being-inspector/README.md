> Runtime: Python’s linked SQLite must be >=3.53.2. Older versions are refused
> before opening a memory database. Qualified Python3.14.7 uses SQLite3.53.4;
> the native executable has its own separately bundled SQLite3.53.2.

# Licorice local memory inspector

Generate a private, standalone HTML snapshot of one explicitly selected native
identity. Open the report in a browser. There is no server, CDN, model call,
memory mutation or automatic upload. Python 3 and `licorice-being` are required.

From a source checkout:

```sh
python3 integrations/being-inspector/inspect.py \
  --binding /absolute/licorice/data/being/active.json \
  --query 'current commitments and changed judgments' \
  --output /absolute/private-directory/memory-inspection.html
```

From a downloaded release, use the binding created by `setup.py`:

```sh
python3 integrations/being-inspector/inspect.py \
  --binding "$HOME/.local/share/licorice/quill/binding.json" \
  --output "$HOME/.local/share/licorice/quill/inspection-01.html"
```

Both native version-1 bindings and setup's environment-mapping bindings are
accepted. They are selected as complete units. The inspector never guesses an
identity from environment variables or combines fields from different profiles.
Alternatively supply all of `--binary ABS --store ABS --agent ID`, with optional
`--relationship ID --embed-dim 256`. Do not mix these flags with `--binding`.
The output's parent directory must exist; existing files and symlinks are refused.
New reports have mode0600. Keep reports outside any public site or synced folder
unless you deliberately intend to disclose their memory content.

## What you can inspect

- Current judgments, endorsement state, reasons, sources and nullable credence.
- Attributed observations, including unadopted ones, with native source assurance.
- Commitments and their actual lifecycle status.
- Relationship-visible event history, including adoption and suspension reasons.
- Relationships, concepts, hypothesis sets and dependency links.
- A continuity-only context query, exact token use and selected/omitted units.

Click a source identifier to inspect that source. Search applies to the selected
section. Global records are included alongside the selected relationship. Records
from other relationships are removed before report serialization; harness
boundary/session metadata is omitted. This is a local view filter, not tenant
authorization or protection against someone who already has the database file.

The native snapshot's revision is identity-wide. Gaps in the displayed scoped
history can therefore be normal. A source's assurance establishes attribution,
not the truth of its content. The inspector supplies no adoption/edit controls.
It displays explicit recorded reasons, not hidden model reasoning.

## Capture and limits

The source is opened through SQLite `mode=ro`. SQLite's backup API takes a
consistent snapshot including committed WAL content; copying only the main
database file or using `immutable=1` would miss live writes. The native CLI runs
only against that disposable private copy. Original stores are not migrated.
Temporary copies are removed even when inspection fails.

This preview supports native schema11, at most128MiB of source/snapshot,
10000journal events and32MiB serialized report data. Oversize or unsupported
input is refused rather than silently clipped or upgraded. Concurrent writers
can continue after the captured snapshot. This HTML is not a backup/restore
format or a full conversation archive.

The context allowance is64..2000 ordinary-cl100k tokens, including native memory
framing. The default is2000. The native episode candidate window is128 scoped
observations; omitted/uncandidate history is not evidence of absence. The report
can display all scoped observations within the inspector bounds. Optional
WordLlama factual retrieval is deliberately not run in this first inspector;
the binding's models path does not change this continuity-only preview.

Reports encode memory as inert base64 JSON, render values with DOM text nodes,
and pin inline assets with SHA256 Content Security Policy hashes. Network
connections are disabled by policy. Opening the report in a browser still
requires the normal trust appropriate to any local executable HTML artifact.

Original Licorice sources are Apache-2.0. See the release's LICENSE and NOTICE.
Hosted service and additional Studio tooling remain planned offerings.
