> Runtime: Python’s linked SQLite must be >=3.53.2. Older versions are refused
> before opening a memory database. Qualified Python3.14.7 uses SQLite3.53.4;
> the native executable has its own separately bundled SQLite3.53.2.

# Portable identity backup and restore

This local tool preserves one native identity's complete SQLite state, including
all its relationships, attributed events, stance history, commitments, ontology,
stored source/factual data and reflection reservations. It does not merge agents,
filter a relationship for sharing, capture missing conversations or run a model.
The original database is never handed to a native executable or migrated.

**Backups contain private memory. They are unencrypted.** The file is created
with mode0600 and restored state with directory0700/files0600. Use encrypted
storage/transport appropriate to the machine. SHA256 and ZIP checks detect
accidental damage; they do not authenticate an author or defeat an attacker who
can rewrite both data and manifest. Never put an archive or restored state in the
public download directory. This is not a remotely authorized import endpoint.

Current source qualification is Apple Silicon macOS. It accepts schema11 only;
other schemas are explicitly refused rather than silently upgraded/downgraded.
The selected native binary must reproduce the saved selected snapshot and one
deterministic continuity packet under2k. A changed runtime that fails those
checks needs explicit compatibility qualification, not a forced restore.

## Create and verify a backup

Use the complete binding selected for this agent. Both version1 native JSON and
the environment-mapping JSON produced by local setup are supported. Relative
paths and automatic project/identity fallback are refused.

```sh
python3 /absolute/licorice/integrations/being-portable/memory.py backup \
  --binding /absolute/private/binding.json \
  --output /absolute/private/quill-20261025.licorice

python3 /absolute/licorice/integrations/being-portable/memory.py verify \
  --archive /absolute/private/quill-20261025.licorice \
  --agent quill \
  --binary /absolute/versioned-install/bin/licorice-being
```

Choose a new output name. The tool refuses replacement, including a destination
that appears after the initial check. SQLite's backup API captures a consistent
point in time, including committed WAL changes. It does not copy only the main
file. Every relationship is preserved even when the binding selects one scope
for the context proof. A backup made while writers continue is a snapshot, not
an ongoing replica or a guarantee that later turns have been saved.

The two-member version1 ZIP_STORED envelope contains `manifest.json` and
`memory.sqlite`. Its metadata records schema/owner/dimension, event count and
full journal fingerprint, SQLite payload SHA256, source binary SHA256, selected
snapshot fingerprint and a fixed-query bounded context fingerprint. Machine
paths, model files, credentials, harness profiles and transcripts outside the
database are not added. Existing memory text may itself contain paths or other
sensitive material; the payload is the full private database.

Initial local tool bounds:256MiB SQLite payload,100000journal events,64KiB
metadata and32MiB native snapshot JSON. They are export-tool limits, not measured
service capacity, native kernel quotas or a second memory allowance. Compression,
encryption flags, duplicate/extra entries, special files and traversal names are
refused. Envelope validation does not certify the semantic truth of the memories.

## Restore into a fresh directory

For a move or upgrade cutover, stop the old harness cleanly first and take the
final backup. Keep the old installation, binding and store available but inactive.
Select the destination binary explicitly; it need not have the same build hash
if it preserves the qualified schema/snapshot/context contract.

```sh
python3 /absolute/licorice/integrations/being-portable/memory.py restore \
  --archive /absolute/private/quill-20261025.licorice \
  --agent quill \
  --binary /absolute/new-versioned-install/bin/licorice-being \
  --destination /absolute/private/quill-restored-01
```

The expected agent must match both manifest and native store owner. The target
directory must not exist, even if empty. Validation, native checks and durable
files are prepared in private disposable directories. Publication reserves the
fresh name exclusively, then atomically replaces that empty reservation with
the complete validated state on the same POSIX filesystem. A competing writer's
existing directory/files are refused rather than overwritten.

The restored directory contains:

- `memory.sqlite`: byte-exact captured SQLite payload at publication.
- `binding.json`: explicit version1 native binding with relocated paths.
- `memory.env`: shell-quoted identity environment; inherited model selection is
  cleared, so initial recovery is explicitly continuity-only.
- `RESTORED.json`: archive/database/binding digests, owner and revision receipt.

Native validation opens only a separate disposable copy. The captured payload
is not mutated by a validation-time migration or journal-mode change. Optional
WordLlama models are not copied; stored factual data remains present, but factual
retrieval with external models needs its own explicit configuration/check.

If interrupted before publication, the selected name may be an **empty, unbound
directory**; no partial binding is activated. A hard process death may also leave
a private `.licorice-restore-*` staging directory in the parent. Inspect locally
and choose a new destination name; do not rerun with an overwrite option. After
publication, the complete state is present even if the process dies before
printing success. Verify its receipt and recover before using it. This qualifies
process-crash behavior; it is not a guarantee against hardware power loss or
filesystem failure.

## Recovery and rollback

Read the new binding or explicitly source its environment, then recover in a
fresh native process and intended harness session. Retain the existing harness
history. This tool does not edit client history, register profiles, update a
profile's pinned store, grant filesystem access or start Claude/Codex.
The existing named-profile utility currently requires local model paths;
its registry is not automatically moved by this continuity-only restore.

On a failed upgrade, stop the new harness and explicitly select the old matching
binding/runtime and store. Do not open a newer-schema store with an older runtime
or copy a newer store over the old one. Turns written after either cutover stay
in that selected store: this format does not automatically merge divergent
branches. Retain another backup before switching again. Keep only one authoritative
writer during the move; the same identity on two unsynchronized stores is not
cross-machine continuity.

Qualification: `scripts/test_being_portable.py` exercises real native histories,
cross-build restore, live WAL, ownership/schema/digest refusal, destination races,
and actual child-process death before and after atomic publication. Native source
is unchanged; Linux/Windows and cross-schema upgrades remain separate gates.
