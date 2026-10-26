# FP16 attribution repair

The three bundled FP16 headers came through sqlite-vector1.1.2 without a local
license file. Preserve upstream MIT license and all three copyright notices.
License source: https://github.com/Maratyszcza/FP16/blob/782eea126dc5c755827be751a099eb01826175cf/LICENSE
License SHA256:17e4f539024be2749ee729d1e2f01d24cef12ece8c9bf18e91a4349be29c80bf.

Comparison at that commit: macros.h is byte-exact; bitcasts.h differs in compiler
include guards; fp16.h differs in explicit numeric casts. This identifies the
upstream family, not an exact original vendoring commit for all three headers.
The existing header bytes were retained; no numerical implementation changed.
Diffs and hashes: runs/portable-memory-20261025/fp16-upstream/receipt.json.

The broader native/subcomponent provenance audit remains open. The frozen
0.2.0-alpha.1 archive is not silently rewritten by this source-only notice fix.
