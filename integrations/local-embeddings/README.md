# Private CPU embedding sidecar

This candidate supplies vectors to Licorice's existing memory service. It is not
a chat model or an external inference API. The implementation uses Python's
standard-library HTTP server with ONNX Runtime, NumPy and the Rust-backed
`tokenizers` library. It has no PyTorch or Transformers dependency.

The selected model is
[`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2),
at revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. Its model card describes
384-dimensional embeddings, attention-mask mean pooling followed by L2
normalization, and a default maximum of 256 word pieces. This sidecar follows
those operations, with no query or document prefix. Its model-card metadata
declares Apache-2.0. The pinned [pooling configuration](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/1_Pooling/config.json)
and [sentence configuration](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/sentence_bert_config.json)
are also checked locally.

## Assets and runtime

`model-manifest.json` records seven exact files, sizes and SHA256 hashes. The
ONNX file is 90,405,214 bytes; its hash comes from official Hugging Face LFS
metadata. Tokenizer, configuration and README hashes were independently computed
from HTTPS bytes at the pinned revision. Startup verifies every local file
before importing the model. A missing or changed asset fails startup; there is
no downloader or `from_pretrained` fallback.

Acquire those files separately from this fixed URL prefix, retaining each
manifest path:

```text
https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/
```

Store the model outside the public website and credentials. Production should
mount it read-only and isolate the sidecar on a private Docker network with no
published port or Internet egress. Health has no credentials and must not be
mistaken for an authenticated public service. The sidecar accepts no API key.

The top-level packages are pinned in `requirements.txt`; all 24 resolved runtime
versions are preserved in `requirements.versions.txt`. `requirements.lock` fixes
the exact **Linux x86_64 / CPython 3.12** wheel URLs and SHA256 hashes selected
by the recorded resolution. Direct official PyPI URLs prevent a resolver from
silently selecting a different platform wheel. The Docker build requires those
hashes. For local development, install the version pins into a new virtualenv,
never the system interpreter:

```sh
python3.12 -m venv /absolute/private/onnx-venv
/absolute/private/onnx-venv/bin/python -m pip install \
  --only-binary=:all: -r integrations/local-embeddings/requirements.versions.txt

/absolute/private/onnx-venv/bin/python integrations/local-embeddings/server.py \
  --model-dir /absolute/local/minilm --port 8091
```

Local installation was verified in `/tmp/licorice-onnx-venv`, using Python 3.13.2
on Apple Silicon, NumPy 2.2.6, ONNX Runtime 1.23.2 and Tokenizers 0.22.1. Official
PyPI metadata confirms wheels for those top-level versions on Linux x86_64
Python 3.11 and 3.12. This does not by itself establish Linux deployment or real
model retrieval quality. `runtime-provenance.json` records the wheel metadata
and an OSV snapshot: all 24 pinned package queries returned no vulnerability
records on 2026-10-26. That is a database snapshot, not a security guarantee.

A complete cross-platform pip dry run also resolved **all 24 pinned packages**
to Python 3.12 Linux x86_64 wheels, with no missing pins or Windows-only
distributions such as `pyreadline3`. `linux312-wheel-resolution.json` preserves
the selected public wheel URLs and hashes. All names, versions and hashes in
`requirements.lock` were validated against that record, including an allowlist
restricting URLs to public HTTPS `files.pythonhosted.org` without userinfo or
query strings. This checks resolution portability;
the actual Linux container must still pass its import and model qualification.

The Dockerfile targets Linux amd64/Python 3.12 and intentionally has **no default base image**. Build with an
operator-resolved official Python 3.12 image digest:

```sh
docker build \
  --platform linux/amd64 \
  --build-arg PYTHON_IMAGE=python:3.12-slim-bookworm@sha256:REVIEWED_DIGEST \
  -t licorice-local-embeddings:qualified integrations/local-embeddings
```

The container runs as UID/GID 10001. Its `/models/minilm` mount must be readable
by that user. Do not publish port 8091; give the memory service a reviewed static
private IP for the sidecar. The memory service's `LocalOnnx` transport deliberately
does not resolve a Docker hostname. Run the deployed container with a read-only
filesystem, restricted resources and no provider credentials.

`runtime-license-inventory.json` inventories all 24 installed distributions and
the exact Linux wheels. Pip's original wheel notices remain in site-packages;
`/app/notices/` additionally holds recovered FlatBuffers and Tokenizers Apache
grants, the Licorice license, inventory and [material distribution notes](license-obligations.md).
This hosted-image qualification does not qualify a public container distribution
under the bundled native-library terms.

## Protocol and limits

`GET /health` reports model/revision, dimensions, pooling, local runtime version,
asset verification and resource limits. `POST /v1/embeddings` accepts:

```json
{
  "model": "sentence-transformers/all-MiniLM-L6-v2",
  "input": ["A selected project observation.", "A second observation."],
  "encoding_format": "float"
}
```

`/embeddings` is an alias. Optional `kind` may be `query` or `document`; both use
the same encoder with no prefix. The response echoes the model and supplies
`data: [{index, embedding}]` in input order, with `dimensions: 384`.

Each request permits 1–40 strings, at most 4,000 UTF-8 bytes per string and
160,000 combined input bytes. The HTTP body limit is 200,000 bytes. The service
requires one explicit Content-Length and refuses transfer encoding, duplicate
lengths, unexpected fields and unsupported encodings. It logs no request paths,
headers or text. Responses are not cached.

**Encoding truncates inputs after 256 word pieces, including tokenizer special
tokens.** This is separate from the memory service's 2,000-token rendered-context
allowance. `tokenization.truncated_input_indices` identifies affected inputs,
and `encoded_tokens` reports how many tokens actually entered the model. Long
observations therefore still need appropriate retrieval chunking; the sidecar
does not claim to encode their entire contents.

Only CPUExecutionProvider is selected. The default uses two intra-op threads,
one inter-op thread, one active inference and at most four HTTP connections.
Busy requests receive 503 immediately rather than entering an inference queue.
Request reads have a five-second timeout. Inference has an eight-second deadline;
a timer sets ONNX Runtime's per-run `RunOptions.terminate` flag, requests graceful
termination, and suppresses late results. It is not a separate hard-killed model
process. The [official ONNX Runtime Python API](https://onnxruntime.ai/docs/api/python/api_summary.html)
documents that cancellation mechanism. No inference retry occurs automatically.

## Controls and qualification boundary

```sh
/tmp/licorice-onnx-venv/bin/python integrations/local-embeddings/test_server.py
```

The 12 controls passed using synthetic tokenizer and ONNX-session fixtures plus
real loopback HTTP. They verify masked pooling, normalization, dimensions and
order, nonfinite output rejection, input/byte limits, explicit truncation,
busy behavior, options-factory failure cleanup, deadline cancellation, asset failure, route compatibility and
HTTP framing. These controls are not a substitute for loading the actual pinned
model, testing unseen retrieval pairs, measuring Linux CPU use or verifying a
production network boundary. Those results belong in the deployment receipt.
