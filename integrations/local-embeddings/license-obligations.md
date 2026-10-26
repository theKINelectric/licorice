# Runtime notice inventory and distribution boundaries

The pinned Licorice embedding runtime has 24 Python distributions. The
machine-readable `runtime-license-inventory.json` records package names,
versions, declared primary licenses, exact Linux wheel URLs/hashes, installed
and Linux-wheel notice files, and upstream project URLs found in metadata.
The audit does not import dependency code.

Keep all wheel `LICENSE`, `LICENCE`, copyright and notice files in site-packages.
In particular, retain NumPy's complete 47,755-byte Linux `LICENSE.txt` and ONNX
Runtime's 326,866-byte `ThirdPartyNotices.txt`; replacing either with a short
primary-license label loses material terms. The Dockerfile does not strip them.

Two wheels omit their full grant files: FlatBuffers 25.12.19 and Tokenizers
0.22.1. Their unmodified Apache-2.0 licenses were recovered from exact upstream
version tags resolved to commit hashes, and are copied into `/app/notices/`.
`notices/provenance.json` records those sources and hashes. The Licorice component's
own Apache-2.0 license is included there as well.

The material license groups are:

- **MIT and BSD:** retain the relevant copyright, conditions and disclaimers;
  preserve non-endorsement clauses where present.
- **Apache-2.0:** retain the license and relevant attribution/notice material;
  follow its change-notice requirements when modifying covered files.
- **MPL-2.0:** Certifi and part of tqdm carry file-level MPL terms, alongside
  MIT-covered material in tqdm. Do not relabel those distributions Apache-2.0;
  preserve their notices and source-availability terms when redistributing them.
- **PSF-2.0:** typing_extensions includes its Python license history; preserve
  the complete supplied license file.
- **NumPy's bundled native libraries:** its Linux notice identifies OpenBLAS
  and LAPACK BSD terms, GCC runtime GPL terms with the GCC Runtime Library
  Exception, and dynamically linked libquadmath under LGPL-2.1-or-later. Preserve
  those grants and exceptions. An externally redistributed runtime image needs
  the corresponding applicable native-library source/relinking compliance work;
  this package inventory does not declare that future distribution gate complete.

The present hosted service runs these unmodified dependencies on our server.
Its users receive memory API responses, not copies of these native libraries.
That operating boundary does not convert their licenses into Apache-2.0 and
must not be confused with qualifying a publicly downloadable container image.

ONNX Runtime's third-party notice lists additional components and terms. The
file is retained wholesale. This inventory does not infer that every entry in
that upstream notice proves a linked component in this particular wheel.

The separately mounted model remains subject to its upstream Apache-2.0 model
card and model-asset provenance. The model README is hash-verified and retained.

Reproduce this inventory with the installed qualification environment and the
exact locked Linux wheels:

```sh
/tmp/licorice-onnx-venv/bin/python integrations/local-embeddings/audit-licenses.py \
  --wheel-dir /tmp/licorice-onnx-linux312-license-wheels
```
