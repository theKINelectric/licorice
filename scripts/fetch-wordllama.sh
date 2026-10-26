#!/usr/bin/env sh
# Fetches WordLlama's l2_supercat 256-d weights and tokenizer (MIT, Lee Miller) from the
# wordllama 0.4.0.post1 package on PyPI into models/wordllama/. Used as Licorice's
# no-network fallback embedder; see crates/licorice-providers/src/embed/wordllama.rs.
set -eu
root="$(cd "$(dirname "$0")/.." && pwd)"
dest="$root/models/wordllama"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
python3 -m pip download --quiet --no-deps wordllama==0.4.0.post1 -d "$work"
pkg="$(ls "$work"/wordllama-* | head -n 1)"
mkdir -p "$dest"
python3 -I - "$pkg" "$dest" <<'PY'
import sys, zipfile, tarfile, pathlib
pkg, dest = sys.argv[1], pathlib.Path(sys.argv[2])
wanted = {"weights/l2_supercat_256.safetensors", "tokenizers/l2_supercat_tokenizer_config.json"}
def store(name, data):
    for w in wanted:
        if name.endswith("wordllama/" + w):
            (dest / pathlib.Path(w).name).write_bytes(data)
if pkg.endswith(".whl"):
    with zipfile.ZipFile(pkg) as z:
        for n in z.namelist():
            if any(n.endswith(w) for w in wanted):
                store(n, z.read(n))
else:
    with tarfile.open(pkg) as t:
        for m in t.getmembers():
            if any(m.name.endswith(w) for w in wanted):
                store(m.name, t.extractfile(m).read())
missing = [w for w in wanted if not (dest / pathlib.Path(w).name).exists()]
sys.exit(f"missing from {pkg}: {missing}" if missing else 0)
PY
echo "WordLlama files in $dest"
