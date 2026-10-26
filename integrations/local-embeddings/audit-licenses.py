#!/usr/bin/env python3
"""Read installed metadata and exact Linux wheels; never import package code."""
import argparse
from email.parser import BytesParser
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import re
from zipfile import ZipFile

HERE = Path(__file__).resolve().parent
LICENSES = {
    "anyio": "MIT", "certifi": "MPL-2.0", "click": "BSD-3-Clause",
    "coloredlogs": "MIT", "filelock": "MIT", "flatbuffers": "Apache-2.0",
    "fsspec": "BSD-3-Clause", "h11": "MIT", "hf-xet": "Apache-2.0",
    "httpcore": "BSD-3-Clause", "httpx": "BSD-3-Clause", "huggingface-hub": "Apache-2.0",
    "humanfriendly": "MIT", "idna": "BSD-3-Clause", "mpmath": "BSD-3-Clause",
    "numpy": "BSD-3-Clause (primary; bundled components have additional terms)",
    "onnxruntime": "MIT (primary; bundled components have additional terms)",
    "packaging": "Apache-2.0 OR BSD-2-Clause", "protobuf": "BSD-3-Clause",
    "pyyaml": "MIT", "sympy": "BSD-3-Clause", "tokenizers": "Apache-2.0",
    "tqdm": "MPL-2.0 AND MIT", "typing-extensions": "PSF-2.0",
}


def canonical(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def is_notice(path):
    return bool(re.search(r"licen[cs]e|notice|copying|copyright|authors", Path(path).name, re.I))


def notice(path, data):
    return {"path": str(path), "bytes": len(data), "sha256": sha(data)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=HERE / "runtime-license-inventory.json")
    args = parser.parse_args()
    resolution = json.loads((HERE / "linux312-wheel-resolution.json").read_text())
    supplements = json.loads((HERE / "notices/provenance.json").read_text())["items"]
    entries = []
    for wheel in resolution["items"]:
        name = canonical(wheel["name"])
        installed = metadata.distribution(wheel["name"])
        assert installed.version == wheel["version"]
        local_notices = [notice(path, installed.locate_file(path).read_bytes()) for path in installed.files or [] if is_notice(str(path)) and installed.locate_file(path).is_file()]
        archive = args.wheel_dir / wheel["filename"]
        assert sha(archive.read_bytes()) == wheel["sha256"], "exact wheel checksum mismatch"
        with ZipFile(archive) as zipped:
            metadata_path = next(path for path in zipped.namelist() if path.endswith(".dist-info/METADATA"))
            raw_metadata = zipped.read(metadata_path)
            linux_metadata = BytesParser().parsebytes(raw_metadata)
            assert canonical(linux_metadata["Name"]) == name and linux_metadata["Version"] == wheel["version"]
            linux_notices = [notice(path, zipped.read(path)) for path in zipped.namelist() if is_notice(path) and not path.endswith("/")]
            summary = {"name": linux_metadata["Name"], "version": linux_metadata["Version"], "license": LICENSES[name],
                       "evidence": "installed dist metadata plus exact hash-verified Linux wheel metadata and license files",
                       "pypi": "https://pypi.org/project/" + name + "/" + wheel["version"] + "/",
                       "wheel": {key: wheel[key] for key in ["url", "filename", "sha256"]},
                       "upstream_metadata_urls": linux_metadata.get_all("Project-URL", []),
                       "metadata_license_expression": linux_metadata.get("License-Expression"),
                       "metadata_license_field_prefix": linux_metadata.get("License", "")[:180],
                       "metadata_sha256": sha(raw_metadata), "installed_notice_files": local_notices, "linux_wheel_notice_files": linux_notices,
                       "supplemental_notices": [item for item in supplements if canonical(item["package"]) == name]}
            if name == "numpy":
                license_text = zipped.read(next(path for path in zipped.namelist() if path.endswith(".dist-info/LICENSE.txt"))).decode()
                components = []
                current = None
                for line in license_text.splitlines():
                    if line.startswith("Name:"):
                        current = {"name": line.split(":", 1)[1].strip()}
                        components.append(current)
                    elif current is not None and line.startswith("License:"):
                        current["license_as_written"] = line.split(":", 1)[1].strip()
                summary["bundled_component_notices"] = components
        assert summary["linux_wheel_notice_files"] or summary["supplemental_notices"], "a distribution has no retained grant"
        entries.append(summary)
    assert len(entries) == 24 and len({canonical(item["name"]) for item in entries}) == 24
    inventory = {"schema": "licorice-embedding-license-inventory/1", "checked_on": "2026-10-26", "packages": sorted(entries, key=lambda value: canonical(value["name"])),
                 "local_metadata_environment": {"python": platform.python_version(), "system": platform.system(), "machine": platform.machine()},
                 "deployment_wheels": "Linux x86_64 CPython 3.12; exact requirements.lock and linux312-wheel-resolution.json",
                 "retention": "pip retains wheel license files in site-packages; Docker additionally copies supplemental notices and this inventory into /app/notices/",
                 "limitations": "Package/notice inventory, not a complete native dependency-closure or legal opinion. ONNX Runtime ThirdPartyNotices is retained wholesale; its entries are not asserted to prove every listed component is linked in this particular build."}
    args.output.write_text(json.dumps(inventory, indent=2) + "\n")
    print(json.dumps({"packages": 24, "linux_notice_files": sum(len(item["linux_wheel_notice_files"]) for item in entries), "supplemented_distributions": [item["name"] for item in entries if item["supplemental_notices"]], "output": str(args.output)}))


if __name__ == "__main__":
    main()
