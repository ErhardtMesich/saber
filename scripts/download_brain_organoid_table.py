#!/usr/bin/env python3
"""Download the published organoid dataset authors' tabular measurements."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

COMMIT = "9334db92c59e381afd843fca0338ea78bffd14e2"
BLOB = "d6017afd6134c6c993bbe607f542fbeaae4518a6"
SHA256 = "472a9da6e5d2eab5081690aab987b1c41847e16721d3f79b6fbec6144f5e33dd"
REPO = "deiluca/robust_monitoring_organoid_growth"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/public/brain_organoid_author_table"))
    parser.add_argument("--verify-existing", action="store_true", help="verify an already downloaded file")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / "df_pred.csv"
    if not args.verify_existing:
        result = subprocess.run(["gh", "api", f"repos/{REPO}/contents/df_pred.csv?ref={COMMIT}",
                                 "-H", "Accept: application/vnd.github.raw+json"],
                                check=True, capture_output=True)
        content = result.stdout
    else:
        content = target.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    blob = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
    if digest != SHA256 or blob != BLOB:
        raise RuntimeError(f"Data checksum mismatch: sha256={digest}, git_blob={blob}")
    target.write_bytes(content)
    manifest = {"source": f"https://github.com/{REPO}/blob/{COMMIT}/df_pred.csv",
                "dataset_doi": "10.5281/zenodo.10301912",
                "article_doi": "10.1038/s41597-024-03330-z", "repository_commit": COMMIT,
                "github_blob_sha1": BLOB, "sha256": SHA256, "bytes": len(content),
                "license": "CC BY 4.0 dataset; repository MIT"}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Verified {target} ({len(content)} bytes, SHA-256 {digest})")


if __name__ == "__main__":
    main()
