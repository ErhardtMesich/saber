#!/usr/bin/env python3
"""Fetch selected members of the published Zenodo ZIP using HTTP byte ranges.

Only original images, their author-provided masks and the dataset CSV are saved.
No full 974 MB archive is downloaded. Each ZIP member is CRC-checked.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
import tempfile
import zlib
from pathlib import Path

URL = "https://zenodo.org/api/records/10301912/files/data.zip/content"
ARCHIVE_SIZE = 973_563_674
ARCHIVE_MD5 = "53e6b41af957ac33c03b927e0fbe69aa"
IDS = tuple(f"{org}_d{day}_LabA" for org in ("org03_wt2D", "org23_TH2-7") for day in ("02", "10", "22"))


def fetch_range(start: int, end: int) -> bytes:
    with tempfile.TemporaryDirectory() as tmp:
        output, headers = Path(tmp) / "bytes", Path(tmp) / "headers"
        subprocess.run(
            ["curl", "-fLsS", "--retry", "3", "--max-time", "90", "-r", f"{start}-{end}",
             "-D", str(headers), "-o", str(output), URL], check=True
        )
        header = headers.read_text(errors="replace").lower()
        expected = f"content-range: bytes {start}-{end}/{ARCHIVE_SIZE}"
        data = output.read_bytes()
        if expected not in header or len(data) != end - start + 1:
            raise RuntimeError(f"Server did not return the requested byte range {start}-{end}")
        return data


def members() -> dict[str, tuple[int, int, int, int, int]]:
    tail_start = ARCHIVE_SIZE - 65_536
    tail = fetch_range(tail_start, ARCHIVE_SIZE - 1)
    end = tail.rfind(b"PK\x05\x06")
    if end < 0:
        raise RuntimeError("ZIP end-of-central-directory record not found")
    sig, disk, cd_disk, disk_count, total, cd_size, cd_offset, comment = struct.unpack_from("<4s4H2LH", tail, end)
    if disk or cd_disk or disk_count != total or comment or cd_offset + cd_size + 22 != ARCHIVE_SIZE:
        raise RuntimeError("Unexpected ZIP structure")
    directory = fetch_range(cd_offset, cd_offset + cd_size - 1)
    result: dict[str, tuple[int, int, int, int, int]] = {}
    pos = 0
    while pos < len(directory):
        fields = struct.unpack_from("<4s6H3L5H2L", directory, pos)
        if fields[0] != b"PK\x01\x02":
            raise RuntimeError("Invalid central directory")
        name_len, extra_len, comment_len = fields[10:13]
        name = directory[pos + 46:pos + 46 + name_len].decode("utf-8")
        result[name] = (fields[-1], fields[8], fields[9], fields[7], fields[4])
        pos += 46 + name_len + extra_len + comment_len
    if len(result) != total:
        raise RuntimeError("Unexpected number of ZIP entries")
    return result


def extract_member(name: str, info: tuple[int, int, int, int, int]) -> bytes:
    offset, compressed_size, size, crc, method = info
    header = fetch_range(offset, offset + 29)
    fields = struct.unpack("<4s5H3L2H", header)
    if fields[0] != b"PK\x03\x04":
        raise RuntimeError(f"Invalid local ZIP header: {name}")
    start = offset + 30 + fields[-2] + fields[-1]
    compressed = fetch_range(start, start + compressed_size - 1)
    if method == 8:
        data = zlib.decompress(compressed, -15)
    elif method == 0:
        data = compressed
    else:
        raise RuntimeError(f"Unsupported compression method {method}: {name}")
    if len(data) != size or zlib.crc32(data) != crc:
        raise RuntimeError(f"Size or CRC mismatch: {name}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/public/brain_organoid_zenodo_10301912"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    entries = members()
    paths = ["data/dataset_overview.csv"]
    for image_id in IDS:
        paths += [f"data/imgs/{image_id}.jpg", f"data/labels/{image_id}.npy"]
    records = []
    for name in paths:
        data = extract_member(name, entries[name])
        target = args.out / name.removeprefix("data/")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        records.append({"archive_member": name, "path": str(target.relative_to(args.out)),
                        "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
        print(f"Downloaded {name}: {len(data)} bytes", flush=True)
    manifest = {"dataset_doi": "10.5281/zenodo.10301912", "article_doi": "10.1038/s41597-024-03330-z",
                "archive_url": URL, "archive_size_bytes": ARCHIVE_SIZE, "archive_published_md5": ARCHIVE_MD5,
                "archive_integrity": "Selected members verified with ZIP CRC-32; full archive MD5 not checked.",
                "license": "CC BY 4.0", "image_ids": IDS, "files": records}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
