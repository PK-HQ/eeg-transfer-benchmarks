"""Fetch and verify the fixed NEMAR clinical pilot subset."""

import argparse
import hashlib
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

BASE = "https://data.nemar.org/nm000309/v1.0.1/"
MANIFEST_SHA256 = "8f1d3e7281e98a917937c40b921e00efd14d123e363b05ed2045db4395f7ab89"
ROOT = Path("data/nm000309/v1.0.1")
HEALTHY = {"sub-11": "C01", "sub-12": "02", "sub-13": "C03"}
STROKE = {"sub-1": "PAC01"}
RECORDINGS = HEALTHY | STROKE
COHORT_HEALTHY = HEALTHY | {f"sub-{i + 10}": f"CONTROL{i:02}" for i in range(4, 9)}
COHORT_STROKE = {f"sub-{i}": f"PAC{i:02}" for i in range(1, 11)}


def events_path(subject):
    return f"{subject}/ses-0pre/eeg/{subject}_ses-0pre_task-imagery_run-0_events.tsv"


def verify(path, entry):
    if not path.is_file() or path.stat().st_size != entry["size"]:
        return False
    with path.open("rb") as source:
        if entry["checksum_algorithm"] == "sha256":
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        elif entry["checksum_algorithm"] == "git":
            blob = f"blob {entry['size']}\0".encode() + source.read()
            digest = hashlib.sha1(blob).hexdigest()
        else:
            raise ValueError(f"Unsupported checksum for {path}")
    return digest == entry["checksum"]


def manifest(root):
    contents = (root / "manifest.json").read_bytes()
    if hashlib.sha256(contents).hexdigest() != MANIFEST_SHA256:
        raise ValueError("Unexpected NEMAR v1.0.1 manifest SHA-256")
    return {item["path"]: item for item in json.loads(contents)}


def pilot_paths(recordings=RECORDINGS):
    return [
        "README.md", "CHANGES", "dataset_description.json", "participants.tsv",
        "sourcedata/sourcedata_provenance.json",
        *(events_path(subject) for subject in recordings),
        *(f"sourcedata/{stem}.{ext}" for stem in recordings.values()
          for ext in ("vhdr", "vmrk", "eeg")),
    ]


def fetch(root, recordings=RECORDINGS):
    root.mkdir(parents=True, exist_ok=True)
    index = root / "manifest.json"
    if not index.exists():
        with urlopen(BASE + index.name, timeout=60) as response:
            contents = response.read()
        if hashlib.sha256(contents).hexdigest() != MANIFEST_SHA256:
            raise ValueError("Unexpected NEMAR v1.0.1 manifest SHA-256")
        index.write_bytes(contents)

    entries = manifest(root)
    paths = pilot_paths(recordings)
    print(f"Selected files: {sum(entries[p]['size'] for p in paths):,} bytes")
    def download(relative):
        entry = entries[relative]
        destination = root / relative
        if verify(destination, entry):
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(destination.name + ".part")
        print(f"Downloading {relative}", flush=True)
        try:
            with urlopen(BASE + relative, timeout=90) as response, partial.open("wb") as output:
                shutil.copyfileobj(response, output)
            if not verify(partial, entry):
                raise ValueError(f"Manifest integrity check failed for {relative}")
            partial.replace(destination)
        finally:
            partial.unlink(missing_ok=True)
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(download, paths))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT)
    parser.add_argument("--cohort", action="store_true")
    args = parser.parse_args()
    fetch(args.data, COHORT_HEALTHY | COHORT_STROKE if args.cohort else RECORDINGS)
