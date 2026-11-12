"""Download and verify the Damm2026 snapshot inputs."""

import argparse
import hashlib
import json
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

from digit_timing_audit import S3, SNAPSHOT

ROOT = Path("data/ds008446/1.0.1")
REPO = "OpenNeuroDatasets/ds008446"
GIT_RAW = f"https://raw.githubusercontent.com/{REPO}/{SNAPSHOT}/"
SUBJECTS = tuple(f"sub-{i:02}" for i in range(1, 21))
STIMULI = tuple(f"sourcedata/Visual Stimuli/{i}.jpg" for i in range(7))


def runs(cohort):
    if not cohort:
        return [f"{s}/eeg/{s}_task-random_run-01" for s in SUBJECTS[:4]]
    return [f"{s}/eeg/{s}_task-{task}_run-{run:02}" for s in SUBJECTS
            for task in ("random", "sequential") for run in (1, 2)]


def read_url(url):
    with urlopen(url, timeout=90) as response:
        return response.read()


def snapshot_tree(root, fetch=False):
    index = root / "tree.json"
    if not index.exists() and fetch:
        index.write_bytes(read_url(f"https://api.github.com/repos/{REPO}/git/trees/{SNAPSHOT}?recursive=1"))
    tree = json.loads(index.read_bytes())
    if tree["sha"] != SNAPSHOT or tree["truncated"]:
        raise ValueError("Unexpected Git snapshot tree")
    return {e["path"]: e for e in tree["tree"] if e["type"] == "blob"}


def check_blob(contents, entry):
    blob = f"blob {len(contents)}\0".encode() + contents
    if hashlib.sha1(blob).hexdigest() != entry["sha"]:
        raise ValueError(f"Git blob verification failed for {entry['path']}")


def annex_entry(root, relative, tree, fetch=False):
    pointer = root / (relative + ".git-pointer")
    if not pointer.exists() and fetch:
        pointer.parent.mkdir(parents=True, exist_ok=True)
        contents = read_url(GIT_RAW + quote(relative))
        check_blob(contents, tree[relative])
        pointer.write_bytes(contents)
    contents = pointer.read_bytes()
    check_blob(contents, tree[relative])
    key = re.search(rb"SHA256E-s(\d+)--([a-f0-9]{64})", contents)
    if key is None:
        raise ValueError(f"Expected SHA256E annex key for {relative}")
    return dict(path=relative, size=int(key[1]), sha256=key[2].decode(),
                pointer_git_sha1=tree[relative]["sha"], url=S3 + quote(relative))


def valid(path, entry):
    if not path.is_file() or path.stat().st_size != entry["size"]:
        return False
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() == entry["sha256"]


def download(root, cohort, stimuli_only=False):
    root.mkdir(parents=True, exist_ok=True)
    tree = snapshot_tree(root, fetch=True)
    for name in ("participants.tsv", "dataset_description.json"):
        path = root / name
        if not path.exists():
            path.write_bytes(read_url(GIT_RAW + name))
        check_blob(path.read_bytes(), tree[name])

    def get(relative):
        entry = annex_entry(root, relative, tree, fetch=True)
        path = root / relative
        if valid(path, entry):
            return
        partial = path.with_suffix(path.suffix + ".part")
        try:
            with urlopen(entry["url"], timeout=120) as response, partial.open("wb") as output:
                shutil.copyfileobj(response, output)
            if not valid(partial, entry):
                raise ValueError(f"Annex verification failed for {relative}")
            partial.replace(path)
            print(relative, flush=True)
        finally:
            partial.unlink(missing_ok=True)

    paths = STIMULI if stimuli_only else [
        run + suffix for run in runs(cohort) for suffix in ("_events.tsv", "_eeg.edf")]
    with ThreadPoolExecutor(max_workers=10) as pool:
        list(pool.map(get, paths))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--cohort", action="store_true")
    mode.add_argument("--stimuli-only", action="store_true")
    args = parser.parse_args()
    download(args.data, args.cohort, args.stimuli_only)
