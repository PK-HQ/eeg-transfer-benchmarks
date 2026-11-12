"""Check MOABB's timing reconstruction on one pinned OpenNeuro event table."""

import argparse
import csv
import hashlib
import io
import json
from collections import Counter
from itertools import pairwise
from pathlib import Path
from urllib.request import Request, urlopen

S3 = "https://s3.amazonaws.com/openneuro.org/ds008446/"
RUN = "sub-01/eeg/sub-01_task-random_run-01"
SNAPSHOT = "8d8448e2dcd731b251f2f3c85b98cce5847cb000"
EVENT_SHA256 = "336c6c7963d975667765d3cbc87077f89ea7471a585aa46f70d99bbc3649e2ce"
EDF_PREFIX_SHA256 = "a2c7dc2f2371b6f12bbae39248f95c57c5678772c6c3b6110bd600d09bce0936"
PREFIX_BYTES = 2048
FS = 512
FINGERS = dict(enumerate(("thumb", "index", "middle", "ring", "pinky"), start=3))


def reconstruct(events, edf_samples, shortened=False, marker_codes=None):
    blocks = []
    last_sample = None
    rows = csv.DictReader(io.StringIO(events.decode("utf-8")), delimiter="\t")
    count = 0
    max_rounding_error = 0.0
    for row in rows:
        code = int(row["trial_type"])
        scaled = float(row["onset"]) * FS**2
        sample = round(scaled)
        max_rounding_error = max(max_rounding_error, abs(sample - scaled))
        if (code not in {1, 2, *FINGERS} or abs(sample - scaled) > 0.01
                or not 0 <= sample < edf_samples
                or (last_sample is not None and sample != last_sample + 1)):
            raise ValueError(f"Invalid reconstructed sample sequence at {sample}")
        if marker_codes is not None and code != marker_codes[sample]:
            raise ValueError(f"EDF/TSV marker disagreement at sample {sample}")
        if not blocks or code != blocks[-1][0]:
            blocks.append((code, sample))
        last_sample = sample
        count += 1
    marker_stop = last_sample + 1 if blocks else 0
    zero_padding = (shortened and marker_codes is not None
                    and not any(marker_codes[marker_stop:]))
    if not blocks or (marker_stop != edf_samples and not zero_padding):
        raise ValueError("Reconstructed marker stream does not reach the EDF boundary")
    if blocks[0][0] != 1:
        raise ValueError("Expected white-screen marker at the start")
    cues = []
    for (code, start), (next_code, end) in pairwise([*blocks, (None, marker_stop)]):
        expected = 3 * FS if code in (1, 2) else 6 * FS
        valid_next = {2} if code == 1 else (FINGERS.keys() | {None} if code == 2 else {1})
        if shortened and next_code is None and 0 < end - start <= expected:
            if code in FINGERS and end - start == expected:
                cues.append((start, code))
            break
        if end - start != expected or next_code not in valid_next:
            raise ValueError(f"Unexpected marker block at sample {start}")
        if code in FINGERS:
            cues.append((start, code))
    fingers = Counter(FINGERS[code] for _, code in cues)
    if set(fingers) != set(FINGERS.values()) or (
        not shortened and any(fingers[label] != 15 for label in FINGERS.values())
    ):
        raise ValueError("Unexpected run class counts")
    return cues, {
        "rows": count, "blocks": len(blocks), "first_marker_sample": blocks[0][1],
        "first_finger_sample": cues[0][0], "last_finger_sample": cues[-1][0],
        "fingers": dict(fingers), "edf_duration_seconds": edf_samples / FS,
        "last_marker_sample": last_sample, "last_marker_code": blocks[-1][0],
        "last_marker_block_samples": marker_stop - blocks[-1][1],
        "edf_trailing_padding_samples": edf_samples - marker_stop,
        "max_rounding_error_samples": max_rounding_error,
    }


def audit(events, header):
    if hashlib.sha256(events).hexdigest() != EVENT_SHA256:
        raise ValueError("Event table differs from the OpenNeuro 1.0.1 annex object")
    if hashlib.sha256(header).hexdigest() != EDF_PREFIX_SHA256:
        raise ValueError("Unexpected EDF header prefix for the audited run")
    channels = int(header[252:256])
    labels = [header[i:i + 16].strip() for i in range(256, 256 + 16 * channels, 16)]
    if header[:8] != b"0       " or channels != 65 or labels[-3:] != [b"Ref1", b"Ref2", b"Marker"]:
        raise ValueError("Unexpected EDF format or acquisition layout")
    edf_samples = round(int(header[236:244]) * float(header[244:252]) * FS)
    _, details = reconstruct(events, edf_samples)
    return {"snapshot_commit": SNAPSHOT, "run": RUN, "events_sha256": EVENT_SHA256,
            "edf_prefix_bytes": PREFIX_BYTES, "edf_prefix_sha256": EDF_PREFIX_SHA256,
            "edf_channels": channels, **details}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cached", type=Path, help="Directory with the event table and EDF prefix")
    args = parser.parse_args()
    if args.cached is not None:
        stem = Path(RUN).name
        events = (args.cached / f"{stem}_events.tsv").read_bytes()
        header = (args.cached / f"{stem}_eeg.edf.prefix").read_bytes()
    else:
        with urlopen(S3 + RUN + "_events.tsv", timeout=60) as response:
            events = response.read()
        request = Request(S3 + RUN + "_eeg.edf", headers={"Range": f"bytes=0-{PREFIX_BYTES - 1}"})
        with urlopen(request, timeout=60) as response:
            header = response.read(PREFIX_BYTES)
    print(json.dumps(audit(events, header), indent=2))


if __name__ == "__main__":
    main()
