#!/usr/bin/env python3
"""
Extract wanted members from a TRUNCATED zip (no central directory) by walking the
local file headers sequentially. Used because media.talkbank.org closes long
transfers at ~10 min and does not support byte ranges, so TinyVox.zip can only be
downloaded partially.

  python direction6/extract_from_partial_zip.py --zip tinyvox_en/TinyVox.zip \
         --manifest tinyvox_en/audio_manifest_en.csv --out tinyvox_en/audio

Prints how far through the archive it got, how many wanted files were found, and
which corpora (by filename prefix) it saw — so you can tell whether the English
clips were all inside the part you have.
"""
import argparse
import struct
import sys
import zlib
from collections import Counter
from pathlib import Path

import pandas as pd

LOCAL_SIG = b"PK\x03\x04"
CENTRAL_SIG = b"PK\x01\x02"

ap = argparse.ArgumentParser()
ap.add_argument("--zip", required=True, type=Path)
ap.add_argument("--manifest", required=True, type=Path, help="csv with an audio_filename column")
ap.add_argument("--out", required=True, type=Path)
ap.add_argument("--list-only", action="store_true", help="scan and report, write nothing")
args = ap.parse_args()

wanted = set(pd.read_csv(args.manifest, usecols=["audio_filename"]).audio_filename)
args.out.mkdir(parents=True, exist_ok=True)
size = args.zip.stat().st_size

seen = Counter()          # prefix -> members seen
found, written, skipped_existing = 0, 0, 0
truncated_member = None
first_name, last_name = None, None

with open(args.zip, "rb") as f:
    pos = 0
    n = 0
    while True:
        f.seek(pos)
        hdr = f.read(30)
        if len(hdr) < 30:
            break
        if hdr[:4] == CENTRAL_SIG:
            print("reached the central directory: the zip is actually complete")
            break
        if hdr[:4] != LOCAL_SIG:
            print(f"lost sync at byte {pos:,} (no local header signature); stopping")
            break
        (_, _, flags, method, _, _, crc, csize, usize, nlen, xlen) = struct.unpack("<4sHHHHHIIIHH", hdr)
        name = f.read(nlen).decode("utf-8", "replace")
        f.seek(xlen, 1)
        data_start = pos + 30 + nlen + xlen
        if flags & 0x08 and csize == 0:
            # sizes live in a trailing data descriptor; can't know the extent without the
            # central directory. Rare for python/zip-written archives; bail loudly.
            sys.exit(f"member {name} uses a data descriptor (streamed zip); this walker can't handle that")
        base = Path(name).name
        prefix = "_".join(base.split("_")[:2]) if base.endswith(".wav") else "(non-wav)"
        seen[prefix] += 1
        if first_name is None:
            first_name = base
        last_name = base
        n += 1
        end = data_start + csize
        if end > size:
            truncated_member = base
            break
        if base in wanted:
            found += 1
            target = args.out / base
            if not args.list_only:
                if target.exists() and target.stat().st_size == usize:
                    skipped_existing += 1
                else:
                    f.seek(data_start)
                    raw = f.read(csize)
                    if method == 0:
                        data = raw
                    elif method == 8:
                        data = zlib.decompress(raw, -15)
                    else:
                        sys.exit(f"unsupported compression method {method} for {name}")
                    if zlib.crc32(data) & 0xFFFFFFFF != crc:
                        print(f"CRC mismatch on {base}; skipping")
                    else:
                        target.write_bytes(data)
                        written += 1
        pos = end
        if n % 20000 == 0:
            print(f"  scanned {n:,} members, {pos/1e9:.1f} GB, found {found:,} wanted", flush=True)

print(f"\nscanned {n:,} members up to byte {pos:,} of {size:,} ({100*pos/size:.1f}% of the file on disk)")
print(f"first member: {first_name}\nlast complete member: {last_name}")
if truncated_member:
    print(f"cut off inside: {truncated_member}")
print(f"\nmembers seen by prefix (language/group): ")
for k, v in sorted(seen.items()):
    print(f"  {k:24s} {v:,}")
print(f"\nwanted (manifest): {len(wanted):,}   found in partial zip: {found:,}   "
      f"written: {written:,}   already present: {skipped_existing:,}   missing: {len(wanted)-found:,}")
missing = wanted - {p.name for p in args.out.glob('*.wav')} if not args.list_only else None
if missing is not None:
    (args.out.parent / "missing_audio.txt").write_text("\n".join(sorted(missing)))
    print(f"missing list written to {args.out.parent/'missing_audio.txt'}")
