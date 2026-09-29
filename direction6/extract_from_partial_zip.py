#!/usr/bin/env python3
"""
Extract wanted members from a TRUNCATED zip (no central directory) by walking the
local file headers sequentially. Used because media.talkbank.org closes long
transfers at ~10 min and does not support byte ranges, so TinyVox.zip can only be
downloaded partially.

Handles both header styles:
  * sizes in the local header (python/zip/7z archives) -> jump straight to the next member
  * streamed archives with data descriptors (macOS Archive Utility, flag bit 3, sizes 0 in
    the header) -> scan forward for the PK\\x07\\x08 descriptor whose compressed size matches
    the distance travelled and which is followed by the next header (or EOF)

  python direction6/extract_from_partial_zip.py --zip tinyvox_en/TinyVox.zip \\
         --manifest tinyvox_en/audio_manifest_en.csv --out tinyvox_en/audio [--list-only]
"""
import argparse
import mmap
import struct
import sys
import zlib
from collections import Counter
from pathlib import Path

import pandas as pd

LOCAL_SIG = b"PK\x03\x04"
CENTRAL_SIG = b"PK\x01\x02"
DESC_SIG = b"PK\x07\x08"

ap = argparse.ArgumentParser()
ap.add_argument("--zip", required=True, type=Path)
ap.add_argument("--manifest", required=True, type=Path, help="csv with an audio_filename column")
ap.add_argument("--out", required=True, type=Path)
ap.add_argument("--list-only", action="store_true", help="scan and report, write nothing")
args = ap.parse_args()

wanted = set(pd.read_csv(args.manifest, usecols=["audio_filename"]).audio_filename)
args.out.mkdir(parents=True, exist_ok=True)
size = args.zip.stat().st_size


def find_descriptor(mm, data_start):
    """Locate the data descriptor that ends the member starting at data_start.
    Returns (crc, csize, usize, next_pos) or None if the file ends first."""
    pos = data_start
    n = len(mm)
    while True:
        idx = mm.find(DESC_SIG, pos)
        if idx < 0:
            return None
        for fmt, ln in (("<III", 16), ("<IQQ", 24)):        # 32-bit sizes, then zip64
            end = idx + ln
            if end > n:
                continue
            crc, cs, us = struct.unpack(fmt, mm[idx + 4:end])
            if cs == idx - data_start and (end == n or mm[end:end + 4] in (LOCAL_SIG, CENTRAL_SIG)):
                return crc, cs, us, end
        pos = idx + 1


seen = Counter()
found = written = skipped_existing = 0
truncated_member = None
first_name = last_name = None
n = 0

with open(args.zip, "rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm:
    pos = 0
    while pos + 30 <= size:
        sig = mm[pos:pos + 4]
        if sig == CENTRAL_SIG:
            print("reached the central directory: the zip is actually complete")
            break
        if sig != LOCAL_SIG:
            print(f"lost sync at byte {pos:,} (no local header signature); stopping")
            break
        (_, _, flags, method, _, _, crc, csize, usize, nlen, xlen) = struct.unpack("<4sHHHHHIIIHH", mm[pos:pos + 30])
        name = mm[pos + 30:pos + 30 + nlen].decode("utf-8", "replace")
        data_start = pos + 30 + nlen + xlen
        base = Path(name).name
        is_meta = name.startswith("__MACOSX/") or base.startswith("._") or name.endswith("/")

        if flags & 0x08:                                     # streamed member: sizes follow the data
            d = find_descriptor(mm, data_start)
            if d is None:
                truncated_member = base
                break
            crc, csize, usize, next_pos = d
        else:
            next_pos = data_start + csize
            if next_pos > size:
                truncated_member = base
                break

        if not is_meta:
            prefix = "_".join(base.split("_")[:2]) if base.endswith(".wav") else "(non-wav)"
            seen[prefix] += 1
            if first_name is None:
                first_name = base
            last_name = base
            if base in wanted:
                found += 1
                target = args.out / base
                if not args.list_only:
                    if target.exists() and target.stat().st_size == usize:
                        skipped_existing += 1
                    else:
                        raw = mm[data_start:data_start + csize]
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
        n += 1
        pos = next_pos
        if n % 20000 == 0:
            print(f"  scanned {n:,} entries, {pos/1e9:.1f} GB, found {found:,} wanted", flush=True)

print(f"\nscanned {n:,} entries up to byte {pos:,} of {size:,} ({100*pos/size:.1f}% of the file on disk)")
print(f"first clip: {first_name}\nlast complete clip: {last_name}")
if truncated_member:
    print(f"cut off inside: {truncated_member}")
print("\nclips seen by prefix (language/group):")
for k, v in sorted(seen.items()):
    print(f"  {k:24s} {v:,}")
print(f"\nwanted (manifest): {len(wanted):,}   found in partial zip: {found:,}   "
      f"written: {written:,}   already present: {skipped_existing:,}   missing: {len(wanted)-found:,}")
if not args.list_only:
    missing = wanted - {p.name for p in args.out.glob("*.wav")}
    (args.out.parent / "missing_audio.txt").write_text("\n".join(sorted(missing)))
    print(f"missing list written to {args.out.parent/'missing_audio.txt'}")
