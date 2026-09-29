#!/usr/bin/env python3
"""
Recreate TinyVox clips that are missing from tinyvox_en/audio by downloading only the
PhonBank session recordings they came from and cutting them the way the TinyVox
pipeline did (ffmpeg -> 16 kHz mono pcm_s16le, then sample-accurate slice by onset/offset).

  python direction6/fill_missing_from_originals.py --root tinyvox_en --plan    # sessions + size, no download
  python direction6/fill_missing_from_originals.py --root tinyvox_en           # do it

Inputs : <root>/metadata_en.csv, <root>/missing_audio.txt (from extract_from_partial_zip.py)
Outputs: clips into <root>/audio, <root>/sessions/ (deleted per session unless --keep-sessions),
         <root>/fill_report.txt
Needs  : ffmpeg on PATH, soundfile, pandas, requests; TALKBANK_EMAIL / TALKBANK_PASSWORD
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import requests
import soundfile as sf

MEDIA = "https://media.talkbank.org/phon/{rel}?f=save"
LOGIN = "https://sla2.talkbank.org/logInUser"

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True, type=Path)
ap.add_argument("--plan", action="store_true", help="list sessions and total size, download nothing")
ap.add_argument("--keep-sessions", action="store_true")
ap.add_argument("--email", default=os.environ.get("TALKBANK_EMAIL"))
ap.add_argument("--password", default=os.environ.get("TALKBANK_PASSWORD"))
args = ap.parse_args()

meta = pd.read_csv(args.root / "metadata_en.csv")
missing = set((args.root / "missing_audio.txt").read_text().split())
audio_dir = args.root / "audio"
still_missing = {m for m in missing if not (audio_dir / m).exists()}
todo = meta[meta.audio_filename.isin(still_missing)].copy()
# relative path on the media server: phon/<group>/<corpus>/<...>.wav
todo["rel"] = todo.original_audio_path.str.extract(r"downloaded_corpora/phon/(.+)$")[0]
sessions = todo.groupby("rel").size().sort_values(ascending=False)
print(f"{len(still_missing):,} clips still missing, from {len(sessions):,} session recordings")
print(sessions.groupby(sessions.index.str.split('/').str[1]).agg(['count', 'sum'])
      .rename(columns={'count': 'sessions', 'sum': 'clips'}).to_string())

if not (args.email and args.password):
    sys.exit("need TALKBANK_EMAIL / TALKBANK_PASSWORD")
r = requests.post(LOGIN, json={"email": args.email, "pswd": args.password},
                  headers={"Content-type": "application/json"}, timeout=60)
r.raise_for_status()
s = requests.Session()
s.cookies.set("talkbank", r.cookies.get("talkbank"))


def locate(rel):
    """Return (url, ext) for the session; PhonBank may hold .wav, .mp3 or .mp4."""
    stem = rel.rsplit(".", 1)[0]
    for ext in ("wav", "mp3", "mp4"):
        u = MEDIA.format(rel=f"{stem}.{ext}")
        h = s.head(u, allow_redirects=True, timeout=60)
        if h.status_code == 200 and "html" not in h.headers.get("Content-Type", ""):
            return u, ext, int(h.headers.get("Content-Length") or 0)
    return None, None, 0


if args.plan:
    total = 0
    for i, rel in enumerate(sessions.index):
        u, ext, n = locate(rel)
        total += n
        if i < 10 or u is None:
            print(f"  {rel} -> {ext or 'NOT FOUND'} {n/1e6:.0f} MB")
    print(f"\nestimated download: {total/1e9:.1f} GB across {len(sessions)} sessions "
          f"(sizes of 0 mean the server did not report Content-Length)")
    sys.exit(0)

sess_dir = args.root / "sessions"
sess_dir.mkdir(exist_ok=True)
audio_dir.mkdir(exist_ok=True)
report, done = [], 0

for k, (rel, n_clips) in enumerate(sessions.items(), 1):
    url, ext, _ = locate(rel)
    if url is None:
        report.append(f"NOT FOUND\t{rel}\t{n_clips}")
        print(f"[{k}/{len(sessions)}] not on server: {rel}")
        continue
    raw = sess_dir / (rel.replace("/", "_").rsplit(".", 1)[0] + f".orig.{ext}")
    wav = raw.with_suffix(".16k.wav")
    if not wav.exists():
        if not raw.exists():
            with s.get(url, stream=True, timeout=900) as resp:
                resp.raise_for_status()
                tmp = raw.with_suffix(".part")
                with open(tmp, "wb") as f:
                    for chunk in resp.iter_content(1 << 20):
                        f.write(chunk)
                tmp.replace(raw)
        # same conversion as data_preparation/convert_audio.py
        subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(raw), "-vn", "-ac", "1", "-ar", "16000",
                        "-acodec", "pcm_s16le", "-y", str(wav)], check=True)
        raw.unlink()
    # same slicing as data_preparation/extract_segments.py::extract_audio_chunk
    ok = fail = 0
    with sf.SoundFile(wav) as af:
        sr, n = af.samplerate, len(af)
        for _, row in todo[todo.rel == rel].iterrows():
            a, b = int(row.onset / 1000 * sr), int(row.offset / 1000 * sr)
            if a >= n or b > n or b <= a:
                fail += 1
                report.append(f"OUT OF RANGE\t{row.audio_filename}")
                continue
            af.seek(a)
            sf.write(audio_dir / row.audio_filename, af.read(b - a), sr)
            ok += 1
    done += ok
    if not args.keep_sessions:
        wav.unlink()
    print(f"[{k}/{len(sessions)}] {rel}: {ok} clips" + (f", {fail} out of range" if fail else ""), flush=True)

(args.root / "fill_report.txt").write_text("\n".join(report))
left = {m for m in missing if not (audio_dir / m).exists()}
print(f"\nrecreated {done:,} clips; {len(left):,} still missing (see fill_report.txt)")
(args.root / "missing_audio.txt").write_text("\n".join(sorted(left)))
