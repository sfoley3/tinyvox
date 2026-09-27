#!/usr/bin/env python3
"""
Download the English slice of TinyVox plus everything Direction 6 needs.

What it fetches
  1. TinyVox metadata (metadata/train/val/test.csv) -- public, ~35 MB zip.
  2. The English rows of metadata.csv -> tinyvox_en/metadata_en.csv
  3. PhonBank "Phon and CHAT data" zips for every corpus that contributes English
     utterances (12 corpora, small). These contain the .cha transcripts with the
     %mod (target) tier that TinyVox's own metadata does NOT carry, and usually the
     Phon .xml sessions with the explicit target<->actual alignment.
  4. The TinyVox one-item audio (561,312 wavs, all languages) -- then keeps only the
     English wavs. Two modes:
       --audio full    : download TinyVox.zip once (resumable), extract English members.
       --audio remote  : use HTTP range requests to pull only English members
                         (needs `pip install remotezip`; only works if the server
                         honours Range requests -- try it first, fall back to full).
       --audio skip    : metadata + transcripts only (default).

Auth: PhonBank needs a TalkBank account. Pass --email/--password or set
TALKBANK_EMAIL / TALKBANK_PASSWORD. Login endpoint is the same one the tinyvox
repo's talkbank_audio_scrapper.py uses.

Usage
  python direction6/download_tinyvox_en.py --out /path/to/tinyvox_en --audio skip
  python direction6/download_tinyvox_en.py --out /path/to/tinyvox_en --audio remote
  python direction6/download_tinyvox_en.py --out /path/to/tinyvox_en --audio full
"""
import argparse
import io
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests

META_ZIP = "https://talkbank.org/phon/access/Derived/0docs/BaBAR.zip"
AUDIO_ZIP = "https://media.talkbank.org/phon/0extra/TinyVox.zip"
LOGIN = "https://sla2.talkbank.org/logInUser"
DATA_ZIP = "https://talkbank.org/data/phon/{group}/{corpus}?f=zip"


def login(email, password):
    r = requests.post(LOGIN, json={"email": email, "pswd": password},
                      headers={"Content-type": "application/json"}, timeout=60)
    r.raise_for_status()
    cookie = r.cookies.get("talkbank")
    if not cookie:
        sys.exit("Login succeeded but no 'talkbank' cookie returned - check credentials.")
    return cookie


def fetch_metadata(out):
    meta_dir = out / "metadata"
    meta_csv = meta_dir / "metadata.csv"
    if meta_csv.exists():
        print(f"[meta] already have {meta_csv}")
        return meta_csv
    print("[meta] downloading BaBAR.zip (metadata, ~35 MB)")
    r = requests.get(META_ZIP, timeout=600)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    meta_dir.mkdir(parents=True, exist_ok=True)
    for name in z.namelist():
        if name.endswith(".csv") and "__MACOSX" not in name:
            (meta_dir / Path(name).name).write_bytes(z.read(name))
            print(f"[meta]   wrote {Path(name).name}")
    return meta_csv


def english_subset(meta_csv, out):
    m = pd.read_csv(meta_csv)
    en = m[m.language.str.lower().str.startswith("eng")].copy()
    # e.g. .../downloaded_corpora/phon/Eng-NA/Davis/Martin/011000.cha -> Eng-NA/Davis, Martin/011000.cha
    parts = en.original_transcript_path.str.extract(
        r"downloaded_corpora/phon/([^/]+)/([^/]+)/(.+)$")
    en["group"], en["corpus"], en["cha_relpath"] = parts[0], parts[1], parts[2]
    en["split"] = "unassigned"
    for split in ["train", "val", "test"]:
        p = meta_csv.parent / f"{split}.csv"
        if p.exists():
            names = set(pd.read_csv(p, usecols=["audio_filename"]).audio_filename)
            en.loc[en.audio_filename.isin(names), "split"] = split
    out_csv = out / "metadata_en.csv"
    en.to_csv(out_csv, index=False)
    hours = (en.offset - en.onset).sum() / 3.6e6
    print(f"[meta] English: {len(en):,} utterances, {en.child_pseudoid.nunique()} children, "
          f"{hours:.1f} h, {en.original_transcript_path.nunique()} session files")
    summary = (en.groupby(["group", "corpus"])
                 .agg(utts=("audio_filename", "size"), children=("child_pseudoid", "nunique"),
                      age_min=("age_months", "min"), age_max=("age_months", "max"))
                 .reset_index())
    summary.to_csv(out / "corpora_en.csv", index=False)
    print(summary.to_string(index=False))
    return en, summary


def fetch_transcripts(summary, out, cookie):
    tdir = out / "transcripts"
    for _, row in summary.iterrows():
        dest = tdir / row.group / f"{row.corpus}.zip"
        if dest.exists() and dest.stat().st_size > 10_000:
            print(f"[cha] have {dest}")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = DATA_ZIP.format(group=row.group, corpus=row.corpus)
        print(f"[cha] {url}")
        with requests.get(url, cookies={"talkbank": cookie}, stream=True, timeout=600) as r:
            r.raise_for_status()
            ctype = r.headers.get("Content-Type", "")
            if "html" in ctype:
                sys.exit(f"Got HTML instead of a zip for {url}: not logged in, or no access to this corpus.")
            with open(dest, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        # unzip next to it
        with zipfile.ZipFile(dest) as z:
            z.extractall(dest.parent)
        n_cha = len(list((dest.parent / row.corpus).rglob("*.cha"))) if (dest.parent / row.corpus).exists() else "?"
        print(f"[cha]   ok ({dest.stat().st_size/1e6:.1f} MB, {n_cha} .cha files)")


def wanted_audio_names(en):
    return set(en.audio_filename)


def extract_english_from_local_zip(zip_path, wanted, audio_dir):
    audio_dir.mkdir(parents=True, exist_ok=True)
    done = 0
    with zipfile.ZipFile(zip_path) as z:
        members = {Path(n).name: n for n in z.namelist() if n.endswith(".wav")}
        missing = wanted - set(members)
        for i, name in enumerate(sorted(wanted & set(members))):
            target = audio_dir / name
            if target.exists():
                continue
            with z.open(members[name]) as src, open(target, "wb") as dst:
                dst.write(src.read())
            done += 1
            if done % 5000 == 0:
                print(f"[audio]   extracted {done:,}")
    print(f"[audio] extracted {done:,} new files; {len(missing)} wanted names not in zip")
    if missing:
        (audio_dir.parent / "missing_audio.txt").write_text("\n".join(sorted(missing)))


def audio_full(out, en, cookie):
    zip_path = out / "TinyVox.zip"
    print("[audio] full download of TinyVox.zip (all languages) with resume; this is large.")
    cmd = ["curl", "-fL", "-C", "-", "--retry", "20", "--retry-delay", "15",
           "-H", f"Cookie: talkbank={cookie}", "-o", str(zip_path), AUDIO_ZIP]
    subprocess.run(cmd, check=True)
    extract_english_from_local_zip(zip_path, wanted_audio_names(en), out / "audio")


def audio_remote(out, en, cookie):
    try:
        from remotezip import RemoteZip
    except ImportError:
        sys.exit("pip install remotezip   (or use --audio full)")
    audio_dir = out / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    wanted = wanted_audio_names(en)
    print("[audio] opening TinyVox.zip remotely via HTTP Range requests")
    with RemoteZip(AUDIO_ZIP, headers={"Cookie": f"talkbank={cookie}"}) as z:
        members = {Path(n).name: n for n in z.namelist() if n.endswith(".wav")}
        todo = sorted(wanted & set(members))
        print(f"[audio] {len(todo):,} English members to fetch; {len(wanted - set(members))} not in zip")
        for i, name in enumerate(todo):
            target = audio_dir / name
            if target.exists():
                continue
            target.write_bytes(z.read(members[name]))
            if (i + 1) % 2000 == 0:
                print(f"[audio]   {i+1:,}/{len(todo):,}")
    print("[audio] done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--audio", choices=["skip", "full", "remote"], default="skip")
    ap.add_argument("--email", default=os.environ.get("TALKBANK_EMAIL"))
    ap.add_argument("--password", default=os.environ.get("TALKBANK_PASSWORD"))
    ap.add_argument("--no-transcripts", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    meta_csv = fetch_metadata(args.out)
    en, summary = english_subset(meta_csv, args.out)

    need_auth = (not args.no_transcripts) or args.audio != "skip"
    cookie = None
    if need_auth:
        if not (args.email and args.password):
            sys.exit("Need --email/--password (or TALKBANK_EMAIL/TALKBANK_PASSWORD) for PhonBank downloads.")
        cookie = login(args.email, args.password)
        print("[auth] logged in")

    if not args.no_transcripts:
        fetch_transcripts(summary, args.out, cookie)

    if args.audio == "full":
        audio_full(args.out, en, cookie)
    elif args.audio == "remote":
        audio_remote(args.out, en, cookie)

    print("\nDone. Next: python direction6/build_tinyvox_en_targets.py --root", args.out)


if __name__ == "__main__":
    main()
