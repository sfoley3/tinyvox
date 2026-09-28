#!/usr/bin/env python3
"""
Download the English slice of TinyVox plus everything Direction 6 needs.

What it fetches
  1. TinyVox metadata (metadata/train/val/test.csv) -- public, ~35 MB zip.
  2. The English rows of metadata.csv -> tinyvox_en/metadata_en.csv
  3. Each English corpus's "Phon and CHAT data" zip from
     talkbank.org/data/phon/<group>/<corpus>?f=zip. The media server holds audio only;
     the .cha transcripts (with the %mod target tier TinyVox's metadata does NOT keep)
     live in these zips. Corpora are located by name on the current PhonBank tree,
     since some have moved since TinyVox was built.
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


MEDIA_ROOT = "https://media.talkbank.org/phon/"


def discover_corpus_locations(session):
    """PhonBank has moved/renamed corpora since TinyVox was built, so map every
    corpus name that exists today to its current group by reading the media
    server's directory listings (group level only, two requests deep)."""
    import re
    loc = {}
    top = session.get(MEDIA_ROOT, timeout=120).text
    groups = sorted({m.rstrip("/").split("/")[-1] for m in re.findall(r'href="[^"]*?/phon/+([^"?/]+)/?"', top)})
    for g in groups:
        html = session.get(f"{MEDIA_ROOT}{g}/", timeout=120).text
        for m in re.findall(r'href="[^"]*?/phon/+' + re.escape(g) + r'/+([^"?/]+)/?"', html):
            if "." not in m:
                loc[m] = g
    return loc


def extract_corpus_zip(zip_path, corpus):
    """Unpack transcripts/<group>/<corpus>.zip into transcripts/<group>/<corpus>/.
    PhonBank zips have no top-level corpus folder, so extracting into the group
    folder merges corpora; always give each its own directory. If the zip does
    carry a single top-level folder named after the corpus, flatten it."""
    import shutil
    target = zip_path.with_suffix("")
    if target.exists() and any(target.rglob("*.cha")):
        return
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(target)
    inner = target / corpus
    if inner.is_dir() and all(p == inner for p in target.iterdir()):
        for p in inner.iterdir():
            shutil.move(str(p), str(target / p.name))
        inner.rmdir()
    n_cha = len(list(target.rglob("*.cha")))
    n_xml = len(list(target.rglob("*.xml")))
    print(f"[cha]   {target}: {n_cha} .cha, {n_xml} .xml")


def fetch_transcripts(en, summary, out, cookie):
    """Download each corpus's 'Phon and CHAT data' zip (talkbank.org/data/phon/<group>/<corpus>?f=zip)
    and unpack it under transcripts/<group>/<corpus>/. Corpora are located by name on the
    current PhonBank tree, because the group recorded in TinyVox's metadata may be stale."""
    tdir = out / "transcripts"
    s = requests.Session()
    s.cookies.set("talkbank", cookie)
    loc = discover_corpus_locations(s)
    print(f"[cha] PhonBank currently lists {len(loc)} corpora across {len(set(loc.values()))} groups")
    not_found = []
    for _, row in summary.iterrows():
        group = loc.get(row.corpus, row.group)
        if row.corpus not in loc:
            print(f"[cha] {row.group}/{row.corpus}: not in the current PhonBank tree; trying old path")
        dest = tdir / group / f"{row.corpus}.zip"
        if dest.exists() and dest.stat().st_size > 10_000:
            print(f"[cha] have {dest}")
            extract_corpus_zip(dest, row.corpus)
            continue
        url = DATA_ZIP.format(group=group, corpus=row.corpus)
        r = s.get(url, stream=True, timeout=900)
        if r.status_code != 200 or "zip" not in r.headers.get("Content-Type", ""):
            not_found.append((row.group, row.corpus, group, r.status_code))
            print(f"[cha]   FAILED {url} (HTTP {r.status_code}, {r.headers.get('Content-Type')})")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
        extract_corpus_zip(dest, row.corpus)
        print(f"[cha]   ok {url} -> {dest.stat().st_size/1e6:.1f} MB")
    if not_found:
        (out / "failed_cha.txt").write_text("\n".join("\t".join(map(str, f)) for f in not_found))
        names = ", ".join(f[1] for f in not_found)
        print(f"[cha] could not locate: {names}. These corpora were probably renamed or merged on PhonBank;")
        print("[cha] see failed_cha.txt, and compare the corpus list printed above with the TinyVox names.")
        candidates = sorted(loc)
        print("[cha] corpora available now: " + ", ".join(f"{c} ({loc[c]})" for c in candidates))


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
        fetch_transcripts(en, summary, args.out, cookie)

    if args.audio == "full":
        audio_full(args.out, en, cookie)
    elif args.audio == "remote":
        audio_remote(args.out, en, cookie)

    print("\nDone. Next: python direction6/build_tinyvox_en_targets.py --root", args.out)


if __name__ == "__main__":
    main()
