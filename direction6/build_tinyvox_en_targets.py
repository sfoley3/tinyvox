#!/usr/bin/env python3
"""
Join TinyVox-EN utterances back to their PhonBank .cha transcripts to recover the
TARGET tier (%mod / %xmod) next to the ACTUAL tier (%pho / %xpho) that TinyVox kept.

Direction 6 needs the target because the score is target-conditioned, and because
aligning on the actual transcription (then mapping back to the target slot) is how
we sidestep forced-alignment failure on toddlers.

Inputs (from download_tinyvox_en.py)
  <root>/metadata_en.csv
  <root>/transcripts/<group>/<corpus>/**/*.cha

Outputs
  <root>/targets_en.csv      one row per TinyVox-EN utterance, with
                             pho_raw, mod_raw, xpho_raw, xmod_raw, n_words_pho, n_words_mod,
                             words_pho|words_mod (pipe-joined, when word counts match),
                             match_method, plus the original TinyVox columns
  <root>/coverage_report.txt per-corpus coverage of %mod and word-count agreement

Usage
  python direction6/build_tinyvox_en_targets.py --root /path/to/tinyvox_en
"""
import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True, type=Path)
ap.add_argument("--tinyvox-repo", type=Path, default=Path(__file__).resolve().parent.parent,
                help="repo root (for chat_toolkit); defaults to the parent of this folder")
ap.add_argument("--tolerance-ms", type=int, default=50, help="onset/offset slack when matching utterances")
args = ap.parse_args()

sys.path.insert(0, str(args.tinyvox_repo))
from chat_toolkit.parser import parse_chat_file  # noqa: E402
from chat_toolkit.cha import Utterance  # noqa: E402

meta = pd.read_csv(args.root / "metadata_en.csv")
meta["cha_stem"] = meta.cha_relpath.str.replace(r"\.cha$", "", regex=True)

# ---- index every .cha we downloaded, keyed by (group, corpus, relpath-without-ext) ----
cha_index = {}
tdir = args.root / "transcripts"
for group_dir in tdir.iterdir():
    if not group_dir.is_dir():
        continue
    for corpus_dir in group_dir.iterdir():
        if not corpus_dir.is_dir():
            continue
        for cha in corpus_dir.rglob("*.cha"):
            rel = cha.relative_to(corpus_dir).with_suffix("")
            cha_index[(group_dir.name, corpus_dir.name, str(rel))] = cha
            # also index by bare stem, in case the zip layout differs from the scraper's
            cha_index.setdefault((group_dir.name, corpus_dir.name, "stem:" + cha.stem), cha)
print(f"indexed {len([k for k in cha_index if not k[2].startswith('stem:')])} .cha files")

TIERS = ["pho", "xpho", "mod", "xmod"]


def split_words(tier):
    """Phon's CHAT export separates words with spaces (syllables may carry '.' or '‸').
    Return the list of word strings, dropping pause/empty tokens."""
    if not isinstance(tier, str) or not tier.strip():
        return []
    toks = [t for t in re.split(r"\s+", tier.strip()) if t and t not in {"(.)", "(..)", "(...)", "‡", "„"}]
    return toks


rows, methods = [], defaultdict(int)
missing_cha = set()
parsed_cache = {}

for (group, corpus, stem), g in meta.groupby(["group", "corpus", "cha_stem"]):
    cha = cha_index.get((group, corpus, stem)) or cha_index.get((group, corpus, "stem:" + Path(stem).name))
    if cha is None:
        missing_cha.add(f"{group}/{corpus}/{stem}.cha")
        for _, r in g.iterrows():
            rows.append({**r.to_dict(), "match_method": "no_cha"})
        continue
    if cha not in parsed_cache:
        try:
            parsed_cache[cha] = parse_chat_file(cha)
        except Exception as e:  # keep going; report
            parsed_cache[cha] = None
            print(f"parse error {cha}: {e}")
    cf = parsed_cache[cha]
    if cf is None:
        for _, r in g.iterrows():
            rows.append({**r.to_dict(), "match_method": "parse_error"})
        continue

    # KCHI = Target_Child utterances with timestamps
    kchi = []
    for u in cf.utterances:
        role = cf.participants.get(u.speaker, {}).get("role", "")
        if role == "Target_Child" and isinstance(u, Utterance):
            kchi.append(u)
    by_onset = sorted(kchi, key=lambda u: u.onset)

    for _, r in g.iterrows():
        # 1) exact timestamps, 2) within tolerance, 3) nearest onset
        cand = [u for u in kchi if u.onset == r.onset and u.offset == r.offset]
        method = "exact"
        if not cand:
            cand = [u for u in kchi if abs(u.onset - r.onset) <= args.tolerance_ms
                    and abs(u.offset - r.offset) <= args.tolerance_ms]
            method = "tolerance"
        if not cand and by_onset:
            u = min(by_onset, key=lambda u: abs(u.onset - r.onset))
            cand, method = ([u] if abs(u.onset - r.onset) <= 1000 else []), "nearest"
        if not cand:
            rows.append({**r.to_dict(), "match_method": "unmatched"})
            methods["unmatched"] += 1
            continue
        u = cand[0]
        methods[method] += 1
        t = {k: u.dependent_tiers.get(k) for k in TIERS}
        pho, mod = t["pho"] or t["xpho"], t["mod"] or t["xmod"]
        wp, wm = split_words(pho), split_words(mod)
        rows.append({
            **r.to_dict(),
            "match_method": method,
            "cha_file": str(cha),
            "utt_content": u.content,
            "pho_raw": t["pho"], "xpho_raw": t["xpho"], "mod_raw": t["mod"], "xmod_raw": t["xmod"],
            "n_words_pho": len(wp), "n_words_mod": len(wm),
            "words_pho": "|".join(wp), "words_mod": "|".join(wm),
            "word_count_match": (len(wp) == len(wm)) and len(wm) > 0,
        })

out = pd.DataFrame(rows)
out.to_csv(args.root / "targets_en.csv", index=False)

# ---- coverage report ----
lines = [f"TinyVox-EN utterances: {len(out):,}", f"match methods: {dict(methods)}",
         f"session files not found: {len(missing_cha)}"]
if missing_cha:
    (args.root / "missing_cha.txt").write_text("\n".join(sorted(missing_cha)))
    lines.append("  (listed in missing_cha.txt)")
lines.append("")
lines.append(f"{'corpus':28s} {'utts':>7s} {'has_mod':>8s} {'has_pho':>8s} {'wc_match':>9s} {'children':>8s}")
for (grp, corp), g in out.groupby(["group", "corpus"]):
    has_mod = g.mod_raw.notna().mean() if "mod_raw" in g else 0
    has_pho = g.pho_raw.notna().mean() if "pho_raw" in g else 0
    wc = g.word_count_match.fillna(False).mean() if "word_count_match" in g else 0
    lines.append(f"{grp+'/'+corp:28s} {len(g):7d} {has_mod:8.2f} {has_pho:8.2f} {wc:9.2f} {g.child_pseudoid.nunique():8d}")
lines.append("")
lines.append("Read has_mod as: fraction of utterances with a %mod/%xmod target tier. Corpora near 0 can't be")
lines.append("used for target-conditioned scoring from CHAT alone -- check the Phon .xml sessions in the same")
lines.append("zip for a target tier and the explicit alignment before giving up on them.")
report = "\n".join(lines)
(args.root / "coverage_report.txt").write_text(report)
print(report)
