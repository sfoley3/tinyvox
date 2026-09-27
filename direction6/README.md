# TinyVox-EN for Direction 6 — getting started

Two scripts, run from the repo root (they import `chat_toolkit` from the parent folder).

```bash
pip install requests pandas remotezip        # remotezip only for --audio remote
export TALKBANK_EMAIL=you@usc.edu TALKBANK_PASSWORD='...'

# 1) metadata (public) + English subset + the 12 PhonBank transcript zips (auth)
python direction6/download_tinyvox_en.py --out /scratch/tinyvox_en --audio skip

# 2) recover the %mod target tier and join it to every TinyVox-EN utterance
python direction6/build_tinyvox_en_targets.py --root /scratch/tinyvox_en

# 3) audio: try range requests first (English members only), fall back to the full zip
python direction6/download_tinyvox_en.py --out /scratch/tinyvox_en --audio remote
python direction6/download_tinyvox_en.py --out /scratch/tinyvox_en --audio full     # resumable curl, then extracts EN wavs
```

## What the English slice is (from the public metadata, checked 2026-09-27)

292,574 utterances, 203 children, 208 h, 1,407 session files, ages 6–96 months, 12 source corpora:

| corpus | utts | children | age (mo) | activity | note for Direction 6 |
|---|---|---|---|---|---|
| Eng-NA/Providence | 134,424 | 6 | 11–48 | toyplay | toddler bulk; longitudinal |
| Eng-NA/Davis | 68,518 | 17 | 6–36 | toyplay | babbling → first words |
| Biling/ChildL2 | 36,863 | 2 | 47–75 | toyplay | L2 English ⚠️ exclude from US norms |
| Clinical/Cummings | 22,101 | 21 | 37–88 | tests | SSD/therapy ⚠️ diagnosis per child |
| Eng-NA/Goad | 8,840 | 2 | 18–43 | toyplay | |
| Clinical/TorringtonEaton | 7,626 | 53 | 49–72 | tests | picture naming; check TD vs SSD |
| Clinical/Preston | 5,481 | 44 | 48–69 | tests | SSD therapy (RSSD) ⚠️ |
| Clinical/McAllister | 4,201 | 1 | 46–51 | pictures | single child, SSD ⚠️ |
| Eng-NA/Penney | 1,297 | 26 | 60–75 | narrative, pictures | 5–6 y TD |
| Clinical/Chiat | 1,230 | 4 | 60–68 | interview | UK? check dialect |
| Biling/Seine-Marne | 1,200 | 26 | 59–96 | tests | bilingual ⚠️ |
| Eng-NA/Menn | 793 | 1 | 12–16 | toyplay | |

Age bins (utterances): 0–12 mo 9.7K · 12–24 97.7K · 24–36 81.6K · 36–48 36.3K · 48–60 35.7K · 60–72 23.1K · 72–96 8.5K.

So the 2–6 y window that matters most for the norms comes mostly from Providence (to 4;0), Cummings, TorringtonEaton and Preston — and three of those are clinical corpora. **The TD-only norm pool inside TinyVox is thinner than the headline numbers suggest**; it is Providence + Davis + Goad + Menn + Penney + the TD children in TorringtonEaton. Confirm per-child diagnosis from the Phon session metadata / corpus docs before building curves (control C9).

## What TinyVox's metadata does and does not give you

- `phones` = the child's **actual** production, normalised to 57 phones, diacritics stripped, **word boundaries kept as `|`**. Good.
- No target tier. `sentence` is orthography, not %mod. Direction 6 needs %mod (and ideally Phon's explicit target↔actual alignment), which is why step 2 exists. The transcript zips are the "Phon and CHAT data" bundles from each corpus page (`https://talkbank.org/data/phon/<group>/<corpus>?f=zip`); they also contain Phon `.xml` sessions, which carry the alignment tier — parse those if CHAT `%mod` coverage turns out low for a corpus.
- `onset`/`offset` are ms in the original session file and are what `build_tinyvox_en_targets.py` uses to join each wav back to its `.cha` utterance (exact → ±50 ms → nearest).

`coverage_report.txt` tells you, per corpus, what fraction of utterances have a `%mod` tier and how often the `%pho`/`%mod` word counts agree (the cheap precondition for word-level target↔actual pairing). That number decides which corpora can enter E1 and the toddler curves.

## Audio size

The TinyVox audio zip is all five languages (561,312 wavs, ~388 h at 16 kHz mono ≈ 45 GB uncompressed; the server does not report the zip size). The English share is ~24 GB. `--audio remote` pulls only English members if the media server honours HTTP Range requests; if it errors, use `--audio full` (resumable) and the script extracts only the English wavs from the local zip. The 238 GB "original" session audio is not needed for Direction 6 unless you want BabAR-style 20 s context windows.

## Next after this

1. Check `coverage_report.txt`; if `%mod` is missing for Providence/Davis, parse the Phon `.xml` sessions in `transcripts/…` for the target tier.
2. Encode Crowe & McLeod 2020 as CSV (to-do in the Direction 6 doc).
3. P5 (covert-contrast ABX) can start from `targets_en.csv` alone: rows where a `%mod` word contains /s/ and the aligned `%pho` word has [t] in that slot.
