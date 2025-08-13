#!/usr/bin/env python3
"""
Join JSUT/JVS transcripts with token paths to final 4-col TSV.
Columns: spk_id <TAB> text <TAB> token.npy <TAB> wav_path
"""

import argparse
import csv
from logging import getLogger, StreamHandler, INFO
from pathlib import Path

logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(INFO)
logger.setLevel(INFO)
logger.addHandler(handler)
logger.propagate = False

ap = argparse.ArgumentParser()
ap.add_argument(
    "--corpus_root",
    type=Path,
    default=Path("data/corpora"),
    help="folder containing jsut/ and jvs/",
)
ap.add_argument(
    "--token_root",
    type=Path,
    default=Path("data/speech_tokens"),
    help="root created by extract_tokens.py",
)
ap.add_argument("--out", type=Path, default=Path("data/manifests/all.tsv"))
args = ap.parse_args()

rows = []

# JSUT
txt = args.corpus_root / "jsut" / "trans.txt"

with txt.open(encoding="utf-8") as f:
    for line in f:
        uid, trans = line.rstrip().split(":")
        tok = args.token_root / "jsut" / f"{uid}.npy"
        wav = args.corpus_root / "jsut" / "wav" / f"{uid}.wav"
        rows.append([0, trans, tok, wav])

# JVS
for spk_id, spkdir in enumerate(sorted((args.corpus_root / "jvs").glob("jvs*")), 1):
    txt = spkdir / "trans.txt"

    with txt.open(encoding="utf-8") as f:
        for line in f:
            uid, trans = line.rstrip().split(":")
            tok = args.token_root / "jvs" / spkdir.name / f"{uid}.npy"
            wav = spkdir / "wav" / f"{uid}.wav"
            rows.append([spk_id, trans, tok, wav])

args.out.parent.mkdir(parents=True, exist_ok=True)

with args.out.open("w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f, delimiter="	")
    writer.writerows(rows)

print("✓ manifest", args.out, "lines", len(rows))
