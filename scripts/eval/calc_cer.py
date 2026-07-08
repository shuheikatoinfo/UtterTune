#!/usr/bin/env python3
"""Character Error Rate (CER) between ASR hypotheses and reference text.

This is the exact script used to produce the CER column of Table 1 in the
UtterTune paper (arXiv:2508.09767). It only performs the text-diff step: the
hypothesis transcriptions are expected to have already been produced upstream
by Whisper large-v3 with decoding constrained to kana + a limited set of
symbols (per the paper's CER methodology). This script does not run ASR
itself.

Inputs:
    --hyp_csv   CSV with header, containing at least an `audio_path` column
                and a `hypothesis` column (ASR output text per utterance).
    --ref_csv   Headerless, single-column CSV of reference texts. Row order
                and row count must exactly match --hyp_csv (rows are paired
                positionally, not by any shared key).

Normalization:
    Both hypothesis and reference text are converted hiragana -> katakana
    (via jaconv.hira2kata) before diffing, since the two sources may use
    different kana forms for the same reading and we only want to measure
    reading-level errors, not kana-script differences.

Hypothesis trimming (--trim):
    Whisper sometimes hallucinates spurious trailing continuations after the
    actual utterance ends (e.g. appending "ました" / "ません" / other
    sentence-final forms that were never spoken). The --trim list is a set of
    common hallucinated tail fragments; each one is stripped, along with
    everything after it, from the end of the hypothesis using a
    "trailing word + trim string" regex. This is a heuristic tuned for the
    paper's eval set and is intentionally left unchanged here.

CER formula:
    Character-level Levenshtein edit distance between (normalized) hypothesis
    and reference, divided by reference character length.

Output:
    A CSV (--output) with the original columns plus `cer_distance` and `cer`
    per row.
"""

import argparse
import re
from pathlib import Path

import pandas as pd
import Levenshtein
from jaconv import hira2kata

def compute_cer(reference: str, hypothesis: str):
    dist = Levenshtein.distance(reference, hypothesis)
    cer  = dist / len(reference) if reference else float("nan")
    return {"cer_distance": dist, "cer": cer}

def main():
    parser = argparse.ArgumentParser(description="CER calculation")
    parser.add_argument("--hyp_csv", required=True, help="audio_path,hypothesis CSV")
    parser.add_argument("--ref_csv", required=True, help="reference CSV (no header)")
    parser.add_argument("--trim", nargs="*", type=str, default=["わけ", "ません", "ます", "ましたよ", "ました", "のもの", "のころ", "どもの", "ところ", "どうか", "ですよね", "ですよ", "ですね", "ですが", "です", "でした", "そのまま", "しょうか", "このまま", "あのー"])
    parser.add_argument("--output", type=Path, default="cer_results.csv", help="output CSV")
    args = parser.parse_args()

    df_hyp = pd.read_csv(args.hyp_csv)
    df_ref = pd.read_csv(args.ref_csv, names=["reference"])
    df = pd.concat([df_hyp, df_ref], axis=1)

    # Normalize hypothesis
    hyp_out: list[str] = []
    for text in df["hypothesis"].fillna("").astype(str).tolist():
        text = hira2kata(text)
        text = re.sub("[、。「」『』・�]", "", text)

        if args.trim is not None:
            # Paper-era code removed whitespace inside the trim loop; that's idempotent,
            # so doing it once up-front preserves results.
            text = re.sub(r"\s", "", text)
            for trim_chars in args.trim:
                text = re.sub(rf"^\w*{hira2kata(trim_chars)}", "", text)
            hyp_out.append(text)
        else:
            hyp_out.append(re.sub(r"\s", "", text))

    df["hypothesis"] = hyp_out

    # Normalize reference
    ref_out: list[str] = []
    for text in df["reference"].fillna("").astype(str).tolist():
        text = hira2kata(text)
        ref_out.append(re.sub("[、。]", "", text))
    df["reference"] = ref_out

    # Compute CER per example
    cer_info = [compute_cer(ref, hyp) for ref, hyp in zip(df["reference"], df["hypothesis"])]
    df["cer_distance"] = [x["cer_distance"] for x in cer_info]
    df["cer"] = [x["cer"] for x in cer_info]

    # 4) Save
    args.output.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Done! Results in {args.output}")

if __name__ == "__main__":
    main()
