# Evaluation scripts

*English | [日本語](README.ja.md)*

These scripts reproduce the objective metrics and statistics reported in
Table 1 of the [UtterTune paper](https://arxiv.org/abs/2508.09767): character error rate
(CER), speaker similarity, UTMOSv2 naturalness, and the significance
tests/effect sizes computed on top of those per-utterance metrics. The full
pipeline is:

1. **Synthesize + trim eval audio**: run
   `scripts/cv2/generate_eval_audio.py` once per condition (e.g. baseline
   CosyVoice 2 with no `--lora_dir`, and CosyVoice 2 + UtterTune with
   `--lora_dir` pointing at the adapter) over the released eval sentence
   sets (`data/eval/easy_50.tsv`, `data/eval/difficult_50.tsv`,
   `data/eval/katakana_50.tsv`; see `data/README.md` for column schemas).
   This script applies the paper's unconditional bidirectional VAD trim to
   every generated utterance — this is the actual eval-audio generation
   procedure used for the paper's reported numbers, and differs from the
   demo `scripts/cv2/infer.py`, whose trimming is opt-in (`--trim_out`) and
   heuristically gated.
2. Run automatic speech recognition (Whisper large-v3, decoding constrained
   to kana + a limited symbol set) on the generated audio to produce
   hypothesis transcriptions. This step is is
   upstream of `calc_cer.py`.
3. Run `calc_cer.py` / `calc_cosine_similarity_eres2net.py` /
   `calc_utmosv2.py` on the generated audio / ASR hypotheses to get
   per-utterance metrics.
4. Pivot the resulting per-utterance metrics into a wide CSV (one column
   per system/condition, one row per utterance, aligned by utterance) and
   run `compare_scores_2systems.py` (pairwise comparisons, e.g. Test Set 1)
   or `compare_scores_3systems.py` (3-way comparisons, e.g. Test Set 2) to
   get the paper's reported significance tests and effect sizes.

## Scripts

### `calc_cer.py` — Character Error Rate

Computes character-level edit distance between ASR hypotheses and reference
text (both normalized to katakana), with a heuristic trim step for common
Whisper hallucinated sentence-final tails. Example:

```bash
python -m scripts.eval.calc_cer \
    --hyp_csv whisper_hypotheses.csv \
    --ref_csv reference_texts.csv \
    --output results/cer_results.csv
```

`whisper_hypotheses.csv` needs `audio_path` and `hypothesis` columns;
`reference_texts.csv` is a headerless single-column CSV with the same row
count/order as the hypothesis CSV.

### `calc_cosine_similarity_eres2net.py` — Speaker similarity

Computes cosine similarity between speaker embeddings (via the ModelScope
`iic/speech_eres2net_sv_zh-cn_16k-common` model) of a reference (prompt) wav
and each synthesized target wav. Example:

```bash
python -m scripts.eval.calc_cosine_similarity_eres2net \
    --target_csv synthesized_wavs.csv \
    --prompts_wav_dir prompts/wav \
    --out results/speaker_sim.csv
```

Either pass `--ref_csv` (a headerless CSV of reference wav paths aligned
row-for-row with `--target_csv`) or `--prompts_wav_dir`, in which case the
reference wav for each target is derived from the target's speaker id
(assumed path layout `<cond>/<speaker>/<set>/<utt>.wav`) as
`<prompts_wav_dir>/<speaker>.wav`.

### `calc_utmosv2.py` — UTMOSv2 naturalness

Computes a UTMOSv2 MOS-style naturalness score per audio file, with
resumable checkpointing and OOM-safe batch-size backoff. Example:

```bash
python -m scripts.eval.calc_utmosv2 \
    --csv synthesized_wavs.csv \
    --out results/utmos_scores.csv \
    --batch_size 64 --tqdm
```

`synthesized_wavs.csv` is a headerless, one-path-per-line list of audio
files. Rerunning with the same `--out` resumes from the existing checkpoint.

### `compare_scores_2systems.py` — pairwise significance test + effect sizes

Compares a per-utterance metric between two systems: one-sided paired
Wilcoxon signed-rank test (direction follows the observed mean difference),
plus paired Cohen's d and paired Cliff's delta with bootstrap CIs, plus an
optional bootstrap CI on the mean paired difference. Example (Test Set 1,
UTMOSv2, baseline vs. UtterTune):

```bash
python -m scripts.eval.compare_scores_2systems \
    --input_csv utmos_easy_wide.csv \
    --systems cv2_base cv2_uttertune \
    --bootstrap --n_iter 1000 --ci 95
```

`utmos_easy_wide.csv` needs one column per system (e.g. `cv2_base`,
`cv2_uttertune`) holding that system's per-utterance score, aligned
row-for-row by utterance.

### `compare_scores_3systems.py` — 3-way comparison + multiple-comparison correction

Compares a per-utterance metric across three systems: an omnibus Friedman
test, pairwise paired Wilcoxon tests for all 3 pairs, and a Holm-Bonferroni
correction across those pairwise p-values (the step-down variant of the
Bonferroni correction the paper's Table 1 caption refers to), plus the same
effect-size/bootstrap reporting as the 2-system script. Example (Test Set
2, CER, 3-way baseline / kana-input / UtterTune):

```bash
python -m scripts.eval.compare_scores_3systems \
    --input_csv cer_difficult_wide.csv \
    --systems cv2_base cv2_base_kana cv2_uttertune \
    --bootstrap --n_iter 1000 --ci 95
```

`cer_difficult_wide.csv` needs one column per system (e.g. `cv2_base`,
`cv2_base_kana`, `cv2_uttertune`) holding that system's per-utterance CER,
aligned row-for-row by utterance.

### Module-style invocation

Running these as `python -m scripts.eval.<script>` (matching the style used
for `scripts/cv2/*` elsewhere in this repo) requires `scripts/eval/` to be an
importable package; an empty `scripts/eval/__init__.py` is included for this
reason, mirroring `scripts/cv2/__init__.py`.

## Accent correctness (not scripted)

Table 1's "Accent" column is a manual, subjective judgment made by the
paper's author while listening to synthesized samples (see the paper's
"Evaluation Metrics" section) — it is not computed by any script, and no
script is provided for it here.

## Required packages

The metric scripts (`calc_cer.py`, `calc_cosine_similarity_eres2net.py`,
`calc_utmosv2.py`) need:

- `Levenshtein`
- `jaconv`
- `pandas`
- `modelscope`
- `soundfile`
- `scipy`
- `torch`
- `utmosv2`
- `tqdm`

The statistics scripts (`compare_scores_2systems.py`,
`compare_scores_3systems.py`) additionally need:

- `numpy`
- `pandas` (already listed above)
- `scipy` (already listed above)
- `statsmodels` (`compare_scores_3systems.py` only, for Holm-Bonferroni
  correction via `statsmodels.stats.multitest.multipletests`)

All of the above are listed in the `eval` extra (`requirements-eval.txt`) at
the repo root — install with `pip install -e ".[eval]"` (see the top-level
README's setup steps).

**On version pins**: `requirements-eval.txt` gives lower bounds (`>=`), not
exact pins. These reflect versions confirmed to work together during
development, not necessarily the exact environment used for the original
arXiv v1 (Aug 2025) experiments — the eval steps ran across
several separate virtual environments whose shared-dependency versions
drifted over time (e.g. numpy 1.26–2.4, torch 2.3–2.11 across different
steps). An exact single-environment pin would overstate the precision of
this reproducibility claim, so we don't provide one; if a specific version
combination breaks for you, any of the lower-bound-or-above versions that
were exercised during development should work.
