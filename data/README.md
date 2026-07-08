# UtterTune Data Release

*English | [日本語](README.ja.md)*

This directory contains the training manifest and evaluation sentence sets used
in the [UtterTune paper](https://arxiv.org/abs/2508.09767). No audio or audio-derived features (waveforms, speech
tokens) are included — see the LICENSE section below.

## `train/jsutjvs_manifest.tsv`

Training manifest text with provenance identifiers, derived from
`data/manifests/all.tsv` in the source training repository. The original
manifest's speech-token (`.npy`) and waveform (`.wav`) path columns have been
removed, since those files are not redistributable (see LICENSE section).

Columns (tab-separated, header row included):

| column   | description |
|----------|-------------|
| `text`   | Sentence text with `<PHON_START>...<PHON_END>` annotation spans. An apostrophe (`'`) marks the accent nucleus and a slash (`/`) marks a phrase boundary within a tagged span. |
| `corpus` | `jsut` or `jvs` |
| `subset` | `basic5000` or `voiceactress100` for JSUT; `parallel100` for JVS |
| `speaker`| Empty for JSUT; `jvsNNN` (e.g. `jvs001`) for JVS |
| `file_id`| Source utterance id, e.g. `BASIC5000_0001`, `VOICEACTRESS100_001` |

Row count: 15,097 (5,000 JSUT basic5000 + 100 JSUT voiceactress100 + 9,997
JVS parallel100).

Given `corpus`, `subset`, `speaker`, and `file_id`, and a legitimately
obtained local copy of JSUT and/or JVS audio, the corresponding waveform can
be reconstructed, e.g.:

- JSUT: `data/jsut/wav/{file_id}.wav`
- JVS: `data/jvs/{speaker}/wav/{file_id}.wav`

**No train/validation split is encoded in this manifest.**
`scripts/cv2/train.py` performs a random split at training time via
`torch.utils.data.random_split`, seeded by `training.seed` in the training
config (default seed 42, `val_ratio` 0.05), so the same manifest plus the
same training config reproduces the same split.

## `eval/`

Evaluation sentence sets used for the paper's listening tests (Table 1 and
§5.2). Each file has a header row and 50 data rows (`id` 1..50, 1-indexed),
one row per underlying sentence, joining across the text variants used for
different conditions.

### `eval/easy_50.tsv` — Test Set 1 (general naturalness, §5.1)

| column        | description |
|---------------|-------------|
| `id`          | 1..50 |
| `vanilla`     | Plain sentence text — baseline condition |
| `phon_tagged` | Same sentence with one word wrapped in `<PHON_START>...<PHON_END>` — proposed method (UtterTune) condition |

### `eval/difficult_50.tsv` — Test Set 2 (pronunciation/accent stress test, §5.1)

Each row has one difficult word per sentence.

| column          | description |
|-----------------|-------------|
| `id`            | 1..50 |
| `vanilla`       | Plain sentence text with the difficult word in kanji/standard orthography — baseline condition |
| `kana_baseline` | Same sentence with the difficult word replaced by its kana-only reading, no accent marks — kana-input baseline condition |
| `phon_tagged`   | Same sentence with the difficult word's kana reading plus accent-nucleus (`'`) and phrase-boundary (`/`) marks, wrapped in `<PHON_START>...<PHON_END>` — proposed method (UtterTune) condition |

### `eval/katakana_50.tsv` — Katakana leak test (§5.2)

50 additional sentences, each containing one katakana word with an accented
nucleus, used to confirm the LoRA does not affect untagged spans. This set
uses the same plain text for both the baseline and proposed-method
conditions (no tagged variant exists, by design of this test).

| column | description |
|--------|-------------|
| `id`   | 1..50 |
| `text` | Plain sentence text |

## LICENSE

The training manifest text is derived from the [JSUT corpus](https://sites.google.com/site/shinnosuketakamichi/publication/jsut) and [JVS
corpus](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus).

JSUT's own LICENCE.txt states text licensing varies by subset: `basic5000`
text is a mix of Wikipedia ([CC-BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/deed)), the Tanaka Corpus ([CC-BY 2.0](https://creativecommons.org/licenses/by/2.0/deed)), and
JSUT's own original sentences ([CC-BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/deed)); `voiceactress100` text is from
the Voice Actress Corpus, CC-BY-SA 4.0. JVS text is derived from JSUT and
follows JSUT's licensing.

No audio or audio-derived data (waveforms, speech tokens) from JSUT/JVS is
redistributed here, per JSUT's terms ("Re-distribution [of audio] is not
permitted").

The pronunciation/accent annotations (`<PHON_START>`/`<PHON_END>` tags,
accent-nucleus/phrase-boundary marks) added by this project on top of the
original text are original annotation work. Because the underlying text
carries CC-BY-SA obligations for a portion of the corpus, this annotated
manifest as a whole is released under **CC BY-SA 4.0**, matching JSUT's own
disclosure style (per-subset license note) and satisfying the ShareAlike
terms of the CC-BY-SA-licensed portions (CC-BY-SA 3.0 and CC-BY-SA 4.0 are on
Creative Commons' official compatible-license list for relicensing adapted
material; the CC-BY 2.0 portion imposes no ShareAlike restriction so is
compatible by default).

Users of this manifest/eval data must still independently obtain JSUT and JVS
audio from the official sources under those corpora's own terms if they want
to reconstruct training audio; only text + accent annotations + provenance
identifiers are provided here.

The eval sentence sets (`easy_50`, `difficult_50`, `katakana_50`) were
authored by the paper's author using ChatGPT o3-assisted generation followed
by manual review (per the paper's Evaluation Data section) and are original
content by the author, released under CC BY-SA 4.0 as well for consistency.
