#!/usr/bin/env python3
"""Generate evaluation wavs reproducing the UtterTune paper's Table 1 audio.

This reproduces the eval-audio generation procedure used for the paper
(arXiv:2508.09767), in particular the VAD trimming step: the paper states
that VAD-based silence trimming (`torchaudio.functional.vad`, applied
forward then reversed, i.e. bidirectional) was applied UNCONDITIONALLY to
every synthesized utterance before scoring. This is different from
`scripts/cv2/infer.py`'s demo `--trim_out` flag, which is opt-in (off by
default) and additionally gated behind a length heuristic. This script does
not have that opt-in gate: every synthesized utterance is trimmed and saved.

For each speaker prompt (wav + transcript) under `--prompts_wav_dir` /
`--prompts_trans_dir`, this script does zero-shot CosyVoice 2 synthesis
(optionally with a LoRA adapter, e.g. UtterTune) over every sentence in one
of the released eval sentence sets (`data/eval/easy_50.tsv`,
`data/eval/difficult_50.tsv`, `data/eval/katakana_50.tsv`; see
`data/README.md` for column schemas), for a single named condition/text
column, then VAD-trims and saves each output wav.

Output layout (consistent with
`scripts/eval/calc_cosine_similarity_eres2net.py`'s `--prompts_wav_dir`
speaker-id derivation, `Path(target_wav).parents[1].name`):

    <out_root>/<condition>/<speaker_id>/<eval_set>/<id:03d>.wav

Example invocation (proposed-method condition, Test Set 1):

    python -m scripts.cv2.generate_eval_audio \\
        --base_model pretrained_models/CosyVoice2-0.5B \\
        --lora_dir lora_weights/UtterTune-CosyVoice2-ja-JSUTJVS \\
        --eval_set easy --condition phon_tagged \\
        --prompts_wav_dir prompts/wav --prompts_trans_dir prompts/trans \\
        --out_root eval_audio

Run once per condition (e.g. `vanilla` with no `--lora_dir`, and
`phon_tagged` with `--lora_dir` pointing at the UtterTune adapter) to
produce the paired data consumed by `scripts/eval/calc_cer.py`,
`calc_cosine_similarity_eres2net.py`, `calc_utmosv2.py`, and finally
`compare_scores_2systems.py` / `compare_scores_3systems.py`.

Prompt speakers: this repo ships 3 sample prompt speakers under
`prompts/wav` and `prompts/trans` for demonstration. The paper selected 48
ReazonSpeech v2.0 clips (one speaker each, WADA-SNR >= 25 dB, <= 4s) as
candidate prompts and, after VAD-based speech-presence filtering, used the
42 that had detected speech segments as the actual eval prompt set, with the
same `<speaker_id>.wav` / `<speaker_id>.txt` layout. That full prompt set is
not redistributed here (ReazonSpeech audio is not ours to redistribute).
Point `--prompts_wav_dir`/`--prompts_trans_dir` at your own
equivalently-laid-out prompt set (48 candidates -> 42 after VAD filtering,
or your own selection) to reproduce the full eval.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
import warnings
from collections.abc import Iterable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path

# Reuse infer.py utilities for wav I/O, VAD trimming, and LoRA loading so
# this script cannot silently drift from the demo inference script's
# definitions of those operations. These are absolute (not relative)
# imports on purpose: relative imports (`from .patch import ...`) only work
# when this module is loaded as part of the `scripts` package (e.g. via
# `python -m scripts.cv2.generate_eval_audio`), and break with "attempted
# relative import with no known parent package" if this file is instead run
# directly as `python scripts/cv2/generate_eval_audio.py` (which executes
# it as `__main__`, outside any package). Absolute imports work in both
# cases, as long as the repo root is importable — install once with
# `pip install -e .` (see pyproject.toml) instead of the sys.path hacks
# this script used to carry.
from scripts.cv2 import infer as infer_mod
from scripts.cv2.patch import apply_patch

apply_patch()

from cosyvoice.cli.cosyvoice import CosyVoice2

# eval-set TSV -> (path relative to repo root, columns usable as --condition)
_EVAL_SET_FILES = {
    "easy": "data/eval/easy_50.tsv",
    "difficult": "data/eval/difficult_50.tsv",
    "katakana": "data/eval/katakana_50.tsv",
}
# Condition (TSV column) options per eval set; see data/README.md.
_EVAL_SET_CONDITIONS = {
    "easy": ["vanilla", "phon_tagged"],
    "difficult": ["vanilla", "kana_baseline", "phon_tagged"],
    "katakana": ["text"],
}


def _read_eval_tsv(path: Path, condition: str) -> list[tuple[str, str]]:
    """Return a list of (id, sentence) pairs for the given condition column."""
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if condition not in (reader.fieldnames or []):
            raise ValueError(
                f"condition column '{condition}' not found in {path} "
                f"(available columns: {reader.fieldnames})"
            )
        rows = []
        for row in reader:
            rid = row["id"].strip()
            sent = row[condition].strip()
            if sent:
                rows.append((rid, sent))
        return rows


def _iter_prompt_ids(prompts_wav_dir: Path) -> Iterable[str]:
    for p in sorted(prompts_wav_dir.glob("*.wav")):
        if p.is_file():
            yield p.stem


def _drain_futures(pending: list[Future], *, keep_at_most: int) -> None:
    if keep_at_most < 0:
        keep_at_most = 0
    if pending:
        done_now = [f for f in pending if f.done()]
        for f in done_now:
            f.result()
        if done_now:
            pending[:] = [f for f in pending if not f.done()]
    while len(pending) > keep_at_most:
        done, not_done = wait(pending, return_when=FIRST_COMPLETED)
        for f in done:
            f.result()
        pending[:] = list(not_done)


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    for name in ["cosyvoice", "cosyvoice.cli", "cosyvoice.cli.cosyvoice", "modelscope", "urllib3", "httpx", "requests"]:
        logging.getLogger(name).setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", category=FutureWarning)

    ap = argparse.ArgumentParser(
        description=(
            "Synthesize + VAD-trim eval audio reproducing the UtterTune paper's "
            "Table 1 procedure (unconditional bidirectional VAD trim of every "
            "synthesized utterance)."
        ),
        epilog=(
            "Example:\n"
            "  python -m scripts.cv2.generate_eval_audio \\\n"
            "      --base_model pretrained_models/CosyVoice2-0.5B \\\n"
            "      --lora_dir lora_weights/UtterTune-CosyVoice2-ja-JSUTJVS \\\n"
            "      --eval_set easy --condition phon_tagged \\\n"
            "      --prompts_wav_dir prompts/wav --prompts_trans_dir prompts/trans \\\n"
            "      --out_root eval_audio"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # NOTE: --base_model is intentionally `str`, not `Path`: CosyVoice2.__init__
    # (in the CosyVoice submodule) does `'-Instruct' in model_dir`, a substring
    # check that raises TypeError on a Path (Path has no __contains__). Keep
    # this in sync with infer.py's --base_model, which has the same constraint.
    ap.add_argument("--base_model", required=True, type=str, help="CosyVoice2 base model directory")
    ap.add_argument("--lora_dir", default=None, type=Path, help="LoRA adapter dir (omit for the base/vanilla system)")
    ap.add_argument("--eval_set", required=True, choices=sorted(_EVAL_SET_FILES), help="Which released eval TSV to synthesize")
    ap.add_argument(
        "--condition",
        required=True,
        type=str,
        help=(
            "Which text column of the eval TSV to synthesize (also used as the "
            "output subdirectory name). Valid per --eval_set: "
            f"easy={_EVAL_SET_CONDITIONS['easy']} "
            f"difficult={_EVAL_SET_CONDITIONS['difficult']} "
            f"katakana={_EVAL_SET_CONDITIONS['katakana']}"
        ),
    )
    ap.add_argument("--eval_tsv", default=None, type=Path, help="Override path to the eval TSV (default: data/eval/<eval_set>_50.tsv)")
    ap.add_argument("--prompts_wav_dir", default=Path("prompts/wav"), type=Path, help="Directory containing per-speaker prompt wavs (*.wav)")
    ap.add_argument("--prompts_trans_dir", default=Path("prompts/trans"), type=Path, help="Directory containing per-speaker prompt transcripts (*.txt)")
    ap.add_argument("--out_root", required=True, type=Path, help="Root directory for generated eval wavs")
    ap.add_argument("--max_speakers", type=int, default=None, help="If set, only process the first N speakers")
    ap.add_argument("--max_sentences", type=int, default=None, help="If set, only process the first N sentences")
    ap.add_argument(
        "--skip_existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip utterances whose output wav already exists (resume-friendly).",
    )
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--seed", type=int, default=0)

    # VAD trim knobs -- defaults match infer.py's trim_wav() defaults.
    ap.add_argument("--trim_trigger_level", type=float, default=7.0, help="torchaudio.functional.vad trigger_level")
    ap.add_argument("--trim_allowed_gap", type=float, default=0.25, help="torchaudio.functional.vad allowed_gap (seconds)")

    ap.add_argument("--save_workers", type=int, default=0, help="Number of worker threads for trim+save (0 = fully serial)")
    ap.add_argument("--max_pending_saves", type=int, default=8, help="Max queued trim+save tasks when --save_workers>0")
    ap.add_argument("--progress_every", type=int, default=10, help="Print a progress line every N utterances")

    args = ap.parse_args()

    if args.save_workers < 0:
        raise ValueError("--save_workers must be >= 0")
    if args.max_pending_saves < 1:
        raise ValueError("--max_pending_saves must be >= 1")
    valid_conditions = _EVAL_SET_CONDITIONS[args.eval_set]
    if args.condition not in valid_conditions:
        raise ValueError(f"--condition {args.condition!r} invalid for --eval_set {args.eval_set!r}; expected one of {valid_conditions}")

    import numpy as np
    import torch

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    cv2 = CosyVoice2(model_dir=args.base_model, fp16=False)
    logging.disable(logging.INFO)

    if args.lora_dir is not None:
        tok = infer_mod.load_lora_adapter(cv2, args.base_model, args.lora_dir, device)
        cv2.frontend.tokenizer = tok

    repo_root = Path(__file__).resolve().parents[2]
    eval_tsv = args.eval_tsv if args.eval_tsv else (repo_root / _EVAL_SET_FILES[args.eval_set])
    rows = _read_eval_tsv(eval_tsv, args.condition)
    if args.max_sentences is not None:
        rows = rows[: args.max_sentences]

    prompts_wav_dir = args.prompts_wav_dir
    prompts_trans_dir = args.prompts_trans_dir
    out_root = args.out_root / args.condition

    SR = 16000  # prompt speech is expected at 16k

    spk_ids = list(_iter_prompt_ids(prompts_wav_dir))
    if args.max_speakers is not None:
        spk_ids = spk_ids[: args.max_speakers]
    print(f"[run] eval_set={args.eval_set} condition={args.condition} speakers={len(spk_ids)} sentences={len(rows)}", file=sys.stderr)

    def _trim_and_save(wav: "torch.Tensor", out_path: Path) -> bool:
        try:
            # Bidirectional VAD trim: this is unconditional here (unlike
            # infer.py's opt-in, heuristic-gated --trim_out), matching the
            # paper's stated eval-audio generation procedure.
            trimmed = infer_mod.trim_wav(
                wav,
                cv2.sample_rate,
                trigger_level=args.trim_trigger_level,
                allowed_gap=args.trim_allowed_gap,
            )
            # Fail open: if VAD decides everything is silence, don't destroy audio.
            wav_to_save = trimmed if int(trimmed.shape[-1]) > 0 else wav
            if wav_to_save is wav:
                print(f"[warn] VAD trim produced empty wav; saving untrimmed audio: path={out_path}", file=sys.stderr)
            infer_mod.save_wav(out_path, wav_to_save, cv2.sample_rate)
            return True
        except Exception as e:
            print(f"[warn] failed trim+save: path={out_path} err={type(e).__name__}: {e}", file=sys.stderr)
            return False

    pool: ThreadPoolExecutor | None = None
    pending: list[Future] = []
    if args.save_workers > 0:
        pool = ThreadPoolExecutor(max_workers=args.save_workers)

    start_wall = time.perf_counter()
    done_utts = 0
    failed_utts = 0
    total_utts = len(spk_ids) * len(rows)

    for spk_pos, spk_id in enumerate(spk_ids, start=1):
        try:
            prompt_wav = prompts_wav_dir / f"{spk_id}.wav"
            prompt_txt = prompts_trans_dir / f"{spk_id}.txt"
            if not prompt_txt.exists():
                raise FileNotFoundError(f"missing prompt transcript: {prompt_txt}")
            prompt_text = prompt_txt.read_text("utf-8").strip()

            # Prompt speech is also VAD-trimmed unconditionally, matching
            # infer.py's (unconditional, non-heuristic) prompt trimming.
            prompt_raw_16k = infer_mod.load_wav(prompt_wav, SR)
            prompt_trimmed_16k = infer_mod.trim_wav(
                prompt_raw_16k, SR, trigger_level=args.trim_trigger_level, allowed_gap=args.trim_allowed_gap
            )
            prompt_speech_16k = prompt_trimmed_16k if int(prompt_trimmed_16k.shape[-1]) > 0 else prompt_raw_16k
            if int(prompt_speech_16k.shape[-1]) <= 0:
                print(f"[skip] empty prompt wav after load/trim: spk_id={spk_id} path={prompt_wav}", file=sys.stderr)
                continue

            out_dir = out_root / spk_id / args.eval_set
            out_dir.mkdir(parents=True, exist_ok=True)

            for rid, sentence in rows:
                out_path = out_dir / f"{rid:0>3}.wav"
                if args.skip_existing and out_path.exists() and out_path.stat().st_size > 44:
                    done_utts += 1
                    continue

                try:
                    t0 = time.perf_counter()
                    # Uses CosyVoice2's high-level `inference_zero_shot` (same call
                    # used by scripts/cv2/infer.py) rather than the source repo's
                    # hand-rolled batched LLM/Flow/vocoder pipeline. Its internal
                    # default `sampling=25` (see cosyvoice/llm/llm.py) matches the
                    # source script's explicit `--llm_sampling=25` default, so this
                    # is equivalent to the paper's eval generation.
                    wav_iter = cv2.inference_zero_shot(
                        tts_text=sentence,
                        prompt_text=prompt_text,
                        prompt_speech_16k=prompt_speech_16k,
                    )
                    wav_dict = next(wav_iter)
                    wav = wav_dict["tts_speech"]
                    dt = time.perf_counter() - t0

                    if pool is None:
                        _trim_and_save(wav, out_path)
                    else:
                        pending.append(pool.submit(_trim_and_save, wav, out_path))
                        _drain_futures(pending, keep_at_most=args.max_pending_saves)
                except Exception as e:
                    failed_utts += 1
                    print(f"[warn] failed synthesis: spk_id={spk_id} id={rid} err={type(e).__name__}: {e}", file=sys.stderr)

                done_utts += 1
                if (done_utts % args.progress_every) == 0 or done_utts == total_utts:
                    elapsed = time.perf_counter() - start_wall
                    print(
                        f"[progress] {done_utts}/{total_utts} spk={spk_pos}/{len(spk_ids)} "
                        f"fail={failed_utts} elapsed={elapsed:.1f}s",
                        file=sys.stderr,
                    )
        except Exception as e:
            print(f"[warn] failed speaker: spk_id={spk_id} err={type(e).__name__}: {e}", file=sys.stderr)
            continue

    if pending:
        _drain_futures(pending, keep_at_most=0)
    if pool is not None:
        pool.shutdown(wait=True)

    elapsed = time.perf_counter() - start_wall
    print(f"[summary] utts={done_utts}/{total_utts} fail={failed_utts} elapsed={elapsed:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
