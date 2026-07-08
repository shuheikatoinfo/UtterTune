#!/usr/bin/env python3
"""Compute cosine similarity between a reference wav and a target wav.

This script is used for speaker similarity evaluation.

Input options:
- Provide both `--ref_csv` and `--target_csv` (1-column CSVs, no header), OR
- Provide `--target_csv` and `--prompts_wav_dir`, in which case reference wavs are
    derived from each target wav path by speaker id.

Assumed target wav path layout when `--prompts_wav_dir` is used:
    <...>/<cond>/<speaker>/<set>/<utt>.wav
So speaker id is `Path(target_wav).parents[1].name`.
"""
import argparse
import math
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


def _load_audio_mono_float32(path: str) -> tuple[np.ndarray, int]:
    import soundfile as sf

    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    # soundfile: (T, C)
    audio_mono = np.mean(audio, axis=1)
    return audio_mono.astype(np.float32), int(sr)


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio.astype(np.float32)

    from scipy.signal import resample_poly

    gcd = math.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd
    return resample_poly(audio, up, down).astype(np.float32)


def _speaker_id_from_target_path(target_wav: str) -> str:
    p = Path(target_wav)
    # .../<cond>/<speaker>/<set>/<utt>.wav
    return p.parents[1].name


def _build_ref_df_from_targets(target_wavs: pd.Series, prompts_wav_dir: Path) -> pd.DataFrame:
    prompts_wav_dir = prompts_wav_dir.expanduser().resolve()
    if not prompts_wav_dir.is_dir():
        raise SystemExit(f"prompts_wav_dir not found: {prompts_wav_dir}")

    ref_wavs = []
    for w in target_wavs:
        spk = _speaker_id_from_target_path(str(w))
        ref = prompts_wav_dir / f"{spk}.wav"
        if not ref.is_file():
            raise SystemExit(f"missing prompt wav for speaker '{spk}': {ref}")
        ref_wavs.append(str(ref))
    return pd.DataFrame({"ref_wav": ref_wavs})


def main(args):
    df_target = pd.read_csv(args.target_csv, names=["target_wav"])
    if args.ref_csv is not None:
        df_ref = pd.read_csv(args.ref_csv, names=["ref_wav"])
    else:
        if args.prompts_wav_dir is None:
            raise SystemExit("Either --ref_csv or --prompts_wav_dir is required")
        df_ref = _build_ref_df_from_targets(df_target["target_wav"], args.prompts_wav_dir)

    if len(df_ref) != len(df_target):
        raise SystemExit(f"ref_csv rows ({len(df_ref)}) != target_csv rows ({len(df_target)})")

    df = pd.concat([df_ref.reset_index(drop=True), df_target.reset_index(drop=True)], axis=1)
    embed_pipe = pipeline(
        Tasks.speaker_verification,
        model="iic/speech_eres2net_sv_zh-cn_16k-common",
        device=args.device,
    )

    scores = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="ERes2Net-large"):
        signal_ref, sr_ref = _load_audio_mono_float32(row["ref_wav"])
        signal_ref = _resample(signal_ref, sr_ref, 16000)

        signal_target, sr_target = _load_audio_mono_float32(row["target_wav"])
        signal_target = _resample(signal_target, sr_target, 16000)

        # ModelScope pipeline expects `list` of items; pass numpy arrays to avoid
        # its internal torchaudio.sox_effects resampling path.
        emb_ref = embed_pipe([signal_ref], output_emb=True)["embs"][0]
        emb_target = embed_pipe([signal_target], output_emb=True)["embs"][0]
        score = np.dot(emb_ref, emb_target) / (np.linalg.norm(emb_ref) * np.linalg.norm(emb_target))  # Cosine similarity
        scores.append(float(score))

    df["cosine_similarity"] = scores
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--target_csv", required=True, help="1-column CSV with target wav paths")
    p.add_argument("--ref_csv", default=None, help="1-column CSV with ref wav paths")
    p.add_argument(
        "--prompts_wav_dir",
        type=Path,
        default=None,
        help="If set and --ref_csv is omitted, reference wav is prompts_wav_dir/<speaker>.wav",
    )
    p.add_argument("--out", type=Path, default="clip_scores.csv")
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="ModelScope device string, e.g. 'cuda', 'cuda:0', or 'cpu'",
    )
    main(p.parse_args())
