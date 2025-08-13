#!/usr/bin/env python3
"""
CosyVoice 2 + LoRA inference script
=================================
Load a prerained LoRA adapter and synthesize speech from any text list.

Usage:
    python -m scripts.cv2.infer \
        --base_model pretrained_models/CosyVoice2-0.5B \
        --lora_dir lora_weights/cv2/ja/jsutjvs/checkpoint-20000 \
        --texts "魑魅魍魎が跋扈する。|<PHON_START>チ'ミ/モーリョー<PHON_END>が<PHON_START>バ'ッコ<PHON_END>する。" \
        --prompt_wav prompts/wav/common_voice_ja_41758953.wav \
        --prompt_text prompts/trans/common_voice_ja_41758953.txt
"""

from __future__ import annotations

import argparse
import time
import warnings
from logging import getLogger, StreamHandler, INFO
from pathlib import Path
from typing import List

import huggingface_hub
import numpy as np
import safetensors.torch as st
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from peft import PeftModel

from scripts.cv2.patch import apply_patch

apply_patch()

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.tokenizer.tokenizer import get_qwen_tokenizer

huggingface_hub.cached_download = hf_hub_download

logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(INFO)
logger.setLevel(INFO)
logger.addHandler(handler)
logger.propagate = False


def load_wav(path: Path, sr_out: int) -> np.ndarray:
    wav, sr = torchaudio.load(path)

    if sr != sr_out:
        wav = torchaudio.functional.resample(wav, sr, sr_out)

    return wav


def trim_wav(wav: torch.Tensor, sr: int, trigger_level: float = 7.0) -> torch.Tensor:
    # Cut the beginning of the audio signal with VAD
    trimmed = torchaudio.functional.vad(wav, sr, trigger_level=trigger_level)

    # Cut the end of the voice signal (reverse and VAD again)
    if trimmed.shape[-1] > 0:
        trimmed_rev = torchaudio.functional.vad(
            trimmed.flip(-1), sr, trigger_level=trigger_level
        )
        trimmed = trimmed_rev.flip(-1)

    return trimmed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base_model",
        type=str,
        required=True,
        default="pretrained_models/CosyVoice2-0.5B",
        help="CosyVoice2 base model directory",
    )
    ap.add_argument(
        "--lora_dir",
        type=Path,
        default=None,
        help="LoRA adapter directory",
    )
    ap.add_argument(
        "--texts",
        required=True,
        help="Synthesise these sentences; separated by | or text file path",
    )
    ap.add_argument(
        "--prompt_wav",
        type=Path,
        required=True,
        help="Prompt wav file path); less than 4-second duration is recommended",
    )
    ap.add_argument(
        "--prompt_text",
        required=True,
        help="Transcription for the prompt wav",
    )
    ap.add_argument("--out_dir", type=Path, default="wavs_out")
    ap.add_argument("--trim_out", action="store_true", help="Trim synthesized speech")
    ap.add_argument(
        "--cpu", action="store_true", help="Force CPU inference (for debug)"
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    SAMPLE_RATE = 16000

    cv2 = CosyVoice2(model_dir=args.base_model, fp16=False)

    if args.lora_dir is not None:
        base_model = cv2.model.llm

        # Expand vocabulary
        tok = get_qwen_tokenizer(
            token_path=f"{args.base_model}/CosyVoice-BlankEN", skip_special_tokens=True
        )

        # Register new special tokens
        new_tokens = ["<PHON_START>", "<PHON_END>"]
        added = tok.tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
        logger.info("Number of tokens added: %s", added)

        # Update the meta information on the QwenTokenizer
        tok.special_tokens["additional_special_tokens"].extend(
            [
                t
                for t in new_tokens
                if t not in tok.special_tokens["additional_special_tokens"]
            ]
        )
        base_model.llm.model.resize_token_embeddings(len(tok.tokenizer))
        new_ids = tok.tokenizer.convert_tokens_to_ids(new_tokens)
        w = cv2.model.llm.llm.model.model.embed_tokens.weight

        # Attach LoRA
        logger.info("Loading LoRA from %s", args.lora_dir)

        # Load LoRA weights
        hf_model = PeftModel.from_pretrained(
            base_model,
            args.lora_dir,
            is_trainable=False,
            torch_dtype=torch.float32,
        )
        hf_model.to(device).eval()

        # Load embeddings of the new tokens
        rows = st.load_file(args.lora_dir / "embed_patch.safetensors")["embed_rows"].to(
            device
        )

        with torch.no_grad():
            hf_model.base_model.llm.model.get_input_embeddings().weight[new_ids] = rows

        cv2.model.llm = hf_model
        w = cv2.model.llm.llm.model.model.embed_tokens.weight
        print(w[new_ids])
        print(f"new ids: {new_ids}")

    # I/O
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if Path(args.texts).is_file():
        sentences: List[str] = [
            ln.strip()
            for ln in Path(args.texts).read_text("utf-8").splitlines()
            if ln.strip()
        ]
    else:
        sentences = [s.strip() for s in args.texts.split("|") if s.strip()]

    prompt_speech_16k = load_wav(args.prompt_wav, SAMPLE_RATE)
    prompt_speech_16k = trim_wav(prompt_speech_16k, SAMPLE_RATE)

    if Path(args.prompt_text).is_file():
        prompt_text = Path(args.prompt_text).read_text("utf_8").strip()
    else:
        prompt_text = args.prompt_text

    for idx, sentence in enumerate(sentences):
        logger.info(f"[ {idx + 1:03d} ] \u270d︎ '{sentence[:30]}...' → synth...")

        t0 = time.perf_counter()

        wav_iter = cv2.inference_zero_shot(
            tts_text=sentence,
            prompt_text=prompt_text,
            prompt_speech_16k=prompt_speech_16k,
        )

        wav_dict = next(wav_iter)  # {'tts_speech': Tensor(1,T)}
        dt = time.perf_counter() - t0

        wav = wav_dict["tts_speech"]
        trimmed = wav

        # Save synthesized file
        out_path = out_dir / f"{idx + 1:03d}.wav"
        torchaudio.save(
            str(out_path), trimmed, cv2.sample_rate, format="wav", encoding="PCM_S"
        )
        logger.info(
            f"    saved → {out_path}  ({dt:.2f}s, {trimmed.shape[-1] / cv2.sample_rate:.2f}s)"
        )

        # [Optional] trim synthesized file
        if args.trim_out:
            if (wav.shape[-1] / cv2.sample_rate) > (
                5 + len(sentence) / 5
            ) and wav.shape[-1] > 0:  # Heuristics
                trimmed = trim_wav(wav, cv2.sample_rate)

                if trimmed.shape[-1] > 0:
                    out_path_trimmed = out_path.with_name(f"{idx + 1:03d}_trimmed.wav")
                    torchaudio.save(
                        str(out_path_trimmed),
                        trimmed,
                        cv2.sample_rate,
                        format="wav",
                        encoding="PCM_S",
                    )
                    logger.info(
                        f"    saved → {out_path_trimmed}  ({dt:.2f}s, {trimmed.shape[-1] / cv2.sample_rate:.2f}s)"
                    )

    logger.info("All sentences have been synthesised.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
    main()
