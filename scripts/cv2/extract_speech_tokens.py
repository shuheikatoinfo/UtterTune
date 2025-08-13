#!/usr/bin/env python3
"""
WAV to speech token extractor **using CosyVoice speech_tokenizer_v2.onnx**.
Produces a one-to-one `.npy` per WAV and writes a helper TSV mapping.

- The ONNX model is distributed with CosyVoice2-0.5B release assets:
    pretrained_models/CosyVoice2-0.5B/speech_tokenizer_v2.onnx
- The model expects 24kHz, mono, normalized PCM -1..1.
"""

import argparse
from logging import getLogger, StreamHandler, INFO
from pathlib import Path

import numpy as np
import onnxruntime as ort
import tqdm
import torchaudio
import whisper

logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(INFO)
logger.setLevel(INFO)
logger.addHandler(handler)
logger.propagate = False

ap = argparse.ArgumentParser()
ap.add_argument("--wav_root", type=Path, required=True)
ap.add_argument("--out_dir", type=Path, required=True)
ap.add_argument(
    "--onnx_path",
    type=Path,
    required=True,
    help="speech_tokenizer_v2.onnx from CosyVoice2 release",
)
args = ap.parse_args()

option = ort.SessionOptions()
option.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
providers = ["CPUExecutionProvider"]  # ["CUDAExecutionProvider"]
sess = ort.InferenceSession(str(args.onnx_path), providers=providers)
input_name = sess.get_inputs()[0].name  # "audio"
output_name = sess.get_outputs()[0].name  # "tokens"

args.out_dir.mkdir(parents=True, exist_ok=True)
manifest_lines = []

for wav in tqdm.tqdm(sorted(args.wav_root.rglob("*.wav"))):
    audio, sr = torchaudio.load(wav, backend="soundfile")

    if sr != 16000:
        audio = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(audio)

    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)

    if audio.shape[1] / 16000 > 30:
        logger.warning("do not support extract speech token for audio longer than 30s")
    else:
        feat = whisper.log_mel_spectrogram(audio, n_mels=128)
        tokens = (
            sess.run(
                None,
                {
                    sess.get_inputs()[0].name: feat.detach().cpu().numpy(),
                    sess.get_inputs()[1].name: np.array(
                        [feat.shape[2]], dtype=np.int32
                    ),
                },
            )[0]
            .flatten()
            .tolist()
        )
        out = args.out_dir / f"{wav.stem}.npy"
        np.save(out, tokens)
        manifest_lines.append(f"{wav}	{out}")

logger.info("✓", len(manifest_lines), "files →", args.out_dir)
