# 🎛️ UtterTune

*English | [日本語](README.ja.md)*

**LoRA-based phoneme-level pronunciation and prosody (rhythm and intonation) control for LLM-based TTS with no G2P**; a tool to convert text to pronunciation, currently supports **Japanese** in **[CosyVoice 2](https://github.com/FunAudioLLM/CosyVoice)**.

[![arXiv](https://img.shields.io/badge/arXiv-2508.09767-b31b1b.svg)](https://www.arxiv.org/abs/2508.09767)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-HuggingFace-yellow)](https://huggingface.co/shuheikatoinfo/UtterTune-CosyVoice2-ja-JSUTJVS)
[![Static Demo](https://img.shields.io/badge/Demo-GitHub%20Pages-blue)](https://shuheikatoinfo.github.io/UtterTune)

<!-- [![Interactive Demo](https://img.shields.io/badge/Demo-Gradio-orange)](https://gradio.app/g/your-username/UtterTune) -->

## 📜 The Story

**Struggling with pronunciation in LLM-based TTS without G2P**? Is there an intuitive way to control pronunciation and prosody in these models?

**UtterTune** is a lightweight LoRA adapter and toolset for precise **phoneme-level pronunciation and prosody** control in LLM-TTS systems that lack explicit G2P modules. Currently, UtterTune supports **Japanese** in [**CosyVoice 2**](https://github.com/FunAudioLLM/CosyVoice).

Omitting G2P modules helps with multilingual training and improves overall performance, but phoneme-level control was lost—an issue for some users, myself included.

UtterTune enables users to modify model pronunciation by using phonograms (such as kana in Japanese) enclosed within newly introduced special tag tokens. Thanks to low-rank adapter (LoRA) technology, the UtterTune model is less than **10 MB**; by comparison, the original CosyVoice 2-0.5B model is nearly 1 GB.

Consider using the Japanese CosyVoice 2 **more effectively** with the pretrained UtterTune model, or explore training your own custom UtterTune.

## ✨️ Features

### LoRA fine-tuning

UtterTune eliminates the need for fine-tuning a full LLM base model.

### Special token injection

Control phoneme-level pronunciation using `<PHON_START>` and `<PHON_END>` tokens.

### No interference with other languages' performance

Apply LoRA only to the target language to maintain performance in other languages.

### Pretrained LoRA weights are available!

[Pretrained weights](https://huggingface.co/shuheikatoinfo/UtterTune-CosyVoice2-ja-JSUTJVS) with a non-commercial license are downloadable from Hugging Face.

## 🛢️ Resources

[![arXiv](https://img.shields.io/badge/arXiv-2508.09767-b31b1b.svg)](https://www.arxiv.org/abs/2508.09767)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-HuggingFace-yellow)](https://huggingface.co/shuheikatoinfo/UtterTune-CosyVoice2-ja-JSUTJVS)
[![Static Demo](https://img.shields.io/badge/Demo-GitHub%20Pages-blue)](https://shuheikatoinfo.github.io/UtterTune)

<!-- [![Interactive Demo](https://img.shields.io/badge/Demo-Gradio-orange)](https://gradio.app/g/your-username/UtterTune) -->

## 💨 Quick Start

### 1. Clone & Update Submodules

```bash
git clone https://github.com/your-username/UtterTune.git
cd UtterTune
git submodule update --init --recursive
```

### 2. Download pretrained models

```bash
mkdir -p pretrained_models

# Download CosyVoice2-0.5B
git clone https://www.modelscope.cn/iic/CosyVoice2-0.5B.git pretrained_models/CosyVoice2-0.5B

# Download LoRA weights
git lfs install
git clone https://huggingface.co/shuheikatoinfo/UtterTune-CosyVoice2-ja-JSUTJVS lora_weights/UtterTune-CosyVoice2-ja-JSUTJVS
```

### 3. Set up a virtual environment

```bash
# For CosyVoice 2
python -m venv venvs/cv2. # 3.10
. venvs/cv2/bin/activate
pip install -r submodules/CosyVoice/requirements.txt  -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com

# Install this repo's own `scripts` package in editable mode, so
# `scripts.cv2.*` / `scripts.eval.*` are importable from anywhere without
# manual sys.path hacks (see pyproject.toml). UtterTune targets multiple
# LLM-TTS backbones over time (currently: CosyVoice 2); `scripts/cv2/*`
# only needs the core deps installed by the plain `pip install -e .` below.
# Add `[eval]` too if you'll also run the evaluation scripts:
pip install -e .
pip install -e ".[eval]"   # only needed for scripts/eval/*

# Add path to CosyVoice repository
python - <<'PY'
import site, os
sp = next(p for p in site.getsitepackages() if p.endswith("site-packages"))
pth = os.path.join(sp, "cosyvoice_submodule.pth")
with open(pth, "w", encoding="utf-8") as f:
    f.write(os.path.abspath("submodules/CosyVoice") + "\n")
    f.write(os.path.abspath("submodules/CosyVoice/third_party/Matcha-TTS") + "\n")
print("Wrote:", pth)
PY

# If sox compatibility issues are raised
# Ubuntu
sudo apt-get install sox libsox-dev
# CentOS
sudo yum install sox sox-devel
```

### 4. Inference

```bash
python -m scripts.cv2.infer \
    --base_model pretrained_models/CosyVoice2-0.5B \
    --lora_dir lora_weights/UtterTune-CosyVoice2-ja-JSUTJVS \
    --texts "魑魅魍魎が跋扈する。|チミモーリョーがバッコする。|<PHON_START>チ'ミ/モーリョー<PHON_END>が<PHON_START>バ'ッコ<PHON_END>する。" \
    --prompt_wav prompts/wav/common_voice_ja_41758953.wav \
    --prompt_text prompts/trans/common_voice_ja_41758953.txt
```

## 💪 Training

### 1. Data preparation

Download JSUT and JVS corpora, and replace portions of words with their pronunciation `<PHON_START>` and `<PHON_END>` in each transcription, like this:

```yaml
# Original
BASIC5000_0004:一週間して、そのニュースは本当になった。

# After replacement
BASIC5000_0004:<PHON_START>イッシュ'ーカン<PHON_END>して、そのニュースは本当になった。
```

Then, use extract_speech_tokens.py and prepare_manifest.py in scripts/cv2.

The training manifest actually used for the paper (text + pronunciation tags only, no audio) is released at `data/train/jsutjvs_manifest.tsv`; see [`data/README.md`](data/README.md) for schema/license details.

### 2. Train

```bash
python -m scripts.cv2.train --config configs/train/jsutjvs.yaml
```

## 📊 Evaluation

Eval audio is generated with scripts/cv2/generate_eval_audio.py, which applies VAD-based silence trimming to every utterance. See [`scripts/eval/README.md`](scripts/eval/README.md) for the full evaluation pipeline and scripts for Table 1: character error rate ([`calc_cer.py`](scripts/eval/calc_cer.py)), speaker similarity ([`calc_cosine_similarity_eres2net.py`](scripts/eval/calc_cosine_similarity_eres2net.py)), UTMOSv2 naturalness ([`calc_utmosv2.py`](scripts/eval/calc_utmosv2.py)), and significance tests ([`compare_scores_2systems.py`](scripts/eval/compare_scores_2systems.py), [`compare_scores_3systems.py`](scripts/eval/compare_scores_3systems.py)).

### Input sentences for the sample files:

```yaml
# 001 (prompt: common_voice_ja_41758953)
# 001_cv2_base.wav (CosyVoice 2)
魑魅魍魎が跋扈する。

# 001_cv2_base_kana.wav (CosyVoice 2)
チミモーリョーがバッコする。

# 001_cv2_uttertune.wav (CosyVoice 2 + UtterTune)
<PHON_START>チ'ミ/モーリョー<PHON_END>が<PHON_START>バ'ッコ<PHON_END>する。

# 002 (prompt: common_voice_ja_36360364)
# 002_cv2_base.wav (CosyVoice 2)
午後に甘いレモンティーを友達と静かに味わった。

# 002_cv2_uttertune.wav (CosyVoice 2 + UtterTune)
午後に甘い<PHON_START>レモ'ンティー<PHON_END>を友達と静かに味わった。

# 003 (prompt: common_voice_ja_41776640)
# 003_cv2_base.wav (CosyVoice 2)
朝練で彼は溌剌と声を出し皆を元気づけ、最後まで練習を引っ張った。

# 003_cv2_base_kana.wav (CosyVoice 2)
朝練で彼はハツラツと声を出し皆を元気づけ、最後まで練習を引っ張った。

# 003_cv2_uttertune.wav (CosyVoice 2 + UtterTune)
朝練で彼は<PHON_START>ハツラツ<PHON_END>と声を出し皆を元気づけ、最後まで練習を引っ張った。
```

## Citation

If you use UtterTune in your research, please cite the [paper](https://www.arxiv.org/abs/2508.09767):

```
@misc{Kato2025UtterTune,
  title={UtterTune: LoRA-Based Target-Language Pronunciation Edit and Control in Multilingual Text-to-Speech},
  author={Shuhei Kato},
  year={2025},
  howpublished={arXiv:2508.09767 [cs.SD]},
}
```