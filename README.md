# 🎛️ UtterTune
**LoRA-based phoneme-level pronunciation and prosody control for LLM-based TTS with no G2P** (currently supports **Japanese** in **[CosyVoice 2](https://github.com/FunAudioLLM/CosyVoice)**)

[![arXiv](https://img.shields.io/badge/arXiv-2508.09767-b31b1b.svg)](https://www.arxiv.org/abs/2508.09767)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-HuggingFace-yellow)](https://huggingface.co/shuheikatoinfo/UtterTune-CosyVoice2-ja-JSUTJVS)
[![Static Demo](https://img.shields.io/badge/Demo-GitHub%20Pages-blue)](https://shuheikatoinfo.github.io/UtterTune)
<!-- [![Interactive Demo](https://img.shields.io/badge/Demo-Gradio-orange)](https://gradio.app/g/your-username/UtterTune) -->

## 📜 The Story
Have you ever **struggled with correcting pronunciation errors** in text-to-speech (TTS) based on a large language model (LLM) architecture due to a **lack of an explicit grapheme-to-phoneme (G2P) module**? Don't we have a way to control pronunciation, **including prosody**, in such a case?

**No**. ***UtterTune*** is a lightweight LoRA adapter and toolset to edit and control **phoneme-level pronunciation and prosody** in LLM-TTS with no explicit G2P. Currently, UtterTune supports **Japanese** in **[CosyVoice 2](https://github.com/FunAudioLLM/CosyVoice)**.

Omitting G2P modules facilitates multilingual training, leading to superb performance in many languages. Users lost, instead, phoneme-level pronunciation controllability. Lacking controllability matters for non-eligible users – **I *was* one of them**.

**Now we have UtterTune**. UtterTune users can teach correct pronunciation to the model using phonograms (kana in the case of Japanese) enclosed by newly-introduced special tag tokens. The size of the UtterTune model is **less than 10MB**, thanks to low-rank adapter (LoRA) technology, while the original CosyVoice 2-0.5B model size is close to 1 GB.

Why not try using Japanese CosyVoice 2 **more comfortably** with UtterTune's pretrained model? Or why not try training your own custom UtterTune?


## ✨️ Features
### LoRA fine-tuning
UtterTune doesn't need full fine-tuning for the base model’s LLM component.

### Special token injection
`<PHON_START>`, `<PHON_END>` is newly introduced for phoneme-level pronunciation control.

### No interference with other languages' performance
Because you only need LoRA for the target language.

### Pretrained LoRA weights are available
[You can download pretrained weights from Hugging Face](https://huggingface.co/shuheikatoinfo/UtterTune-CosyVoice2-ja-JSUTJVS) (*non-commercial license* due to the training data).

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


## Setup
### Update submodules
```bash
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

### 3. Setup a virtual environment
```bash
# For CosyVoice 2
python -m venv venvs/cv2. # 3.10
. venvs/cv2/bin/activate
pip install -r submodules/CosyVoice/requirements.txt  -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com

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

# If sox compatibility issues raised
# Ubuntu
sudo apt-get install sox libsox-dev
# CentOS
sudo yum install sox sox-devel
```

### 4. Inference
```bash
python -m scripts.cv2.infer \
    --base_model pretrained_models/CosyVoice2-0.5B \
    --lora_dir lora_weights/UtterTune-CosyVoice2-jp-JSUTJVS \
    --texts "魑魅魍魎が跋扈する。|チミモーリョーがバッコする。|<PHON_START>チ'ミ/モーリョー<PHON_END>が<PHON_START>バ'ッコ<PHON_END>する。" \
    --prompt_wav prompts/wav/common_voice_ja_41758953.wav \
    --prompt_text prompts/trans/common_voice_ja_41758953.txt
```

## 💪 Training

### 1. Data preparation
Download [JSUT](https://sites.google.com/site/shinnosuketakamichi/publication/jsut) and [JVS](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus) corpora, and replace portion of words with its pronunciation `<PHON_START>` and `<PHON_END>` in each transcription like this:

```yaml
# Original
BASIC5000_0004:一週間して、そのニュースは本当になった。

# After replacement
BASIC5000_0004:<PHON_START>イッシュ'ーカン<PHON_END>して、そのニュースは本当になった。
```

*We plan to provide patch for JSUT and JVS corpora.

Then, use `extract_speech_tokens.py` and `prepare_manifest.py` in `scripts/cv2`.

### 2. Train
```bash
python -m scripts.cv2.train --config configs/train/jsutjvs.yaml
```

## Input sentences for the sample files

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
  title={UtterTune: UtterTune: LoRA-Based Target-Language Pronunciation Edit and Control in Multilingual Text-to-Speech},
  author={Kato, Shuhei},
  year={2025},
  howpublished={arXiv:2508.09767 [cs.CL]},
}
```