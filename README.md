# 🎛️ UtterTune
**LoRA-based phoneme-level pronunciation and prosody control for LLM-based TTS with no G2P** (currently supports **Japanese** in **[CosyVoice 2](https://github.com/FunAudioLLM/CosyVoice)**)

[![arXiv](https://img.shields.io/badge/arXiv-2501.xxxxx-b31b1b.svg)](https://arxiv.org/abs/2501.xxxxx)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-HuggingFace-yellow)](https://huggingface.co/your-username/UtterTune)
[![Static Demo](https://img.shields.io/badge/Demo-GitHub%20Pages-blue)](https://your-username.github.io/UtterTune)
[![Interactive Demo](https://img.shields.io/badge/Demo-Gradio-orange)](https://gradio.app/g/your-username/UtterTune)

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
On Hugging Face (*non-commercial license* due to the training data).


## 🎮️ Resources
[![arXiv](https://img.shields.io/badge/arXiv-2501.xxxxx-b31b1b.svg)](https://arxiv.org/abs/2501.xxxxx)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-HuggingFace-yellow)](https://huggingface.co/your-username/UtterTune)
[![Static Demo](https://img.shields.io/badge/Demo-GitHub%20Pages-blue)](https://your-username.github.io/UtterTune)
[![Interactive Demo](https://img.shields.io/badge/Demo-Gradio-orange)](https://gradio.app/g/your-username/UtterTune)

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

### Download pretrained models
```bash
mkdir -p pretrained_models

# Download CosyVoice2-0.5B
git clone https://www.modelscope.cn/iic/CosyVoice2-0.5B.git pretrained_models/CosyVoice2-0.5B
```

### Setup a virtual environment
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

