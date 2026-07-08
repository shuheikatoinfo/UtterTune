# 🎛️ UtterTune

*[English](README.md) | 日本語*

**G2Pを持たないLLMベースTTSのための、音素レベルの発音・韻律（リズムとイントネーション）制御をLoRAで実現する手法**です。テキストを発音に変換するツールで、現在は**[CosyVoice 2](https://github.com/FunAudioLLM/CosyVoice)における日本語**に対応しています。

[![arXiv](https://img.shields.io/badge/arXiv-2508.09767-b31b1b.svg)](https://www.arxiv.org/abs/2508.09767)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-HuggingFace-yellow)](https://huggingface.co/shuheikatoinfo/UtterTune-CosyVoice2-ja-JSUTJVS)
[![Static Demo](https://img.shields.io/badge/Demo-GitHub%20Pages-blue)](https://shuheikatoinfo.github.io/UtterTune)

<!-- [![Interactive Demo](https://img.shields.io/badge/Demo-Gradio-orange)](https://gradio.app/g/your-username/UtterTune) -->

## 📜 背景

**G2Pを持たないLLMベースTTSで、発音の誤りに悩まされたことはありませんか？** そうしたモデルでも、発音や韻律を直感的にコントロールする方法はないのでしょうか？

**UtterTune**は、明示的なG2Pモジュールを持たないLLM-TTSシステムに対して、**音素レベルの発音・韻律**を精密に制御するための軽量なLoRAアダプタ＆ツールセットです。現在は[**CosyVoice 2**](https://github.com/FunAudioLLM/CosyVoice)における日本語に対応しています。

G2Pモジュールを省略することは多言語学習に有利で全体の性能向上にも寄与しますが、その代わりに音素レベルの制御性が失われます。これは一部のユーザー（筆者自身も含む）にとって問題でした。

UtterTuneでは、新たに導入した特殊タグトークンで囲んだ表音文字（日本語の場合はカナ）を使って、モデルの発音を書き換えることができます。低ランクアダプタ (LoRA) 技術のおかげで、UtterTuneのモデルサイズは**10MB未満**に抑えられています（元のCosyVoice 2-0.5Bモデルは約1GBです）。

事前学習済みのUtterTuneモデルを使って日本語CosyVoice 2を**より快適に**使ってみませんか？ あるいは、自分だけのカスタムUtterTuneを学習させてみませんか？

## ✨️ 特長

### LoRAファインチューニング

UtterTuneはベースモデルのLLM部分をフルファインチューニングする必要がありません。

### 特殊トークンの導入

`<PHON_START>`と`<PHON_END>`という新規トークンで、音素レベルの発音を制御できます。

### 他言語の性能に影響しない

LoRAを対象言語にのみ適用するため、他言語の性能はそのまま維持されます。

### 事前学習済LoRA重みを提供！

[Hugging Faceから事前学習済の重みをダウンロード](https://huggingface.co/shuheikatoinfo/UtterTune-CosyVoice2-ja-JSUTJVS)できます（学習データの都合上、*非商用ライセンス*です）。

## 🛢️ リソース

[![arXiv](https://img.shields.io/badge/arXiv-2508.09767-b31b1b.svg)](https://www.arxiv.org/abs/2508.09767)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-HuggingFace-yellow)](https://huggingface.co/shuheikatoinfo/UtterTune-CosyVoice2-ja-JSUTJVS)
[![Static Demo](https://img.shields.io/badge/Demo-GitHub%20Pages-blue)](https://shuheikatoinfo.github.io/UtterTune)

<!-- [![Interactive Demo](https://img.shields.io/badge/Demo-Gradio-orange)](https://gradio.app/g/your-username/UtterTune) -->

## 💨 クイックスタート

### 1. クローン＆サブモジュール更新

```bash
git clone https://github.com/your-username/UtterTune.git
cd UtterTune
git submodule update --init --recursive
```

### 2. 事前学習済モデルのダウンロード

```bash
mkdir -p pretrained_models

# CosyVoice2-0.5Bをダウンロード
git clone https://www.modelscope.cn/iic/CosyVoice2-0.5B.git pretrained_models/CosyVoice2-0.5B

# LoRA重みをダウンロード
git lfs install
git clone https://huggingface.co/shuheikatoinfo/UtterTune-CosyVoice2-ja-JSUTJVS lora_weights/UtterTune-CosyVoice2-ja-JSUTJVS
```

### 3. 仮想環境のセットアップ

```bash
# CosyVoice 2用
python -m venv venvs/cv2. # 3.10
. venvs/cv2/bin/activate
pip install -r submodules/CosyVoice/requirements.txt  -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com

# CosyVoiceリポジトリへのパスを追加
python - <<'PY'
import site, os
sp = next(p for p in site.getsitepackages() if p.endswith("site-packages"))
pth = os.path.join(sp, "cosyvoice_submodule.pth")
with open(pth, "w", encoding="utf-8") as f:
    f.write(os.path.abspath("submodules/CosyVoice") + "\n")
    f.write(os.path.abspath("submodules/CosyVoice/third_party/Matcha-TTS") + "\n")
print("Wrote:", pth)
PY

# soxの互換性エラーが出た場合
# Ubuntu
sudo apt-get install sox libsox-dev
# CentOS
sudo yum install sox sox-devel
```

### 4. 推論

```bash
python -m scripts.cv2.infer \
    --base_model pretrained_models/CosyVoice2-0.5B \
    --lora_dir lora_weights/UtterTune-CosyVoice2-ja-JSUTJVS \
    --texts "魑魅魍魎が跋扈する。|チミモーリョーがバッコする。|<PHON_START>チ'ミ/モーリョー<PHON_END>が<PHON_START>バ'ッコ<PHON_END>する。" \
    --prompt_wav prompts/wav/common_voice_ja_41758953.wav \
    --prompt_text prompts/trans/common_voice_ja_41758953.txt
```

## 💪 学習

### 1. データ準備

JSUTおよびJVSコーパスをダウンロードし、各書き起こしテキストの一部の単語をその発音で`<PHON_START>`と`<PHON_END>`を使って置き換えます。

例:

```yaml
# 元のテキスト
BASIC5000_0004:一週間して、そのニュースは本当になった。

# 置換後
BASIC5000_0004:<PHON_START>イッシュ'ーカン<PHON_END>して、そのニュースは本当になった。
```

その後、`scripts/cv2`内の`extract_speech_tokens.py`と`prepare_manifest.py`を使用します。

なお、実際に論文で使用した学習manifest（テキスト＋発音タグのみ、音声データ本体は含まない）は`data/train/jsutjvs_manifest.tsv`として公開しています。ライセンス等の詳細は[`data/README.md`](data/README.md)を参照してください。

### 2. 学習

```bash
python -m scripts.cv2.train --config configs/train/jsutjvs.yaml
```

## 📊 評価

評価用音声は`scripts/cv2/generate_eval_audio.py`で生成しており、すべての発話に対してVADによる無音区間トリミングを適用しています。評価パイプライン全体、および論文のTable 1を再現するスクリプト一式（文字誤り率: [`calc_cer.py`](scripts/eval/calc_cer.py)、話者類似度: [`calc_cosine_similarity_eres2net.py`](scripts/eval/calc_cosine_similarity_eres2net.py)、UTMOSv2による自然性: [`calc_utmosv2.py`](scripts/eval/calc_utmosv2.py)、有意差検定: [`compare_scores_2systems.py`](scripts/eval/compare_scores_2systems.py)、[`compare_scores_3systems.py`](scripts/eval/compare_scores_3systems.py)）については[`scripts/eval/README.md`](scripts/eval/README.md)を参照してください。

### サンプルファイルの入力文:

```yaml
# 001（プロンプト: common_voice_ja_41758953）
# 001_cv2_base.wav（CosyVoice 2）
魑魅魍魎が跋扈する。

# 001_cv2_base_kana.wav（CosyVoice 2）
チミモーリョーがバッコする。

# 001_cv2_uttertune.wav（CosyVoice 2 + UtterTune）
<PHON_START>チ'ミ/モーリョー<PHON_END>が<PHON_START>バ'ッコ<PHON_END>する。

# 002（プロンプト: common_voice_ja_36360364）
# 002_cv2_base.wav（CosyVoice 2）
午後に甘いレモンティーを友達と静かに味わった。

# 002_cv2_uttertune.wav（CosyVoice 2 + UtterTune）
午後に甘い<PHON_START>レモ'ンティー<PHON_END>を友達と静かに味わった。

# 003（プロンプト: common_voice_ja_41776640）
# 003_cv2_base.wav（CosyVoice 2）
朝練で彼は溌剌と声を出し皆を元気づけ、最後まで練習を引っ張った。

# 003_cv2_base_kana.wav（CosyVoice 2）
朝練で彼はハツラツと声を出し皆を元気づけ、最後まで練習を引っ張った。

# 003_cv2_uttertune.wav（CosyVoice 2 + UtterTune）
朝練で彼は<PHON_START>ハツラツ<PHON_END>と声を出し皆を元気づけ、最後まで練習を引っ張った。
```

## 引用

研究にUtterTuneをご利用いただく際は、[論文](https://www.arxiv.org/abs/2508.09767)を引用してください。

```
@misc{Kato2025UtterTune,
  title={UtterTune: LoRA-Based Target-Language Pronunciation Edit and Control in Multilingual Text-to-Speech},
  author={Shuhei Kato},
  year={2025},
  howpublished={arXiv:2508.09767 [cs.SD]},
}
```
