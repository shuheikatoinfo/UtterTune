# 評価スクリプト

*[English](README.md) | 日本語*

これらのスクリプトを使用することで、[UtterTune論文](https://arxiv.org/abs/2508.09767)のTable 1で報告している客観指標と統計量 —— 文字誤り率（CER）、話者類似度、UTMOSv2による自然性、およびそれらの発話単位の指標をもとにした有意差検定・効果量 —— を再現することができます。

全体のパイプラインは次の通りです。

1. **評価用音声の合成＋トリミング**: `scripts/cv2/generate_eval_audio.py`を条件ごとに1回ずつ実行します（例: `--lora_dir`なしのbaseline CosyVoice 2、`--lora_dir`にアダプタを指定したCosyVoice 2 + UtterTune）。対象は公開済みの評価用テスト文セット（`data/eval/easy_50.tsv`、`data/eval/difficult_50.tsv`、`data/eval/katakana_50.tsv`。列構成は`data/README.md`を参照）です。このスクリプトは論文の手順どおり、**すべての**生成音声に対して無条件で双方向VADトリミングを適用します。これは論文の報告数値を実際に生成した手順そのもので、トリミングが任意指定（`--trim_out`）かつヒューリスティック判定であるデモ用の`scripts/cv2/infer.py`とは異なります。
2. 生成音声に対して自動音声認識（Whisper large-v3、かな＋限定記号セットに制約したデコーディング）を実行し、書き起こしを得ます。このステップはここには含まれておらず、`calc_cer.py`より前段の処理です。
3. `calc_cer.py` / `calc_cosine_similarity_eres2net.py` / `calc_utmosv2.py`を生成音声あるいはASRの書き起こしに対して実行し、発話単位の指標を得ます。
4. 得られた発話単位の指標を、システム/条件ごとに1列、発話ごとに1行となるようワイド形式のCSVに整形し、`compare_scores_2systems.py`（2システム比較、例: Test Set 1）または`compare_scores_3systems.py`（3システム比較、例: Test Set 2）を実行することで、論文が報告する有意差検定と効果量を得ることができます。

## スクリプト一覧

### `calc_cer.py` — 文字誤り率

ASRの書き起こしと参照テキスト（両方ともカタカナに正規化）の間で文字レベルの編集距離を計算します。Whisperがしばしば生成する末尾の幻覚的な連続文字列を除去するトリミング処理も含みます。

使用例:

```bash
python -m scripts.eval.calc_cer \
    --hyp_csv whisper_hypotheses.csv \
    --ref_csv reference_texts.csv \
    --output results/cer_results.csv
```

`whisper_hypotheses.csv`には`audio_path`列と`hypothesis`列が必要です。`reference_texts.csv`はヘッダーなしの1列CSVで、仮説CSVと同じ行数・行順である必要があります。

### `calc_cosine_similarity_eres2net.py` — 話者類似度

参照（プロンプト）音声と各対象音声との間で、話者埋め込み（ModelScopeの`iic/speech_eres2net_sv_zh-cn_16k-common`モデルを使用）のコサイン類似度を計算します。

使用例:

```bash
python -m scripts.eval.calc_cosine_similarity_eres2net \
    --target_csv synthesized_wavs.csv \
    --prompts_wav_dir prompts/wav \
    --out results/speaker_sim.csv
```

`--ref_csv`（`--target_csv`と行単位で対応する参照音声パスのヘッダーなしCSV）または`--prompts_wav_dir`のいずれかを指定します。後者の場合、各対象音声の参照音声は、対象音声の話者IDから（想定パス構造`<cond>/<speaker>/<set>/<utt>.wav`をもとに）`<prompts_wav_dir>/<speaker>.wav`として導出されます。

### `calc_utmosv2.py` — UTMOSv2自然性

音声ファイルごとにUTMOSv2による自然性スコアを計算します。再開可能なチェックポイント機能とOOM対策のバッチサイズ自動縮小に対応しています。

使用例:

```bash
python -m scripts.eval.calc_utmosv2 \
    --csv synthesized_wavs.csv \
    --out results/utmos_scores.csv \
    --batch_size 64 --tqdm
```

`synthesized_wavs.csv`はヘッダーなし・1行1パスの音声ファイル一覧です。同じ`--out`で再実行すると、既存のチェックポイントから再開することができます。

### `compare_scores_2systems.py` — 2システム間の有意差検定＋効果量

2システム間で発話単位の指標を比較します: 片側対応ありWilcoxon符号順位検定（観測された平均差の向きに応じて片側の方向を決定）、対応ありCohen's d、対応ありCliff's delta（いずれもブートストラップ信頼区間付き）、および対応差の平均に対するブートストラップ信頼区間（任意）。使用例（Test Set 1、UTMOSv2、baseline vs. UtterTune）:

```bash
python -m scripts.eval.compare_scores_2systems \
    --input_csv utmos_easy_wide.csv \
    --systems cv2_base cv2_uttertune \
    --bootstrap --n_iter 1000 --ci 95
```

`utmos_easy_wide.csv`には、システムごとに1列（例: `cv2_base`、`cv2_uttertune`）でその発話単位のスコアを持たせ、発話ごとに行を揃えておく必要があります。

### `compare_scores_3systems.py` — 3システム比較＋多重比較補正

3システム間で発話単位の指標を比較します: 全体検定としてのFriedman検定、3ペアすべてに対する対応あり片側Wilcoxon検定（観測された平均差の向きに応じて片側の方向を決定し、`compare_scores_2systems.py`および論文の記述する手法と整合させています）、それらペアワイズp値に対するHolm-Bonferroni補正（論文のTable 1脚注が言及する「Bonferroni correction」のステップダウン版であり、単純なBonferroni補正と同等以上の検出力を持ちつつ同じ家族単位有意水準を制御します）、さらに2システム版と同様の効果量・ブートストラップ報告を行います。使用例（Test Set 2、CER、baseline / kana-input / UtterTuneの3-way比較）:

```bash
python -m scripts.eval.compare_scores_3systems \
    --input_csv cer_difficult_wide.csv \
    --systems cv2_base cv2_base_kana cv2_uttertune \
    --bootstrap --n_iter 1000 --ci 95
```

`cer_difficult_wide.csv`には、システムごとに1列（例: `cv2_base`、`cv2_base_kana`、`cv2_uttertune`）でその発話単位のCERを持たせ、発話ごとに行を揃えておく必要があります。

### モジュール形式での実行

このリポジトリの`scripts/cv2/*`と同じスタイルで`python -m scripts.eval.<script>`として実行するには、`scripts/eval/`がインポート可能なパッケージである必要があります。そのため、`scripts/cv2/__init__.py`にならって空の`scripts/eval/__init__.py`を同梱しています。

## アクセント正答率（スクリプト化なし）

Table 1の「Accent」列は、論文の著者が合成サンプルを聴取して行った手動・主観的な判定です（論文の「Evaluation Metrics」節を参照）。スクリプトでは計算されておらず、ここにもそのためのスクリプトは提供していません。

## 必要なパッケージ

指標算出スクリプト（`calc_cer.py`、`calc_cosine_similarity_eres2net.py`、`calc_utmosv2.py`）には以下が必要です:

- `Levenshtein`
- `jaconv`
- `pandas`
- `modelscope`
- `soundfile`
- `scipy`
- `torch`
- `utmosv2`
- `tqdm`

統計スクリプト（`compare_scores_2systems.py`、`compare_scores_3systems.py`）にはさらに以下が必要です:

- `numpy`
- `pandas`（上記と重複）
- `scipy`（上記と重複）
- `statsmodels`（`compare_scores_3systems.py`のみ、Holm-Bonferroni補正 `statsmodels.stats.multitest.multipletests` のため）

これらはすべてリポジトリルートの`eval`extra（`requirements-eval.txt`）に記載されています。`pip install -e ".[eval]"`でインストールしてください（トップレベルREADMEのセットアップ手順を参照）。

**バージョン指定について**: `requirements-eval.txt`は下限（`>=`）のみを指定しており、厳密なバージョン固定ではありません。これらは開発中に組み合わせて動作確認できたバージョンを反映したものであり、arXiv v1（2025年8月）の元の実験環境そのものとは必ずしも一致しません。この評価パイプラインの各ステップは複数の独立した仮想環境で実行されており、共通依存パッケージ（例: numpy 1.26〜2.4、torch 2.3〜2.11）のバージョンがステップごとに異なっていました。単一環境への厳密な固定は、この再現性の主張の精度を過大に見せることになるため、あえて行っていません。特定のバージョンの組み合わせで問題が発生した場合は、開発中に動作確認された下限以上のバージョンであれば動作するはずです。
