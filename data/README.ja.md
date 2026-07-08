# UtterTune データ公開

*[English](README.md) | 日本語*

このディレクトリには、[UtterTune論文](https://arxiv.org/abs/2508.09767)で使用した学習マニフェストと評価用テスト文セットが含まれます。音声データや音声由来の特徴量（波形、音声トークン）は一切含まれません。詳細は下記のLICENSE節を参照してください。

## `train/jsutjvs_manifest.tsv`

出典情報付きの学習マニフェストで、学習元リポジトリの`data/manifests/all.tsv`から派生させたものです。著者が実験に使用したマニフェストに存在した音声トークン（`.npy`）と波形（`.wav`）へのパス列は、それらのファイルが再配布不可のため削除しています（LICENSE節参照）。

列構成（タブ区切り、ヘッダー行あり）:

| 列 | 説明 |
|----------|-------------|
| `text`   | `<PHON_START>...<PHON_END>`で注釈されたスパンを含む文テキスト。アポストロフィ（`'`）はアクセント核、スラッシュ（`/`）はタグ内のアクセント句境界を表します。 |
| `corpus` | `jsut`または`jvs` |
| `subset` | JSUTの場合`basic5000`または`voiceactress100`、JVSの場合`parallel100` |
| `speaker`| JSUTの場合は空、JVSの場合`jvsNNN`（例: `jvs001`） |
| `file_id`| 元発話ID（例: `BASIC5000_0001`、`VOICEACTRESS100_001`） |

行数: 15,097件（JSUT basic5000が5,000件、JSUT voiceactress100が100件、JVS parallel100が9,997件）。

`corpus`、`subset`、`speaker`、`file_id`と、正規に入手したJSUT・JVSの音声データがあれば、対応する波形ファイルを以下のように再構成できます:

- JSUT: `data/jsut/wav/{file_id}.wav`
- JVS: `data/jvs/{speaker}/wav/{file_id}.wav`

**このmanifestには学習/検証分割は含まれていません。**
`scripts/cv2/train.py`が学習実行時に`torch.utils.data.random_split`でランダム分割を行います。分割は学習設定の`training.seed`（デフォルト42）と`val_ratio`（デフォルト0.05）でシード化されているため、同じmanifestと同じ学習設定を使えば同じ分割が再現されます。

## `eval/`

論文のリスニングテスト（Table 1および§5.2）で使用した評価用テスト文セットです。各ファイルはヘッダー行と50件のデータ行（`id` 1〜50、1始まり）を持ち、条件間で使うテキストのバリエーションを列として横並びにしています。

### `eval/easy_50.tsv` — Test Set 1（一般的な自然性評価、§5.1）

| 列 | 説明 |
|---------------|-------------|
| `id`          | 1〜50 |
| `vanilla`     | 通常の文テキスト — ベースライン条件 |
| `phon_tagged` | 同じ文で1単語を`<PHON_START>...<PHON_END>`で囲んだもの — 提案手法 (UtterTune) 条件 |

### `eval/difficult_50.tsv` — Test Set 2（発音・アクセント制御性のストレステスト、§5.1）

各行に難読語を1つずつ含みます。

| 列 | 説明 |
|-----------------|-------------|
| `id`            | 1〜50 |
| `vanilla`       | 難読語を漢字・通常表記のまま含む文テキスト — ベースライン条件 |
| `kana_baseline` | 難読語をアクセント記号なしのカナ読みのみに置換した文 — カナ入力ベースライン条件 |
| `phon_tagged`   | 難読語のかな読みにアクセント核（`'`）・句境界（`/`）を付与し`<PHON_START>...<PHON_END>`で囲んだもの — 提案手法 (UtterTune) 条件 |

### `eval/katakana_50.tsv` — カタカナ・リークテスト（§5.2）

アクセント核を持つカタカナ語を1つずつ含む、追加の50文です。LoRAがタグで囲まれていない部分に影響を与えないことを確認するために使用します。この検証の性質上、ベースライン・提案手法の両方で同一の（タグなし）テキストを使用します。

| 列 | 説明 |
|--------|-------------|
| `id`   | 1〜50 |
| `text` | 通常の文テキスト |

## ライセンス

学習マニフェストのテキストは、[JSUTコーパス](https://sites.google.com/site/shinnosuketakamichi/publication/jsut)および[JVSコーパス](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus)に由来します。

JSUT自身のLICENCE.txtによれば、テキストのライセンスはサブセットごとに異なります。`basic5000`のテキストはWikipedia（[CC-BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/deed)）、Tanaka Corpus（[CC-BY 2.0](https://creativecommons.org/licenses/by/2.0/deed)）、およびJSUT独自の文（[CC-BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/deed)）の混在です。`voiceactress100`のテキストはVoice Actress Corpus由来でCC-BY-SA 4.0です。JVSのテキストはJSUT由来であり、JSUTのライセンスに従います。

JSUTの規約（「音声データの再配布は認められていません」）に従い、JSUT/JVS由来の音声・音声由来データ（波形、音声トークン）はここには一切含まれていません。

このプロジェクトが元のテキストに付加した発音・アクセント注釈（`<PHON_START>`/`<PHON_END>`タグ、アクセント核・句境界の記号）はオリジナルの注釈作業です。元テキストの一部がCC-BY-SAの義務を負っているため、注釈付きmanifest全体を**CC BY-SA 4.0**で公開します。これはJSUT自身の開示方式（サブセットごとのライセンス注記）に倣ったものであり、CC-BY-SAでライセンスされた部分の継承 (ShareAlike) 条項を満たします（CC-BY-SA 3.0およびCC-BY-SA 4.0はCreative Commonsの公式な互換ライセンス一覧に含まれ、派生物の再ライセンスが可能です。CC-BY 2.0の部分はShareAlike制約を課さないため、デフォルトで互換です）。

このマニフェストおよび評価データの利用者は、学習音声を再構成したい場合、JSUT・JVSの音声を各コーパス自身の規約のもとで別途正規に入手する必要があります。ここで提供しているのはテキスト＋アクセント注釈＋出典識別子のみです。

評価用テスト文セット（`easy_50`、`difficult_50`、`katakana_50`）は、著者がChatGPT o3による生成支援と手動レビューを経て作成したもの（論文のEvaluation Data節を参照）であり、著者によるオリジナルコンテンツとして、同様にCC BY-SA 4.0で公開します。
