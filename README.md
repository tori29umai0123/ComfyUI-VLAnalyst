# ComfyUI-VLAnalyst

## 対応モデル

次の公式VLモデルに対応します。

| サイズ | モデル |
| --- | --- |
| 4B | `Qwen/Qwen3.5-4B` |
| 9B | `Qwen/Qwen3.5-9B` |
| 9B | `zai-org/GLM-4.6V-Flash` |
| 14B | `OpenGVLab/InternVL3_5-14B-HF` |

量子化モデルやGGUFには対応していません。Loaderには上表の4モデルだけが常に表示され、選択した
モデルが `models/LLM` にない場合だけHugging Faceから自動ダウンロードします。
非公式モデルや任意のローカルモデルは表示しません。

InternVL3.5-14Bは[公式のHF形式](https://huggingface.co/OpenGVLab/InternVL3_5-14B-HF)を使用します。[元のGitHub形式](https://huggingface.co/OpenGVLab/InternVL3_5-14B)は`model_type=internvl_chat`で、このLoaderにはそのまま読み込めません。InternVLの動画解析では、選択したフレームを時刻付きの画像列として渡します。

VLモデルをローカル実行し、ComfyUIの画像またはデコード済み動画フレームを分析するカスタムノードです。UI表示時には通信せず、未配置の公式モデルをLoaderで実際に選択・実行したときだけダウンロードします。

## ノード

- **VL Analyst Loader**: 対応VLモデルをロードし、1モデルだけキャッシュします。
- **VL Analyst Image**: `images`に接続したComfyUI `IMAGE`バッチを静止画として分析します。複数枚はバッチ順に`Picture 1`、`Picture 2`…として区別します。
- **VL Analyst Video**: 時系列の`IMAGE`バッチと元FPSを受け、指定区間をサンプリングして映像として渡します。音声は解析しません。
- **VL Analyst Pose Prompt**: 機械生成された関節記述と、任意の視点ヒント・ユーザー補足から、画像と一緒にVLモデルへ渡すポーズ分析用の英語プロンプトを生成します。
- **VL Analyst Release Model**: この拡張が所有するモデルだけを解放します。

すべて `VL Analyst` カテゴリに表示されます。既存workflow用の内部ノードIDは維持します。

## 確認済み環境

2026-09-26に、ノード登録とprocessorの画像・動画前処理経路を次の環境で確認しました。

| 項目 | 状態 |
| --- | --- |
| OS / GPU | Windows / NVIDIA GeForce RTX 5090 32GB |
| Python | 3.11.9 |
| PyTorch / CUDA | 2.13.0+cu130 / CUDA 13.0、GPU認識済み |
| Transformers | 5.5.3 |
| huggingface-hub | 1.33.0 |
| accelerate | 1.12.0 |
| `Qwen/Qwen3.5-4B` processor | 画像・動画テンソル生成経路を確認 |
| `Qwen/Qwen3.5-4B` 画像推論 | 対応 |
| `Qwen/Qwen3.5-4B` 動画推論 | 対応 |
| Qwen3.5-9B | モデル未配置、未実測 |
| GLM-4.6V-Flash | モデル未配置、未実測 |
| InternVL3.5-14B-HF | モデル未配置、未実測 |

「ノード登録」「前処理」「モデル推論」を区別しています。スモークテストでは画像ロード約4.60秒・推論約0.95秒、2フレーム動画推論約0.92秒でした。これは白黒の小画像と最大16生成トークンによる経路確認値で、性能指標ではありません。未実測の組合せを対応済みとは扱いません。

## モデル配置

画像対応Transformersモデル一式を、次のように配置します。

```text
C:\ComfyUI\models\LLM\Qwen3.5-4B\
  config.json
  tokenizer_config.json
  preprocessor_config.json
  model*.safetensors
  ...
```

GLM-4.6V-Flashは`C:\ComfyUI\models\LLM\GLM-4.6V-Flash\`へ同様に配置します。
InternVL3.5-14B-HFは`C:\ComfyUI\models\LLM\InternVL3_5-14B-HF\`へ配置します。

サブフォルダも探索します。フォルダ名ではなく`config.json`の`model_type`と`architectures`を確認します。初期版は以下に限定しています。

- `model_type`: `qwen3_5`、`glm4v` または `internvl`
- architecture: `ForConditionalGeneration`
- Transformers形式の非量子化モデル（BF16を推奨）

量子化形式はエラーにします。手動配置も可能で、未配置の場合は選択した公式モデルだけを自動取得します。

## 導入

このフォルダを`C:\ComfyUI\custom_nodes\ComfyUI-VLAnalyst`として置き、ComfyUIを再起動します。

既存環境が確認済みバージョンを満たすなら、追加インストールは不要です。不足時だけ、状態を記録してからComfyUIのPythonで導入します。

```powershell
C:\ComfyUI\.venv\Scripts\python.exe -m pip freeze > before-qia-install.txt
C:\ComfyUI\.venv\Scripts\python.exe -m pip install -r C:\ComfyUI\custom_nodes\ComfyUI-VLAnalyst\requirements.txt
```

`--upgrade`は付けないでください。本環境にはVibeVoice等との既知の依存競合があるため、この拡張はそれらを一括修復しません。復旧時は`before-qia-install.txt`を参照し、この拡張のために変更したパッケージだけを元へ戻してください。

## 使い方

1. Loaderでモデル、dtype、attention実装を選択します。標準経路では`bfloat16`、`auto`から始めます。
2. 静止画はAnalyzeの`images`へ接続します。複数枚をまとめて解析する場合は`IMAGE`バッチにして接続し、プロンプトでは`Picture 1`、`Picture 2`…で指定します。動画は動画ローダー等でデコードした時系列`IMAGE`バッチをAnalyze Videoへ接続し、正しい`source_fps`を指定します。
3. 同一モデルで続けるなら`keep_mode=gpu`、VRAMを空けて再ロードを避けるなら`cpu`、完全解放なら`unload`を使います。
4. 後続の画像生成前に、`cpu`／`unload`またはRelease Modelを使うことを推奨します。

ポーズだけを定型形式で分析する場合は、VL Analyst Pose Promptの`user_prompt`出力をVL Analyst Imageの同名入力へ接続します。`automatic_joint_description`は必須です。視点ヒントと補足が空白なら、それぞれの見出しや補足向けの指示は生成プロンプトから完全に省かれます。ユーザー補足は翻訳・要約せず原文のまま挿入されます。

`max_long_edge=0`では外部リサイズを行わずprocessorへ渡します。正値では縦横比を保って縮小し、クロップや反転はしません。processor内部の正規化・リサイズはモデル設定に従います。

動画入力は既にデコードされたフレーム列です。`sample_fps`でフレームを選び、上限超過時は`uniform`で全区間から均等削減、または`error`で停止します。既に抽出したフレームをprocessorへ渡すため、processor側のFPS再サンプリングを明示的に無効化し、元FPSと元フレーム番号をtimestamp token生成用metadataとして渡します。可変フレームレートの正確なPTS入力は初期版では未対応です。

## Thinkingと生成

chat templateへ必ず`enable_thinking=False`を渡します。GLMでは公式テンプレートがこれを`/nothink`指示へ変換します。入力トークンを除いた新規生成分だけをデコードします。

`do_sample=false`ではtemperature、top_p、top_kを`generate()`へ渡しません。JSON modeはプロンプト指定と`json.loads`による構文検証であり、制約デコードではありません。不正JSONも元出力と検証エラーを保持し、自動修正しません。

## キャッシュとメモリ

キャッシュ上限は1モデルです。キーは正規化パス、dtype、attention実装です。別モデルへ切り替えると以前のモデルを解放してからロードします。モデルとprocessorだけを保持し、入力画像、動画、生成テンソルは保持しません。

- `gpu`: GPU上で再利用します。ComfyUIの全モデル管理へ強制統合するものではありません。
- `cpu`: `model.to("cpu")`でRAMへ退避します。CPU推論ではありません。
- `unload`: 所有参照を外してGC後にCUDAキャッシュを整理します。

CUDAコンテキストや他ノードの割当が残るため、解放後のVRAMがゼロになる保証はありません。OOM時に精度、モデル、フレーム数を勝手に変更しません。

## 出力メタデータ

run ID、日時、入力hash、画像サイズ、動画時刻・フレーム番号、モデル、プロンプト、生成条件、Thinking設定、JSON検証、ロード／前処理／推論時間、出力tokens/sec、PyTorchプロセス全体のCUDA統計、実行環境を記録します。CUDA統計はモデル専用VRAMではありません。

## Workflow

`workflows/`には次を同梱しています。

- `minimal_image_api.json`: Loader → 画像分析のAPI workflow例
- `minimal_video_api.json`: デコード済みIMAGEバッチ → 動画分析のAPI workflow例
- `model_comparison.md`: 同じ入力を複数モデルへ順番に渡す構成と測定上の注意

API JSONの画像ローダー名やモデル選択値は、手元のファイル名へ置き換えてください。ComfyUI UIのバージョン差を避けるため、例はAPI形式です。

## 制約

- GPU推論専用です。CPU推論、FlashAttention必須化、ネイティブ動画型、ローカル動画ファイルのデコード、音声理解、チャンク統合は未実装です。
- `source_fps`方式のためVFR動画の正確なPTSは保持できません。VFRでは事前にCFR化するか、今後のPTS対応アダプタを使用してください。
- 同じseedでもGPU、ドライバ、カーネル差による完全一致は保証しません。
- 画像内の文字は分析対象であり、system/user命令として扱わないよう既定system promptへ明記しています。
