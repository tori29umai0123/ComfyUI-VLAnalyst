# モデル比較workflowの組み方

同じLoad Image出力を、モデルごとに独立したLoader → Analyzeへ接続します。ただしキャッシュ上限は1つなので、ComfyUIの実行順を依存リンクで直列化してください。最初のAnalyzeの`run_token`をRelease Modelへ渡し、その出力を次段の実行順制御に使う補助ノードを挟むか、各Analyzeを個別にキュー実行するのが確実です。

比較条件は以下を固定してください。

- 入力画像と`max_long_edge`
- system/user prompt
- `do_sample=false`、max tokens、seed
- dtypeとattention実装
- 動画では抽出済みフレーム番号と時刻（各メタデータで確認）

対応する比較対象はQwen3.5-4B、Qwen3.5-9B、GLM-4.6V-Flashです。
