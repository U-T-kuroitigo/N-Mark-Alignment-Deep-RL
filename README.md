# N-Mark Alignment Deep Reinforcement Learning

多人数対応の N 目並べ（任意サイズの盤面で N 個揃えるゲーム）を Deep Q-Network（DQN）で強化学習するプロジェクトです。  
将来的には XAI（説明可能な AI）手法を組み込み、学習過程や意思決定を可視化・説明できるエージェントを目指しています。

## プロジェクトの狙い

- **XAI を意識した N 目並べ AI**  
  勝敗だけでなく、中間評価や行動根拠を説明できるエージェントを開発する。
- **Self-Play を軸にした学習フロー**  
  現状は自己対戦による学習（Self-Play）と DQN ベースの更新が中心。
- **学習・評価・追加学習をワークフロー化**  
  学習済みモデルの保存形式、評価結果の記録、追加学習といった一連の運用を統一。

## 現在の状態（2025/07 時点）

- Self-Play による初期学習 (`train/train_dqn.py`) が実装済み。
- DQN エージェント（1 チャンネル盤面 + チーム ID ベクトル）を `agent/dqn/dqn_agent.py` に実装。
- 報酬計算や中間評価を `agent/utils/reward_utils.py` で管理。
- モデル保存・メタ情報・リプレイバッファは `saver/dqn_agent_saver/` でまとめて管理。
- 学習結果や評価結果は CSV に記録し、後から可視化可能。

## ディレクトリ構成（抜粋）

| ディレクトリ / ファイル | 役割 |
|-------------------------|------|
| `agent/` | エージェント本体、DQN ネットワーク、ReplayBuffer、ユーティリティ |
| `agent_model/` | 学習済みモデル・メタデータ・リプレイバッファの保存先（Git 管理外） |
| `evaluate/` | ラウンドロビン評価や可視化補助のスクリプト群 |
| `saver/dqn_agent_saver/` | `.pt` / `.json` / `.pkl` など学習成果物の入出力クラス |
| `train/` | 学習スクリプト (`train_dqn.py`)、追加学習 (`train_dqn_finetune.py`) |
| `tests/` | 学習ログの可視化・ONNX エクスポートなど開発補助ツール |
| `N_Mark_Alignment_env.py` | 盤面管理・ターン制御・勝敗判定など環境ロジック |

## 環境構築

Conda を利用する前提です。Windows CPU 環境向けに `environment.yml` を同梱しています。

```bash
conda env create -f environment.yml
conda activate Deep_RL_XAI_310
```

GPU で学習する場合は `environment.yml` の `cpuonly` を外し、`pytorch` チャンネルの GPU パッケージ（例：`pytorch-cuda=12.1`）を追加してください。

## 使い方

### 学習

```bash
python -m train.train_dqn
```

盤面サイズや報酬設定、エピソード数などは `train/train_dqn.py` 冒頭の定数で調整できます。学習済みモデルとメタ情報は `agent_model/` 以下に保存されます。

### 追加学習（ファインチューニング）

```bash
python -m train.train_dqn_finetune
```

`train/train_dqn_finetune.py` の `TARGET_MODEL_PATH` と `PLAYER_SETTINGS` を編集し、追学習させたいモデルや対戦相手（モデル / 同一モデル / NPC）を指定してください。

### 評価

複数モデルを総当たりで対戦させる例:

```bash
python -m evaluate.evaluate_models
```

評価対象や対戦設定は `evaluate/evaluate_models.py` の `get_model_list()` で調整します。

### 開発補助スクリプト

`tests/` には学習ログの可視化や ONNX エクスポートなどのユーティリティが揃っています。

- `tests/plot_loss_rate.py` : 評価履歴（CSV）から敗北率の推移を描画。
- `tests/plot_win_minus_lose.py` : 勝率と敗北率の差を可視化。
- `tests/export_qnetwork_onnx.py` : QNetwork を ONNX 形式で出力。

## 学習成果物とログ

### モデル保存先

```
agent_model/
 └── {モデルタイプ}_{盤面サイズ}_{N列揃え}_{プレイヤー数}/
     ├── {モデルタイプ}_{盤面サイズ}_{N}_{プレイヤー数}_{学習数}_{タイムスタンプ}.pt
     ├── meta/
     │    └── {モデルタイプ}_{...}.json      # メタデータ
     ├── replay/
     │    └── {モデルタイプ}_{...}.pkl       # リプレイバッファ（任意）
     └── evaluation_history.csv             # 学習・評価履歴
```

評価履歴 CSV には `episode`, `win_rate`, `lose_rate`, `draw_rate`, `timestamp` などが記録されます。これを元に `tests/plot_*` で可視化できます。

### 評価ログ

`evaluate/result/`（Git 管理外）配下に、評価対象ごとの CSV を保存できます。`tests/plot_win_minus_lose.py` などで参照します。

## 今後の展望

- 学習結果をもとに XAI ライブラリ（例: SHAP など）と連携し、意思決定根拠を可視化する。
- matplotlib 等を用いたログビューアの整備。
- 追加学習や評価を自動化する設定ファイル／スクリプトの拡充。

## 開発メモ

開発者向けの詳細メモは `docs/internal/` 配下に保管しています（Git 管理外）。README には主要な使い方とガイドのみを記載しています。
