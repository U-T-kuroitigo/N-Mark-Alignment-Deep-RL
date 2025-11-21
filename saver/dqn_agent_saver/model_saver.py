import os
import csv
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import torch

from agent.utils.model_version_utils import (
    make_version_dir_and_filename,
    ModelVersionPaths,
)
from saver.dqn_agent_saver.meta_info_saver import MetaInfoSaver
from saver.dqn_agent_saver.replay_buffer_saver import ReplayBufferSaver
from agent.dqn.dqn_agent import DQN_Agent


@dataclass(frozen=True)
class ModelSaveArtifacts:
    """
    モデル保存処理で生成された成果物へのパスをまとめたデータコンテナ。

    Attributes:
        model_path (str): 保存されたモデル重みファイル (.pt) のパス。
        eval_result_path (Optional[str]): 勝率などを記録したテキストのパス。
        eval_history_path (Optional[str]): CSV形式の評価履歴ファイルのパス。
        metadata_path (Optional[str]): メタ情報 JSON のパス。
        replay_buffer_path (Optional[str]): リプレイバッファ pickle のパス。
        version_paths (ModelVersionPaths): バージョンディレクトリ情報。
    """

    model_path: str
    eval_result_path: Optional[str]
    eval_history_path: Optional[str]
    metadata_path: Optional[str]
    replay_buffer_path: Optional[str]
    version_paths: ModelVersionPaths


class ModelSaver:
    """
    モデルの保存および関連情報（メタデータ、リプレイバッファなど）の保存を管理するクラス。
    各モデルタイプ毎にバージョンディレクトリを生成し、
    重みファイル、メタ情報、リプレイバッファを一貫して管理します。
    """

    def __init__(self, save_dir: str = "agent_model"):
        """
        保存先のルートディレクトリを初期化する。

        Args:
            save_dir (str): モデルを保存するルートディレクトリのパス
        """
        self.save_dir = save_dir
        self.meta_saver = MetaInfoSaver()
        self.replay_saver = ReplayBufferSaver()

    def save(
        self,
        agent: DQN_Agent,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> ModelSaveArtifacts:
        """
        DQN_Agentインスタンスからモデル、メタデータ、リプレイバッファを保存する。
        モデルタイプのディレクトリ配下にバージョンディレクトリを作成し、
        ファイルを格納します。

        Args:
            agent (DQN_Agent): 保存対象のエージェントインスタンス
            extra_metadata (Optional[Dict[str, Any]]): メタ情報に追記する追加項目。

        Returns:
            ModelSaveArtifacts: 保存処理で生成されたファイルパス群。
        """
        # エージェントから必要な情報を取得
        policy_net = agent.policy_net
        metadata = agent.get_metadata()
        # バージョンディレクトリおよびファイル名を生成
        version_paths = make_version_dir_and_filename(metadata, self.save_dir)

        # ディレクトリを事前生成
        os.makedirs(version_paths.version_dir, exist_ok=True)

        # モデル重み (.pt) の保存
        model_path = os.path.join(
            version_paths.version_dir, f"{version_paths.base_filename}.pt"
        )
        torch.save(policy_net.state_dict(), model_path)

        win, lose, draw = agent.get_rate()
        # 学習回数を取得（agent側で保持していることを想定）
        learning_count = getattr(agent, "learning_count", -1)
        eval_result_path, eval_history_path = self._write_evaluation_outputs(
            version_paths, (win, lose, draw), learning_count
        )

        # メタ情報の保存
        metadata_path = self.meta_saver.save(
            agent, version_paths, extra_metadata=extra_metadata
        )

        # ReplayBuffer の保存があれば実行
        replay_path = None
        if getattr(agent, "replay_buffer", None) is not None:
            replay_path = self.replay_saver.save(agent, version_paths)

        return ModelSaveArtifacts(
            model_path=model_path,
            eval_result_path=eval_result_path,
            eval_history_path=eval_history_path,
            metadata_path=metadata_path,
            replay_buffer_path=replay_path,
            version_paths=version_paths,
        )

    def load(
        self,
        model_path: str,
        agent: DQN_Agent,
        load_replay: bool = False,
    ) -> DQN_Agent:
        """
        保存されたモデルとメタデータを読み込み、
        DQN_Agentインスタンスに復元する。

        Args:
            model_path (str): 保存されたモデルファイルのパス
            agent (DQN_Agent): 学習済みモデルを読み込む DQN_Agent インスタンス
            load_replay (bool, optional): True の場合、リプレイバッファを復元する

        Returns:
            DQN_Agent: モデルおよび関連情報を復元したエージェントインスタンス
        """
        # ネットワーク重みを読み込む
        agent.policy_net.load_state_dict(torch.load(model_path))

        # バージョンディレクトリと base_filename を取得
        version_dir = os.path.dirname(model_path)
        base_filename = os.path.splitext(os.path.basename(model_path))[0]

        # メタデータを読み込み、エージェントに反映
        self.meta_saver.load(agent, version_dir, base_filename)

        # ReplayBufferを必要に応じて復元
        if load_replay:
            agent.replay_buffer = self.replay_saver.load(
                agent, version_dir, base_filename
            )

        return agent

    def _write_evaluation_outputs(
        self,
        version_paths: ModelVersionPaths,
        rates: Tuple[float, float, float],
        learning_count: int,
    ) -> Tuple[str, str]:
        """
        評価結果テキストと履歴CSVを出力する。

        Args:
            version_paths (ModelVersionPaths): 保存先バージョンディレクトリ情報。
            rates (Tuple[float, float, float]): (win_rate, lose_rate, draw_rate) の順で指定。
            learning_count (int): 学習ステップ数またはエピソード数。

        Returns:
            Tuple[str, str]: テキストファイルと CSV ファイルのパス。
        """
        win, lose, draw = rates

        eval_result_path = os.path.join(
            version_paths.version_dir, f"{version_paths.base_filename}_eval_result.txt"
        )
        # ランダムNPCとの評価結果を保存（勝率, 負率, 引き分け率）
        with open(eval_result_path, "w", encoding="utf-8") as f:
            f.write(
                "ランダムNPC評価結果\n"
                f"勝率: {win:.1f}%\n"
                f"負率: {lose:.1f}%\n"
                f"引き分け率: {draw:.1f}%\n"
            )

        eval_history_path = os.path.join(
            version_paths.model_type_dir, "evaluation_history.csv"
        )
        os.makedirs(version_paths.model_type_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        write_header = not os.path.exists(eval_history_path)
        # 評価履歴（CSV）をバージョンディレクトリに追記保存
        with open(eval_history_path, mode="a", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(
                csvfile,
                fieldnames=[
                    "filename",
                    "episode",
                    "win_rate",
                    "lose_rate",
                    "draw_rate",
                    "timestamp",
                ],
            )
            if write_header:
                writer.writeheader()
            writer.writerow(
                {
                    "filename": version_paths.base_filename,
                    "episode": learning_count,
                    "win_rate": f"{win:.1f}",
                    "lose_rate": f"{lose:.1f}",
                    "draw_rate": f"{draw:.1f}",
                    "timestamp": timestamp,
                }
            )

        return eval_result_path, eval_history_path
