import json
import os
from typing import Any, Dict, Optional

from agent.dqn.dqn_agent import DQN_Agent
from agent.utils.model_version_utils import ModelVersionPaths


class MetaInfoSaver:
    """
    モデルのメタ情報をJSON形式で保存および読み込みし、
    DQN_Agent インスタンスに反映するユーティリティクラス。
    """

    def save(
        self,
        agent: DQN_Agent,
        version_paths: ModelVersionPaths,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        DQN_Agent インスタンスからメタ情報を取得し、
        バージョンディレクトリ内の meta/ サブディレクトリに
        JSONファイルとして保存する。

        Args:
            agent (DQN_Agent): 保存対象のエージェントインスタンス
            version_paths (ModelVersionPaths): 保存先バージョンディレクトリ情報
            extra_metadata (Optional[Dict[str, Any]]): 追記したい任意メタ情報

        Returns:
            str: 保存されたメタ情報ファイルのパス
        """
        # エージェントからメタ情報を取得
        metadata: Dict[str, Any] = agent.get_metadata().copy()
        if extra_metadata:
            metadata.update(extra_metadata)

        # バージョンディレクトリとベースファイル名を生成
        meta_dir = os.path.join(version_paths.version_dir, "meta")
        os.makedirs(meta_dir, exist_ok=True)
        # meta サブディレクトリ
        path = os.path.join(meta_dir, f"{version_paths.base_filename}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        return path

    def load(self, agent: DQN_Agent, version_dir: str, base_filename: str) -> DQN_Agent:
        """
        バージョンディレクトリ内の meta/ サブディレクトリから
        JSONファイルを読み込み、DQN_Agent インスタンスの
        属性にメタ情報を反映する。

        Args:
            agent (DQN_Agent): メタ情報を反映するエージェントインスタンス
            version_dir (str): バージョンディレクトリのパス
            base_filename (str): ファイル名のベース文字列

        Returns:
            DQN_Agent: メタ情報を反映したエージェント
        """
        # メタファイルのパス
        path = os.path.join(version_dir, "meta", f"{base_filename}.json")
        with open(path, "r", encoding="utf-8") as f:
            metadata: Dict[str, Any] = json.load(f)

        # エージェント属性に反映
        agent.board_side = metadata.get("board_side", agent.board_side)
        agent.reward_line = metadata.get("reward_line", agent.reward_line)
        agent.learning_count = metadata.get("learning_count", agent.learning_count)
        agent.GAMMA = metadata.get("gamma", agent.GAMMA)
        agent.LEARNING_RATE = metadata.get("learning_rate", agent.LEARNING_RATE)
        agent.EPSILON_START = metadata.get("epsilon_start", agent.EPSILON_START)
        agent.EPSILON_MIN = metadata.get("epsilon_min", agent.EPSILON_MIN)
        agent.EPSILON_DECAY = metadata.get("epsilon_decay", agent.EPSILON_DECAY)
        agent.BATCH_SIZE = metadata.get("batch_size", agent.BATCH_SIZE)
        agent.BUFFER_SIZE = metadata.get("buffer_size", agent.BUFFER_SIZE)
        agent.REACH_CREATED_REWARD = metadata.get(
            "reach_created_reward", agent.REACH_CREATED_REWARD
        )
        agent.REACH_BLOCKED_REWARD = metadata.get(
            "reach_blocked_reward", agent.REACH_BLOCKED_REWARD
        )

        return agent
