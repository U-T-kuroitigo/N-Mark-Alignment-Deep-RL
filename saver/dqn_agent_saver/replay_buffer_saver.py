import os
import pickle
from typing import Dict, Any

from agent.buffer.replay_buffer import ReplayBuffer
from agent.dqn.dqn_agent import DQN_Agent
from agent.utils.model_version_utils import ModelVersionPaths


class ReplayBufferSaver:
    """
    DQN_Agent のリプレイバッファを保存および読み込みするクラス。
    ModelSaver / MetaInfoSaver と同様に、agent インスタンスを渡すだけで完結します。
    """

    def __init__(self, filename: str = "replay_buffer"):
        """
        Args:
            filename (str): 保存ファイルのベース名（拡張子除く）
        """
        self.filename = filename

    def save(
        self,
        agent: DQN_Agent,
        version_paths: ModelVersionPaths,
    ) -> str:
        """
        DQN_Agent インスタンスから ReplayBuffer を取得し、
        pickle 形式で保存する。

        Args:
            agent (DQN_Agent): 保存対象のエージェントインスタンス
            version_paths (ModelVersionPaths): 保存先バージョンディレクトリ情報

        Returns:
            str: 保存された ReplayBuffer ファイルのパス
        """
        replay_dir = os.path.join(version_paths.version_dir, "replay")
        os.makedirs(replay_dir, exist_ok=True)
        replay_buffer: ReplayBuffer = getattr(agent, "replay_buffer", None)
        path = os.path.join(replay_dir, f"{version_paths.base_filename}.pkl")
        with open(path, "wb") as f:
            pickle.dump(replay_buffer.buffer, f)
        return path

    def load(
        self, agent: DQN_Agent, version_dir: str, base_filename: str
    ) -> ReplayBuffer:
        """
        バージョンディレクトリの replay サブディレクトリから
        pickle ファイルを読み込み、エージェントの ReplayBuffer に復元する。

        Args:
            agent (DQN_Agent): 復元対象のエージェントインスタンス
            version_dir (str): バージョンディレクトリのパス
            base_filename (str): ファイル名のベース文字列

        Returns:
            ReplayBuffer: 復元されたエージェントの ReplayBuffer
        """
        replay_buffer: ReplayBuffer = agent.replay_buffer
        path = os.path.join(version_dir, "replay", f"{base_filename}.pkl")
        with open(path, "rb") as f:
            data = pickle.load(f)
        replay_buffer.clear()
        for transition in data:
            replay_buffer.append(transition)
        return replay_buffer
