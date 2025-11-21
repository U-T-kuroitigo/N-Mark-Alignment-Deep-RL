"""
training_config.py

トレーニングスクリプト用の設定読み込み処理をまとめたモジュール。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import yaml


@dataclass
class TrainingConfig:
    """
    DQN トレーニングスクリプトの設定値を格納するデータクラス。
    """

    board_side: int = 3
    reward_line: int = 3
    num_team_values: int = 2
    total_episodes: int = 100
    learn_iterations_per_episode: int = 1
    save_frequency: int = 5
    log_frequency: int = 5
    eval_episodes: int = 50
    log_level: str = "INFO"

    @classmethod
    def from_yaml(cls, path: Path) -> "TrainingConfig":
        """
        YAML ファイルから設定を読み込んで TrainingConfig を生成する。

        Args:
            path (Path): 設定ファイルのパス。

        Returns:
            TrainingConfig: 生成された設定インスタンス。
        """
        if path.suffix.lower() not in {".yaml", ".yml"}:
            raise ValueError(
                "トレーニング設定は YAML (.yaml / .yml) を使用してください。"
            )
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(
                "設定ファイルの形式が不正です。最上位は辞書である必要があります。"
            )
        return cls(**data)

    def to_trainer_dict(self) -> Dict[str, int]:
        """
        Trainer に渡す設定辞書を作成する。

        Returns:
            Dict[str, int]: Trainer 用の設定辞書。
        """
        return {
            "total_episodes": self.total_episodes,
            "learn_iterations_per_episode": self.learn_iterations_per_episode,
            "save_frequency": self.save_frequency,
            "log_frequency": self.log_frequency,
        }
