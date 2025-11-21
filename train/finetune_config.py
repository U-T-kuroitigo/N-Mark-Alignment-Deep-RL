"""
finetune_config.py

追加学習用スクリプトの設定読み込みを担当するモジュール。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

DEFAULT_BASE_MODEL_PATH = r"agent_model\DQN-Agent_3_3_2\DQN-Agent_3_3_2_10_202510141639\DQN-Agent_3_3_2_10_202510141639.pt"


@dataclass
class FinetunePlayerSetting:
    """
    追加学習に参加するプレイヤー設定を表すデータクラス。

    Attributes:
        type (str): プレイヤー種別（base / model / self / npc）。
        icon (str): 表示用のアイコン文字。
        learning (bool): 学習モードにするかどうか。
        path (Optional[str]): 参照モデルのパス。type により必須となる。
        load_replay (bool): リプレイバッファを読み込むかどうか。
        player_value (Optional[int]): チーム値。未指定なら自動で割り当てる。
    """

    type: str
    icon: str
    learning: bool = False
    path: Optional[str] = None
    load_replay: bool = False
    player_value: Optional[int] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FinetunePlayerSetting":
        """
        辞書から FinetunePlayerSetting を生成する。
        """
        return cls(
            type=data["type"],
            icon=data.get("icon", "?"),
            learning=data.get("learning", False),
            path=data.get("path"),
            load_replay=data.get("load_replay", False),
            player_value=data.get("player_value"),
        )


def _default_player_settings() -> List[FinetunePlayerSetting]:
    """
    デフォルトのプレイヤー構成を生成する。
    """
    return [
        FinetunePlayerSetting(type="base", icon="A", learning=True, load_replay=False),
        FinetunePlayerSetting(type="npc", icon="B", learning=False),
    ]


@dataclass
class FinetuneConfig:
    """
    追加学習スクリプト全体の設定値をまとめたデータクラス。
    """

    base_model_path: str = DEFAULT_BASE_MODEL_PATH
    player_settings: List[FinetunePlayerSetting] = field(
        default_factory=_default_player_settings
    )
    learning_count: int = 100
    learn_iterations_per_episode: int = 1
    save_frequency: Optional[int] = None
    log_frequency: Optional[int] = None
    eval_episodes: int = 100
    device_type: str = "auto"
    reset_epsilon: bool = True
    log_level: str = "INFO"
    config_dir: Optional[Path] = field(default=None, repr=False)

    @classmethod
    def from_yaml(cls, path: Path) -> "FinetuneConfig":
        """
        YAML ファイルから FinetuneConfig を生成する。

        Args:
            path (Path): 設定ファイルのパス。

        Returns:
            FinetuneConfig: 生成された設定インスタンス。
        """
        if path.suffix.lower() not in {".yaml", ".yml"}:
            raise ValueError("追加学習設定は YAML (.yaml / .yml) を使用してください。")
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(
                "設定ファイルの形式が不正です。最上位は辞書である必要があります。"
            )

        player_settings_data = data.get("player_settings", [])
        if not player_settings_data:
            raise ValueError("player_settings が空です。最低 1 件は指定してください。")
        player_settings = [
            FinetunePlayerSetting.from_dict(entry) for entry in player_settings_data
        ]

        if player_settings[0].type != "base":
            raise ValueError("player_settings の先頭は type='base' にしてください。")

        base_model_path = data.get("base_model_path", DEFAULT_BASE_MODEL_PATH)
        config = cls(
            base_model_path=base_model_path,
            player_settings=player_settings,
            learning_count=data.get("learning_count", 100),
            learn_iterations_per_episode=data.get("learn_iterations_per_episode", 1),
            save_frequency=data.get("save_frequency"),
            log_frequency=data.get("log_frequency"),
            eval_episodes=data.get("eval_episodes", 100),
            device_type=data.get("device_type", "auto"),
            reset_epsilon=data.get("reset_epsilon", True),
            log_level=data.get("log_level", "INFO"),
        )
        config.config_dir = path.parent
        return config

    def trainer_config(self) -> Dict[str, int]:
        """
        Trainer に渡す設定辞書を生成する。

        Returns:
            Dict[str, int]: Trainer 用設定。
        """
        save_freq = self.save_frequency or max(1, self.learning_count // 20)
        log_freq = self.log_frequency or max(1, self.learning_count // 20)
        return {
            "total_episodes": self.learning_count,
            "learn_iterations_per_episode": self.learn_iterations_per_episode,
            "save_frequency": save_freq,
            "log_frequency": log_freq,
        }

    def resolve_path(self, raw_path: Optional[str]) -> Path:
        """
        設定ファイルに記載されたパスを絶対パスへ解決する。

        Args:
            raw_path (Optional[str]): 設定値として指定されたパス。

        Returns:
            Path: 解決済みの絶対パス。
        """
        target = Path(raw_path or self.base_model_path)
        if target.is_absolute():
            return target
        # まず設定ファイルのディレクトリを起点として解決し、存在すれば採用する
        if self.config_dir:
            candidate = (self.config_dir / target).resolve()
            if candidate.exists():
                return candidate
        # 存在しない場合はリポジトリルート（カレントディレクトリ）を基準に解決する
        return (Path.cwd() / target).resolve()
