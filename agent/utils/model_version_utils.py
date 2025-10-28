import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Union


@dataclass(frozen=True)
class ModelVersionPaths:
    """
    モデル保存に関するディレクトリ情報をまとめたコンテナ。

    Attributes:
        model_type_dir (str): モデル種別ごとのルートディレクトリ。
        version_dir (str): 今回の保存に利用するバージョンディレクトリ。
        base_filename (str): 保存ファイル名のベース文字列。
        timestamp (str): バージョン生成に利用したタイムスタンプ文字列（YYYYMMDDHHMM）。
    """

    model_type_dir: str
    version_dir: str
    base_filename: str
    timestamp: str


def make_version_dir_and_filename(
    metadata: dict,
    root_dir: str = "agent_model",
    timestamp: Optional[Union[str, datetime]] = None,
) -> ModelVersionPaths:
    """
    メタデータをもとにモデルタイプディレクトリとバージョンディレクトリを生成し、
    保存に利用するベースファイル名とともに返す。

    Args:
        metadata (dict): モデル情報を含む辞書
            必須キー: "agent_name", "board_side", "reward_line", "team_count", "learning_count"
        root_dir (str): 保存ルートディレクトリ (デフォルト: "agent_model")
        timestamp (Optional[Union[str, datetime]]): タイムスタンプ指定。
            省略時は現在時刻から `YYYYMMDDHHMM` 形式を生成する。

    Returns:
        ModelVersionPaths: モデル保存に必要なディレクトリ情報のセット。

    Example:
        metadata = {
            "agent_name": "DQN_Agent",
            "board_side": 9,
            "reward_line": 5,
            "team_count": 2,
            "learning_count": 100000,
        }
        paths = make_version_dir_and_filename(metadata)
        print(paths.version_dir)
    """
    # メタ情報から必要なフィールドを取得
    agent_name = metadata["agent_name"]
    board_side = metadata["board_side"]
    reward_line = metadata["reward_line"]
    team_count = metadata["team_count"]
    learning_count = metadata["learning_count"]

    # タイムスタンプを生成（YYYYMMDDHHMM）
    if timestamp is None:
        resolved_timestamp = datetime.now().strftime("%Y%m%d%H%M")
    elif isinstance(timestamp, datetime):
        resolved_timestamp = timestamp.strftime("%Y%m%d%H%M")
    else:
        resolved_timestamp = str(timestamp)

    # モデルタイプディレクトリ
    model_type_dir = os.path.join(
        root_dir,
        f"{agent_name}_{board_side}_{reward_line}_{team_count}",
    )
    # バージョン用ファイル名（学習回数＋タイムスタンプ）
    base_filename = (
        f"{agent_name}_{board_side}_{reward_line}_{team_count}_"
        f"{learning_count}_{resolved_timestamp}"
    )
    # バージョンディレクトリ
    version_dir = os.path.join(model_type_dir, base_filename)

    return ModelVersionPaths(
        model_type_dir=model_type_dir,
        version_dir=version_dir,
        base_filename=base_filename,
        timestamp=resolved_timestamp,
    )
