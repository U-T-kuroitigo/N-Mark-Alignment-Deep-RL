import os
from datetime import datetime


def make_version_dir_and_filename(
    metadata: dict,
    root_dir: str = "agent_model",
    timestamp: datetime = None,
) -> tuple[str, str]:
    """
    メタデータをもとにモデルタイプとバージョン用ディレクトリおよび
    ベースファイル名を生成するユーティリティ。

    Args:
        metadata (dict): モデル情報を含む辞書
            必須キー: "agent_name", "board_side", "reward_line", "team_count", "learning_count"
        root_dir (str): 保存ルートディレクトリ (デフォルト: "agent_model")

    Returns:
        tuple[str, str]:
            version_dir: バージョンディレクトリのパス
            base_filename: モデルおよびバージョンを表すファイル名ベース

    Example:
        metadata = {
            "agent_name": "DQN_Agent",
            "board_side": 9,
            "reward_line": 5,
            "team_count": 2,
            "learning_count": 100000,
        }
        version_dir, base_fn = make_version_dir_and_filename(metadata)
    """
    # メタ情報から必要なフィールドを取得
    agent_name = metadata["agent_name"]
    board_side = metadata["board_side"]
    reward_line = metadata["reward_line"]
    team_count = metadata["team_count"]
    learning_count = metadata["learning_count"]

    # タイムスタンプを生成（YYYYMMDDHHMM）
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d%H%M")

    # モデルタイプディレクトリ
    model_type_dir = os.path.join(
        root_dir,
        f"{agent_name}_{board_side}_{reward_line}_{team_count}",
    )
    # バージョン用ファイル名（学習回数＋タイムスタンプ）
    base_filename = (
        f"{agent_name}_{board_side}_{reward_line}_{team_count}_"
        f"{learning_count}_{timestamp}"
    )
    # バージョンディレクトリ
    version_dir = os.path.join(model_type_dir, base_filename)

    return model_type_dir, version_dir, base_filename
