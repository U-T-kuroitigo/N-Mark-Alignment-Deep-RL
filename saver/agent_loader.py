"""
学習済みモデルのロード処理をまとめたユーティリティ。
評価や追加学習など複数モジュールから共通の手順で DQN エージェントを構築できるようにする。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import torch

from agent.dqn.dqn_agent import DQN_Agent
from agent.network.q_network import set_network
from saver.dqn_agent_saver.model_saver import ModelSaver


def _load_metadata_from_model_path(model_path: Path) -> Dict[str, Any]:
    """
    モデルファイルと同じディレクトリに保存されたメタ情報を読み込む。

    Args:
        model_path (Path): モデルファイル (.pt) のパス。

    Returns:
        Dict[str, Any]: メタデータ。存在しない場合は空 dict。
    """

    meta_path = model_path.parent / "meta" / f"{model_path.stem}.json"
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {}


def load_agent_model(
    filepath: str,
    board_side: int,
    reward_line: int,
    num_team_values: int,
    *,
    player_icon: str = "?",
    player_value: int = 0,
    learning: bool = False,
    load_replay: bool = False,
    device: torch.device | None = None,
) -> DQN_Agent:
    """
    指定されたパスから DQN エージェントを復元する。

    Args:
        filepath (str): モデルファイルのパス。
        board_side (int): 盤面サイズ。
        reward_line (int): 勝利条件となるライン数。
        num_team_values (int): チーム数。
        player_icon (str, optional): 復元後に設定するアイコン。デフォルトは "?"。
        player_value (int, optional): 復元後に設定するチーム値。
        learning (bool, optional): 復元後に設定する学習フラグ。
        load_replay (bool, optional): リプレイを読み込むかどうか。
        device (torch.device, optional): 配置デバイス。省略時は CUDA 優先で自動判定。

    Returns:
        DQN_Agent: 復元されたエージェント。
    """

    filepath_path = Path(filepath)
    metadata = _load_metadata_from_model_path(filepath_path)
    board_side = metadata.get("board_side", board_side)
    reward_line = metadata.get("reward_line", reward_line)
    if "team_value_list" in metadata and isinstance(metadata["team_value_list"], list):
        num_team_values = len(metadata["team_value_list"])
    elif "team_list" in metadata and isinstance(metadata["team_list"], list):
        num_team_values = len(metadata["team_list"])
    elif "team_count" in metadata:
        num_team_values = int(metadata["team_count"])

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        state_dict = torch.load(filepath, map_location="cpu")
    except Exception:  # pragma: no cover
        state_dict = None
    else:
        embedding = state_dict.get("team_embedding.weight")
        if embedding is not None:
            inferred = max(embedding.shape[0] - 1, 1)
            num_team_values = inferred

    policy_net, target_net = set_network(board_side, num_team_values, device)

    dummy_agent = DQN_Agent(
        player_icon=player_icon,
        player_value=player_value,
        learning=learning,
        policy_net=policy_net,
        target_net=target_net,
        device=device,
    )

    dummy_agent.reward_line = reward_line
    dummy_agent.board_side = board_side
    if "team_value_list" in metadata and isinstance(metadata["team_value_list"], list):
        dummy_agent.team_value_list = metadata["team_value_list"]
    elif "team_list" in metadata and isinstance(metadata["team_list"], list):
        dummy_agent.team_value_list = metadata["team_list"]
    else:
        dummy_agent.team_value_list = list(range(num_team_values))

    agent = ModelSaver().load(filepath, dummy_agent, load_replay=load_replay)
    agent.set_learning(learning)
    return agent
