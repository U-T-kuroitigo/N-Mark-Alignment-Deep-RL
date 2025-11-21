"""
評価用エージェントを生成するためのファクトリ。
XAI フック付きのラッパを差し替えやすくするため、ロード処理を 1 箇所に集約する。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from agent.model.N_Mark_Alignment_agent_model import Agent_Model
from saver.agent_loader import load_agent_model


class AgentFactory:
    """
    保存済みモデルから評価用エージェントを生成するファクトリ。
    """

    def __init__(self, device: Optional[torch.device] = None) -> None:
        """
        ファクトリを初期化する。

        Args:
            device (Optional[torch.device]): 生成するエージェントに使用するデバイス。
        """

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def build(
        self,
        model_path: Path,
        board_side: int,
        reward_line: int,
        num_team_values: int,
        *,
        player_icon: str,
        player_value: int,
    ) -> Agent_Model:
        """
        モデルパスと盤面情報からエージェントを生成する。

        Args:
            model_path (Path): モデルファイルのパス。
            board_side (int): 盤面サイズ。
            reward_line (int): 勝利ライン数。
            num_team_values (int): チーム数。
            player_icon (str): 使用するアイコン。
            player_value (int): チーム値。

        Returns:
            Agent_Model: 復元されたエージェント。
        """

        agent = load_agent_model(
            filepath=str(model_path),
            board_side=board_side,
            reward_line=reward_line,
            num_team_values=num_team_values,
            player_icon=player_icon,
            player_value=player_value,
            learning=False,
            device=self.device,
        )
        return agent
