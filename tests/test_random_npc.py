"""
ランダム NPC エージェントの挙動を検証するテスト。

空きマスのみを選択すること、残り 1 マスでも例外なく動作することを確認する。
"""

import random
import sys
from pathlib import Path
from typing import List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import N_Mark_Alignment_env as env_module  # noqa: E402
from agent.N_Mark_Alignment_random_npc import N_Mark_Alignment_random_npc  # noqa: E402
from agent.model.N_Mark_Alignment_agent_model import Agent_Model  # noqa: E402


class FixedAgent(Agent_Model):
    """決められた順序で着手する簡易エージェント。"""

    def __init__(self, player_icon: str, player_value: int, moves: List[int]) -> None:
        super().__init__(player_icon, player_value)
        self.moves = moves
        self.idx = 0
        self.player_name = f"Fixed-{player_icon}"
        self.learning = False

    def set_learning(self, learning: bool) -> None:
        self.learning = learning

    def get_action(self, env: env_module.N_Mark_Alignment_Env) -> int:
        if self.idx < len(self.moves):
            action = self.moves[self.idx]
            self.idx += 1
        else:
            board = env.get_board()
            action = next(i for i, v in enumerate(board) if v == self.empty_value)
        self.prev_action = action
        return action

    def append_continue_result(
        self, action: int, state: List[int], actor_team_value: int, next_team_value: int
    ) -> None:
        pass

    def append_finish_result(
        self, action: int, state: List[int], result_value: int
    ) -> None:
        self.game_count += 1

    def get_learning_count(self) -> int:
        return 0


def test_random_npc_picks_only_empty_cells() -> None:
    """複数回の呼び出しで常に空きマスが選択されることを確認する。"""
    random.seed(0)
    agent_a = FixedAgent("A", 0, moves=[0, 4, 8])
    npc = N_Mark_Alignment_random_npc("B", 1)

    env = env_module.N_Mark_Alignment_Env(
        board_side=3, reward_line=3, player_list=[agent_a, npc]
    )
    env.reset()
    # いくつかのマスを埋める
    env.board[0] = agent_a.get_my_team_value()
    env.board[4] = agent_a.get_my_team_value()
    env.board[5] = npc.get_my_team_value()

    for _ in range(20):
        action = npc.get_action(env)
        assert env.board[action] == env.EMPTY


def test_random_npc_selects_only_remaining_cell() -> None:
    """残り 1 マスの状況でそのマスを返すか確認する。"""
    agent_a = FixedAgent("A", 0, moves=[0])
    npc = N_Mark_Alignment_random_npc("B", 1)

    env = env_module.N_Mark_Alignment_Env(
        board_side=3, reward_line=3, player_list=[agent_a, npc]
    )
    env.reset()
    env.board[:] = [
        agent_a.get_my_team_value(),
        agent_a.get_my_team_value(),
        env.EMPTY,
        npc.get_my_team_value(),
        agent_a.get_my_team_value(),
        npc.get_my_team_value(),
        agent_a.get_my_team_value(),
        npc.get_my_team_value(),
        agent_a.get_my_team_value(),
    ]

    action = npc.get_action(env)
    assert action == 2
