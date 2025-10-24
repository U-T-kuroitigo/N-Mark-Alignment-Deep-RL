"""
環境クラス (`N_Mark_Alignment_Env`) の基本挙動を検証するテスト。

盤面更新・ターン切り替え・勝敗判定が期待通りに動くかを最小構成で確認する。
"""

import sys
from pathlib import Path
from typing import List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import N_Mark_Alignment_env as env_module  # noqa: E402
from agent.model.N_Mark_Alignment_agent_model import Agent_Model  # noqa: E402


class FixedAgent(Agent_Model):
    """あらかじめ決めた手順で指すテスト用エージェント。"""

    def __init__(self, player_icon: str, player_value: int, moves: List[int]) -> None:
        """
        Args:
            player_icon: 盤面表示用アイコン。
            player_value: チーム ID。
            moves: 着手するマスのインデックス列。
        """
        super().__init__(player_icon, player_value)
        self.moves = moves
        self.index = 0
        self.player_name = f"Fixed-{player_icon}"
        self.learning = False

    def set_learning(self, learning: bool) -> None:
        """テスト用エージェントなので学習は行わない。"""
        self.learning = learning

    def get_action(self, env: env_module.N_Mark_Alignment_Env) -> int:
        """順番にマスを返し、尽きたら空きマスを探す。"""
        if self.index < len(self.moves):
            action = self.moves[self.index]
            self.index += 1
        else:
            board = env.get_board()
            action = next(i for i, v in enumerate(board) if v == self.empty_value)
        self.prev_action = action
        return action

    def append_continue_result(
        self, action: int, state: List[int], actor_team_value: int, next_team_value: int
    ) -> None:
        """継続結果は利用しない。"""

    def append_finish_result(self, action: int, state: List[int], result_value: int) -> None:
        """勝敗カウントのみ更新。"""
        self.game_count += 1
        if result_value == self.my_team_value:
            self.win += 1
        elif result_value == self.empty_value:
            self.draw += 1
        else:
            self.lose += 1

    def get_learning_count(self) -> int:
        """学習を行わないため常に 0。"""
        return 0


def test_step_updates_board_and_turn() -> None:
    """step() で盤面が更新され、ターンが入れ替わることを確認する。"""
    agent_a = FixedAgent("A", 0, moves=[0])
    agent_b = FixedAgent("B", 1, moves=[4])

    env = env_module.N_Mark_Alignment_Env(board_side=3, reward_line=3, player_list=[agent_a, agent_b])
    env.reset()
    env.player_turn_list = [agent_a, agent_b]  # 順番を固定する

    action, prev_board, next_board, actor_team_value, next_team_value, done, _ = env.step()

    assert action == 0
    assert prev_board[action] == env.EMPTY  # 手前は空きマス
    assert next_board[action] == actor_team_value  # 手番のチーム値が入る
    assert next_team_value == agent_b.get_my_team_value()
    assert done is False


def test_auto_play_finishes_with_winner() -> None:
    """あらかじめ勝てる手順を与えた場合に auto_play が勝敗を返すか確認する。"""
    # A が斜めに 3 連を作る、B は別マスに着手する設定
    agent_a = FixedAgent("A", 0, moves=[0, 4, 8])
    agent_b = FixedAgent("B", 1, moves=[1, 2, 5])

    env = env_module.N_Mark_Alignment_Env(board_side=3, reward_line=3, player_list=[agent_a, agent_b])
    env.reset_game_rate()

    board, _ = env.auto_play()

    assert agent_a.win == 1
    assert agent_b.lose == 1
    # 斜め 0,4,8 のマスが A のチーム値になっていることを確認
    assert [board[i] for i in (0, 4, 8)] == [agent_a.get_my_team_value()] * 3
