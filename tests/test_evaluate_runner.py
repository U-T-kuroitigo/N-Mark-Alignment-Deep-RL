"""
RoundRobinMatchRunner のスモークテスト。

少数エージェントを用いた対戦が正しくサマリ化され、評価結果 CSV が出力されるか確認する。
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import N_Mark_Alignment_env as env_module  # noqa: E402
from agent.model.N_Mark_Alignment_agent_model import Agent_Model  # noqa: E402
from evaluate.round_robin_match_runner import RoundRobinMatchRunner  # noqa: E402


class DummyAgent(Agent_Model):
    """固定パターンで着手する単純なテスト用エージェント。"""

    def __init__(self, player_icon: str, player_value: int, moves: list[int]) -> None:
        super().__init__(player_icon, player_value)
        self.moves = moves
        self.idx = 0
        self.player_name = f"Dummy-{player_icon}"
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
        self, action: int, state: list[int], actor_team_value: int, next_team_value: int
    ) -> None:
        pass

    def append_finish_result(self, action: int, state: list[int], result_value: int) -> None:
        self.game_count += 1
        if result_value == self.my_team_value:
            self.win += 1
        elif result_value == self.empty_value:
            self.draw += 1
        else:
            self.lose += 1

    def get_learning_count(self) -> int:
        return 0


def test_round_robin_runner_creates_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    RoundRobinMatchRunner で 2 エージェントの総当たり戦を実行し、
    結果サマリと CSV ファイルが生成されるか確認する。
    """
    agent_a = DummyAgent("A", 0, moves=[0, 4, 8])
    agent_b = DummyAgent("B", 1, moves=[1, 2, 5])

    env = env_module.N_Mark_Alignment_Env(board_side=3, reward_line=3, player_list=[agent_a, agent_b])
    runner = RoundRobinMatchRunner(env, eval_episodes=2)

    monkeypatch.chdir(tmp_path)
    summary = runner.evaluate([agent_a, agent_b])

    assert len(summary) == 2
    assert {entry["agent_name"] for entry in summary} == {agent_a.player_name, agent_b.player_name}

    csv_path = Path("evaluate") / "result" / "3_3_2" / "evaluation_history.csv"
    assert csv_path.exists()
