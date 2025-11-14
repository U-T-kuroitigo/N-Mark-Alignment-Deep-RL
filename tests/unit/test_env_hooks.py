import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from N_Mark_Alignment_env import (  # noqa: E402  pylint: disable=wrong-import-position
    AFTER_STEP_HOOK,
    BEFORE_STEP_HOOK,
    N_Mark_Alignment_Env,
)
from agent.model.N_Mark_Alignment_agent_model import (  # noqa: E402  pylint: disable=wrong-import-position
    Agent_Model,
)


class DummyAgent(Agent_Model):
    """テスト用に最小限の手続きを実装したダミーエージェント。"""

    def __init__(self, icon: str, player_value: str) -> None:
        super().__init__(icon, player_value)
        self.learning = False

    def set_learning(self, learning):
        self.learning = learning

    def get_action(self, env):
        # 先頭の空きマスをそのまま選ぶ
        for idx, value in enumerate(env.board):
            if value == env.EMPTY:
                return idx
        raise RuntimeError("no legal moves")

    def append_continue_result(self, action, state, actor_team_value, next_team_value):
        pass

    def append_finish_result(self, action, state, result_value):
        pass

    def get_metadata(self) -> dict:
        return {}

    def get_learning_count(self):
        return 0


def make_env(board_side: int = 3, reward_line: int = 3) -> N_Mark_Alignment_Env:
    agents = [DummyAgent("A", "p1"), DummyAgent("B", "p2")]
    return N_Mark_Alignment_Env(board_side, reward_line, agents)


def test_after_step_hook_and_history_capture_context():
    env = make_env()
    captured = []

    env.register_hook(AFTER_STEP_HOOK, lambda ctx: captured.append(ctx.copy()))

    result = env.step()

    assert len(captured) == 1
    ctx = captured[0]

    # hook で受け取った内容と step_history の先頭が一致すること
    history = env.get_step_history()
    assert history and history[0] == ctx

    # 主要フィールドが含まれており、値が整合すること
    assert ctx["action"] == result[0]
    assert ctx["board_before"][ctx["action"]] == env.EMPTY
    assert ctx["board_after"][ctx["action"]] == ctx["actor_team_value"]
    assert ctx["next_team_value"] == result[4]


def test_before_step_hook_receives_board_snapshot():
    env = make_env()
    before_payloads = []

    env.register_hook(BEFORE_STEP_HOOK, before_payloads.append)

    action, *_ = env.step()

    assert len(before_payloads) == 1
    payload = before_payloads[0]

    # BEFORE_STEP_HOOK の board_before は行動前の状態のまま保持されていること
    assert payload["board_before"][action] == env.EMPTY
    assert payload["turn_index"] == 0
    # 実際の盤面は行動後に更新済みになっている
    assert env.board[action] != env.EMPTY
