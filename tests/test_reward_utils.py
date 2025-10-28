"""
reward_utils モジュールの挙動を検証するテスト。

reach 作成・阻止・中間報酬計算など、主要関数の戻り値が期待通りかを確認する。
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.utils import reward_utils  # noqa: E402


def test_is_reach_created_detects_new_reach() -> None:
    """一手で reach を作れる場合に True を返すことを確認する。"""
    board_side = 3
    reward_line = 3
    team_value = 0
    state = [
        team_value,
        -1,
        team_value,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
    ]
    action = 4  # 中央に打って縦方向に 2 個揃っている状況を完成させる

    assert reward_utils.is_reach_created(
        state, action, team_value, board_side, reward_line
    )


def test_is_reach_blocked_detects_blocking() -> None:
    """相手の reach を阻止できる場合に True を返すことを確認する。"""
    board_side = 3
    reward_line = 3
    team_value = 0

    opponent_state = [1, 1, -1, -1, -1, -1, -1, -1, -1]
    action = 2

    assert reward_utils.is_reach_blocked(
        opponent_state, action, team_value, board_side, reward_line
    )


def test_intermediate_reward_combines_created_and_blocked() -> None:
    """reach 作成と阻止が同時に起こるケースでの中間報酬を確認する。"""
    board_side = 3
    reward_line = 3
    team_value = 0
    state_before = [
        1,
        1,
        -1,
        -1,
        team_value,
        -1,
        -1,
        -1,
        -1,
    ]
    action = 2

    reward = reward_utils.calculate_intermediate_reward(
        state_before,
        action,
        team_value,
        board_side,
        reward_line,
        reach_created_reward=0.4,
        reach_blocked_reward=0.6,
    )

    # reach 作成と阻止の両方が発生して合計される
    assert pytest.approx(reward) == pytest.approx(1.0)


def test_normalize_intermediate_rewards_handles_zero_sum() -> None:
    """中間報酬の合計が 0 の場合に全て 0.0 が返ることを確認する。"""
    rewards = [0.0, 0.0, 0.0]
    normalized = reward_utils.normalize_intermediate_rewards(rewards)
    assert normalized == [0.0, 0.0, 0.0]


def test_normalize_intermediate_rewards_scales_values() -> None:
    """中間報酬の合計が正の値の場合に、合計が 1.0 になるようスケーリングされることを確認する。"""
    rewards = [1.0, 2.0, 1.0]
    normalized = reward_utils.normalize_intermediate_rewards(rewards)
    assert pytest.approx(sum(normalized)) == 1.0
    assert normalized == pytest.approx([0.25, 0.5, 0.25])


def test_find_two_step_reach_positions_returns_all_candidates() -> None:
    """4×4 盤で二手リーチが 2 か所存在するケースを検証する。"""
    board_side = 4
    reward_line = 4
    team_value = 1
    state = [
        team_value,
        team_value,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
    ]

    positions = reward_utils.find_two_step_reach_positions(
        state, team_value, board_side, reward_line
    )
    assert positions == {2, 3}


def test_find_direct_reach_positions_detects_diagonal_reach() -> None:
    """斜め方向のリーチ位置が正しく検出されることを確認する。"""
    board_side = 3
    reward_line = 3
    team_value = 2
    state = [
        team_value,
        -1,
        -1,
        -1,
        team_value,
        -1,
        -1,
        -1,
        -1,
    ]

    positions = reward_utils.find_direct_reach_positions(
        state, team_value, board_side, reward_line
    )
    assert positions == {8}
