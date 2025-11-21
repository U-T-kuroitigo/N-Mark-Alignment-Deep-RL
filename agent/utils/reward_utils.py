"""
報酬評価ユーティリティ。
中間報酬（リーチ、阻止など）および最終報酬（勝敗）を評価する関数群を提供。
DQNや他の強化学習モデルから共通して呼び出すことを想定。
"""

from functools import lru_cache
from typing import List, Set, Tuple


EMPTY_VALUE: int = -1
"""盤面上の空きマスを表す値（環境全体で固定の -1 ）。"""

_DIRECTION_VECTORS: Tuple[Tuple[int, int], ...] = (
    (0, 1),  # 右方向（横）
    (1, 0),  # 下方向（縦）
    (1, 1),  # 右下方向（↘）
    (1, -1),  # 左下方向（↙）
)
"""リーチ判定に用いる方向ベクトルの一覧。"""


@lru_cache(maxsize=None)
def _generate_all_lines(
    board_side: int, reward_line: int
) -> Tuple[Tuple[int, ...], ...]:
    """
    盤面サイズと勝利条件に基づき、評価対象となる直線（インデックス列）を生成する。

    Args:
        board_side (int): 盤面の一辺サイズ。
        reward_line (int): 勝利に必要な連続数。

    Returns:
        Tuple[Tuple[int, ...], ...]: 各ラインを構成する一次元インデックスのタプル一覧。
    """
    if board_side <= 0 or reward_line <= 0:
        return tuple()

    lines: List[Tuple[int, ...]] = []
    board_limit = board_side - 1

    for row in range(board_side):
        for col in range(board_side):
            for delta_row, delta_col in _DIRECTION_VECTORS:
                # 勝利条件分だけ進んだ先が盤面内かを事前に確認
                last_row = row + delta_row * (reward_line - 1)
                last_col = col + delta_col * (reward_line - 1)
                if not (0 <= last_row <= board_limit and 0 <= last_col <= board_limit):
                    continue

                indices: List[int] = []
                for step in range(reward_line):
                    current_row = row + delta_row * step
                    current_col = col + delta_col * step
                    # ステップごとに座標を進め、一次元インデックスへ変換
                    indices.append(current_row * board_side + current_col)
                lines.append(tuple(indices))

    return tuple(lines)


def _extract_line_values(state: List[int], indices: Tuple[int, ...]) -> List[int]:
    """
    指定されたインデックス列から盤面の値をまとめて取得する。

    Args:
        state (List[int]): 盤面状態（一次元リスト）。
        indices (Tuple[int, ...]): 取得対象となるインデックス列。

    Returns:
        List[int]: 対象インデックスに対応する盤面の値を並べたリスト。
    """
    return [state[idx] for idx in indices]


def is_win(result_value: int, my_team_value: int) -> bool:
    """
    指定の勝者が自分自身であるかを判定。

    Args:
        result_value (int): 勝者のチーム値
        my_team_value (int): 自身のチーム値

    Returns:
        bool: 自分が勝者であれば True
    """
    return result_value == my_team_value


def is_draw(result_value: int, empty_value: int) -> bool:
    """
    引き分けかどうかを判定。

    Args:
        result_value (int): 勝者のチーム値または引き分け時の空き値
        empty_value (int): 空きマスを表す値

    Returns:
        bool: 引き分けなら True
    """
    return result_value == empty_value


def is_loss(result_value: int, my_team_value: int, empty_value: int) -> bool:
    """
    自身が負けたかどうかを判定。

    Args:
        result_value (int): 勝者のチーム値または引き分け時の空き値
        my_team_value (int): 自身のチーム値
        empty_value (int): 空きマスを表す値

    Returns:
        bool: 自身が負けていれば True
    """
    return not is_win(result_value, my_team_value) and not is_draw(
        result_value, empty_value
    )


def find_two_step_reach_positions(
    state: List[int], team_value: int, board_side: int, reward_line: int
) -> Set[int]:
    """
    指定された盤面状態から、二手リーチ（N-2連結＋空き2）の可能性がある空きマスの位置を検出する。

    二手リーチとは、直線方向に team_value が (N-2) 個、
    空きマス（-1）が2個だけ存在する並びであり、
    今後の2手以内でリーチ（または勝利）に繋がる可能性のある箇所を示す。

    横・縦・右下斜め・左下斜めの4方向すべてを対象とし、
    行をまたいで曲がっている（盤面上では直線でない）ケースを除外する。

    Args:
        state (List[int]): 現在の盤面状態（一次元リスト）
        team_value (int): 判定対象のチーム値
        board_side (int): 盤面の一辺サイズ（例: 5 → 5×5）
        reward_line (int): 勝利に必要な連続数（例: 3目、5目）

    Returns:
        Set[int]: 二手リーチに該当する空きマスのインデックス集合
    """
    reach_positions: Set[int] = set()

    for line in _generate_all_lines(board_side, reward_line):
        values = _extract_line_values(state, line)
        if (
            values.count(team_value) == reward_line - 2
            and values.count(EMPTY_VALUE) == 2
        ):
            for idx, value in zip(line, values):
                if value == EMPTY_VALUE:
                    reach_positions.add(idx)

    return reach_positions


def is_reach_created(
    state_before: List[int],
    action: int,
    team_value: int,
    board_side: int,
    reward_line: int,
) -> bool:
    """
    指定の行動がリーチ（N-1連結）を生み出したかどうかを判定。
    ※現時点では未実装、今後拡張予定。

    Args:
        state_before (List[int]): 行動前の盤面状態
        action (int): 実行した行動インデックス
        team_value (int): 実行プレイヤーのチーム値
        board_side (int): 盤面の一辺のサイズ
        reward_line (int): 勝利に必要な連続数

    Returns:
        bool: リーチが作られたと判定されれば True
    """
    two_step_positions = find_two_step_reach_positions(
        state_before, team_value, board_side, reward_line
    )
    return action in two_step_positions


def find_direct_reach_positions(
    state: List[int], team_value: int, board_side: int, reward_line: int
) -> Set[int]:
    """
    指定された盤面状態から、リーチ（N-1連結＋空き1）の可能性がある空きマスの位置を検出する。

    リーチとは、直線方向に team_value が (N-1) 個、空きマス（-1）が1個だけ存在する並びであり、
    次の1手で勝利される危険性がある箇所を示す。
    横・縦・右下斜め・左下斜めの4方向すべてを対象とし、
    行をまたいで曲がっている（盤面上では直線でない）ケースを除外する。

    Args:
        state (List[int]): 現在の盤面状態（一次元リスト）
        team_value (int): 判定対象のチーム値
        board_side (int): 盤面の一辺サイズ（例: 5 → 5×5）
        reward_line (int): 勝利に必要な連続数（例: 3目、5目）

    Returns:
        Set[int]: リーチに該当する空きマスのインデックス集合
    """
    reach_positions: Set[int] = set()

    for line in _generate_all_lines(board_side, reward_line):
        values = _extract_line_values(state, line)
        if (
            values.count(team_value) == reward_line - 1
            and values.count(EMPTY_VALUE) == 1
        ):
            empty_index = values.index(EMPTY_VALUE)
            reach_positions.add(line[empty_index])

    return reach_positions


def is_reach_blocked(
    state_before: List[int],
    action: int,
    team_value: int,
    board_side: int,
    reward_line: int,
) -> bool:
    """
    指定の行動が他者のリーチを阻止したかどうかを判定する。
    盤面上に存在する敵チームのリーチ位置をすべて取得し、
    今回の行動がそのいずれかに該当すれば「阻止成功」とみなす。

    Args:
        state_before (List[int]): 行動前の盤面状態
        action (int): 実行した行動インデックス
        team_value (int): 実行プレイヤーのチーム値
        board_side (int): 盤面の一辺のサイズ
        reward_line (int): 勝利に必要な連続数

    Returns:
        bool: 阻止成功なら True
    """
    opponent_teams = set(state_before) - {team_value, EMPTY_VALUE}
    all_reach_positions: Set[int] = set()

    for opp in opponent_teams:
        opp_reaches = find_direct_reach_positions(
            state_before, opp, board_side, reward_line
        )
        all_reach_positions |= opp_reaches

    return action in all_reach_positions


def calculate_intermediate_reward(
    state_before: List[int],
    action: int,
    team_value: int,
    board_side: int,
    reward_line: int,
    reach_created_reward: float = 0.5,
    reach_blocked_reward: float = 0.5,
) -> float:
    """
    与えられた状態と行動に対して、中間報酬を計算する。

    Args:
        state_before (List[int]): 行動前の盤面状態
        action (int): 実行した行動のインデックス
        team_value (int): 評価対象のチーム値（視点）
        board_side (int): 盤面の一辺の長さ
        reward_line (int): 勝利条件の連数
        reach_created_reward (float): リーチ生成時の報酬
        reach_blocked_reward (float): リーチ阻止時の報酬

    Returns:
        float: 中間報酬（0.0〜1.0）
    """
    intermediate_reward: float = 0.0
    if is_reach_created(state_before, action, team_value, board_side, reward_line):
        intermediate_reward += reach_created_reward
    if is_reach_blocked(state_before, action, team_value, board_side, reward_line):
        intermediate_reward += reach_blocked_reward
    return intermediate_reward


def normalize_intermediate_rewards(intermediate_rewards: List[float]) -> List[float]:
    """
    中間報酬リストを合計1.0に正規化する。

    Args:
        intermediate_rewards (List[float]): 各ステップでの中間報酬

    Returns:
        List[float]: 正規化された中間報酬リスト
    """
    total: float = sum(intermediate_rewards)
    if total == 0:
        return [0.0] * len(intermediate_rewards)
    return [r / total for r in intermediate_rewards]
