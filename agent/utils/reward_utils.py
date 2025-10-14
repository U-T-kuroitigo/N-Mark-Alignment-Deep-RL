"""
報酬評価ユーティリティ。
中間報酬（リーチ、阻止など）および最終報酬（勝敗）を評価する関数群を提供。
DQNや他の強化学習モデルから共通して呼び出すことを想定。
"""

from typing import List, Set


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
    directions = [1, board_side, board_side + 1, board_side - 1]  # 横・縦・↘・↙

    def is_valid_index(idx: int) -> bool:
        return 0 <= idx < board_side * board_side

    for start in range(board_side * board_side):
        for d in directions:
            indices = [start + i * d for i in range(reward_line)]
            if not all(is_valid_index(idx) for idx in indices):
                continue

            base_row = start // board_side
            rows = [idx // board_side for idx in indices]

            # 横方向（→）
            if d == 1 and any(r != base_row for r in rows):
                continue

            # 縦・斜め方向共通チェック
            if d in (board_side, board_side + 1, board_side - 1) and rows != [
                base_row + i for i in range(reward_line)
            ]:
                continue

            values = [state[idx] for idx in indices]
            if values.count(team_value) == reward_line - 2 and values.count(-1) == 2:
                for i, v in zip(indices, values):
                    if v == -1:
                        reach_positions.add(i)

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
    directions = [1, board_side, board_side + 1, board_side - 1]  # 横・縦・↘・↙

    def is_valid_index(idx: int) -> bool:
        return 0 <= idx < board_side * board_side

    for start in range(board_side * board_side):
        for d in directions:
            indices = [start + i * d for i in range(reward_line)]
            if not all(is_valid_index(idx) for idx in indices):
                continue

            base_row = start // board_side
            rows = [idx // board_side for idx in indices]

            # 横方向（→）
            if d == 1 and any(r != base_row for r in rows):
                continue
            # 縦・斜め方向共通チェック（すべて下方向に一段ずつ下がる）
            if d in (board_side, board_side + 1, board_side - 1) and rows != [
                base_row + i for i in range(reward_line)
            ]:
                continue

            # 値チェック：N-1個が判定対象のチーム値、1個が空き
            values = [state[idx] for idx in indices]
            if values.count(team_value) == reward_line - 1 and values.count(-1) == 1:
                reach_positions.add(indices[values.index(-1)])

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
    opponent_teams = set(state_before) - {team_value, -1}
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
