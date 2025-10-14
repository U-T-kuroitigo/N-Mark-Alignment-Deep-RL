import numpy as np
import random
from enum import Enum

import copy
import math
import pickle
import random
import agent.model.N_Mark_Alignment_agent_model as agent_model
from typing import Tuple, List, Dict, Any


class N_Mark_Alignment_Env:
    EMPTY = -1

    # -1 is empty
    #
    # board
    # 0 1 2
    # 3 4 5
    # 6 7 8

    def __init__(
        self, board_side, reward_line, player_list: list[agent_model.Agent_Model]
    ):
        if board_side <= 0:
            raise ValueError("board_side must be greater than 0")
        self.BOARD_SIDE = board_side
        self.REWARD_LINE = reward_line
        self.PLAYER_LIST = player_list

        self.create_team_data()
        self.throw_init_data()

        self.shuffle_turn_list()
        self.reset()

    # プレイヤーに自身のチーム値と盤面に存在するチーム情報を与える関数
    def create_team_data(self):
        team_list = {}
        PLAYER_LIST = self.PLAYER_LIST
        for player in PLAYER_LIST:
            player_value = player.get_player_value()
            if player_value not in team_list:
                team_number = len(team_list)
                team_list[player_value] = team_number
        self.team_list = team_list

    # プレイヤーにゲームの情報を流し込む関数
    def throw_init_data(self):
        PLAYER_LIST = self.PLAYER_LIST
        for player in PLAYER_LIST:
            player.init_after_env(self)

    # 盤面をemptyで初期化するための関数
    def reset_board(self):
        self.board = [self.EMPTY] * (self.BOARD_SIDE**2)
        self.board_icon = [self.EMPTY] * (self.BOARD_SIDE**2)

    # 盤面をemptyで初期化するための関数
    def reset_prev(self):
        self.prev_board = [self.EMPTY] * (self.BOARD_SIDE**2)
        self.prev_action = -1

    # ターンをリセットする関数
    def reset_turn(self):
        self.this_turn = 0
        self.elapsed_turn = 0

    # 環境を初期状態にリセットする関数
    def reset(self):
        """
        環境を初期状態にリセットする関数。
        盤面、前の盤面、ターン情報をリセットし、
        プレイヤーに初期化データを提供する。
        """
        self.reset_board()
        self.reset_prev()
        self.reset_turn()

    # 順番をシャッフルする関数
    def shuffle_turn_list(self):
        self.player_turn_list = random.sample(self.PLAYER_LIST, len(self.PLAYER_LIST))

    def set_player(self, player_list: list[agent_model.Agent_Model]):
        """
        プレイヤーのリストを再登録し、チーム情報・初期化情報を更新する関数。
        主に外部からプレイヤー構成を変えて再試行する用途で使用する。

        Args:
            new_player_list (list[Agent_Model]): 新しいプレイヤーインスタンスのリスト
        """
        self.PLAYER_LIST = player_list
        self.create_team_data()
        self.throw_init_data()
        self.shuffle_turn_list()

    # 盤面のサイズを取得する関数
    def get_board_size(self):
        return len(self.board)

    # 盤面を取得する関数
    def get_board(self):
        return self.board

    # 縦方向の勝利条件を満たしているか判定する関数
    def judge_vertical(self, state):
        winner_flag = False
        for col in range(
            0,
            self.BOARD_SIDE * (1 + (self.BOARD_SIDE - self.REWARD_LINE)),
        ):
            if self.EMPTY != state[col]:
                for r_point in range(1, self.REWARD_LINE):
                    if state[col] == state[col + self.BOARD_SIDE * r_point]:
                        winner_flag = True
                    else:
                        winner_flag = False
                        break
                if winner_flag:
                    return winner_flag, state[col]
        return winner_flag, self.EMPTY

    # 横方向の勝利条件を満たしているか判定する関数
    def judge_horizontal(self, state):
        winner_flag = False
        for c_r in range(
            0,
            1 + (self.BOARD_SIDE - self.REWARD_LINE),
        ):
            for row in range(0, self.BOARD_SIDE):
                if self.EMPTY != state[(row * self.BOARD_SIDE) + c_r]:
                    for c_point in range(1, self.REWARD_LINE):
                        if (
                            state[(row * self.BOARD_SIDE) + c_r]
                            == state[((row * self.BOARD_SIDE) + c_r) + c_point]
                        ):
                            winner_flag = True
                        else:
                            winner_flag = False
                            break
                    if winner_flag:
                        return winner_flag, state[(row * self.BOARD_SIDE) + c_r]
        return winner_flag, self.EMPTY

    # 斜め方向の勝利条件のうち、左上から右下にかけて満たしているか判定する関数
    def judge_diagonal_top_left_to_bottom_right(self, state):
        winner_flag = False
        for start_point_col in range(
            0,
            1 + (self.BOARD_SIDE - self.REWARD_LINE),
        ):
            for start_point_row in range(
                0,
                1 + (self.BOARD_SIDE - self.REWARD_LINE),
            ):
                if (
                    self.EMPTY
                    != state[start_point_col + (start_point_row * self.BOARD_SIDE)]
                ):
                    for diagonal_point in range(1, self.REWARD_LINE):
                        if (
                            state[start_point_col + (start_point_row * self.BOARD_SIDE)]
                            == state[
                                (start_point_col + (start_point_row * self.BOARD_SIDE))
                                + ((self.BOARD_SIDE + 1) * diagonal_point)
                            ]
                        ):
                            winner_flag = True
                        else:
                            winner_flag = False
                            break
                    if winner_flag:
                        return (
                            winner_flag,
                            state[
                                start_point_col + (start_point_row * self.BOARD_SIDE)
                            ],
                        )
        return winner_flag, self.EMPTY

    # 斜め方向の勝利条件のうち、右上から左下にかけて満たしているか判定する関数
    def judge_diagonal_top_right_to_bottom_left(self, state):
        winner_flag = False
        for start_point_col in range(
            (self.BOARD_SIDE - 1) - (self.BOARD_SIDE - self.REWARD_LINE),
            self.BOARD_SIDE,
        ):
            for start_point_row in range(
                0,
                1 + (self.BOARD_SIDE - self.REWARD_LINE),
            ):
                if (
                    self.EMPTY
                    != state[start_point_col + (start_point_row * self.BOARD_SIDE)]
                ):
                    for diagonal_point in range(1, self.REWARD_LINE):
                        if (
                            state[start_point_col + (start_point_row * self.BOARD_SIDE)]
                            == state[
                                (start_point_col + (start_point_row * self.BOARD_SIDE))
                                + ((self.BOARD_SIDE - 1) * diagonal_point)
                            ]
                        ):
                            winner_flag = True
                        else:
                            winner_flag = False
                            break
                    if winner_flag:
                        return (
                            winner_flag,
                            state[
                                start_point_col + (start_point_row * self.BOARD_SIDE)
                            ],
                        )
        return winner_flag, self.EMPTY

    # 斜め方向の勝利条件を満たしているか判定する関数
    def judge_diagonal(self, state):
        winner_flag, winner_value = self.judge_diagonal_top_left_to_bottom_right(state)
        if not winner_flag:
            winner_flag, winner_value = self.judge_diagonal_top_right_to_bottom_left(
                state
            )
        return winner_flag, winner_value

    # 盤面が勝利条件を満たしているかを判定する関数
    def judge_board(self, state):
        winner_flag, winner_value = self.judge_vertical(state)
        if not winner_flag:
            winner_flag, winner_value = self.judge_horizontal(state)
            if not winner_flag:
                winner_flag, winner_value = self.judge_diagonal(state)
        return winner_flag, winner_value

    # ゲームが終了条件を満たしているかを判定する関数
    def judge_game_finish(self, state):
        winner_flag, winner_value = self.judge_board(state)
        if winner_flag:
            return True, winner_value
        else:
            if state.count(self.EMPTY) == 0:
                return True, self.EMPTY
        return False, self.EMPTY

    # 次のターンを準備する関数
    def set_next_turn(self):
        self.elapsed_turn += 1
        self.this_turn = self.elapsed_turn % len(self.player_turn_list)

    # 次のターンを取得する関数
    def get_this_turn(self):
        return self.this_turn

    # ターンを進める関数
    def advance_turn(self):
        this_turn = self.get_this_turn()
        self.set_next_turn()
        return this_turn

    # 盤面を1手だけ進める関数
    def step(self) -> Tuple[int, List[int], List[int], int, int, bool, int]:
        """
        環境を1手だけ進め、その結果を返す。

        Returns:
            action (int): 実行したマス番号
            prev_board (List[int]): 行動前の盤面状態（コピー）
            next_board (List[int]): 行動後の盤面状態（コピー）
            actor_team_value (int): 行動したプレイヤーのチーム値
            next_team_value (int): 次に行動するプレイヤーのチーム値
            done (bool): ゲーム終了フラグ（Trueなら終局）
            result_value (int): 終局時の勝敗チーム値、または引き分けを示す値
        """
        # 1) 手番管理
        current_turn = self.advance_turn()
        next_turn = self.this_turn

        # 2) 行動決定
        player = self.player_turn_list[current_turn]
        action = player.get_action(self)

        # 3) 盤面更新
        prev_board = self.board.copy()
        team_value = player.get_my_team_value()
        self.board[action] = team_value
        self.board_icon[action] = player.get_agent_id()

        # 4) 終局判定
        done, result_value = self.judge_game_finish(self.board)

        # 5) 次手番チーム値取得
        next_team_value = self.player_turn_list[next_turn].get_my_team_value()

        # 6) 戻り値としてまとめて返す
        return (
            action,
            prev_board,
            self.board.copy(),
            team_value,
            next_team_value,
            done,
            result_value,
        )

    # 最後まで自動でプレイする関数
    def auto_play(self):
        self.shuffle_turn_list()
        self.reset()
        finish_flag = False

        for player in self.player_turn_list:
            player.game_init()

        while not finish_flag:
            (
                action,
                prev_board,
                next_board,
                actor_team_value,
                next_team_value,
                finish_flag,
                result_value,
            ) = self.step()
            for player in self.player_turn_list:
                player.append_continue_result(
                    action,
                    prev_board,
                    actor_team_value,
                    next_team_value,
                )

        for player in self.player_turn_list:
            player.append_finish_result(action, self.board, result_value)
        return self.board, self.board_icon

    def print_board(self, board_icon):
        count = 0
        for board_icon_square in board_icon:
            if board_icon_square == self.EMPTY:
                print(" ", end="")
            else:
                for player in self.player_turn_list:
                    if board_icon_square == player.get_agent_id():
                        print(player.get_player_icon(), end="")
                        break

            if count % self.BOARD_SIDE == self.BOARD_SIDE - 1:
                print("")
            count += 1

    def get_rendered_board_data(self) -> List[List[str]]:
        """
        現在の盤面を、2次元リスト形式で取得する。

        各マスには、プレイヤーのアイコン（文字）または空きマスのインデックス（文字列化された数字）が入る。

        Returns:
            List[List[str]]: プレイヤーアイコンまたはマス番号からなる2次元リスト。
        """
        rendered_board = []
        for row in range(self.BOARD_SIDE):
            row_data = []
            for col in range(self.BOARD_SIDE):
                idx = row * self.BOARD_SIDE + col
                value = self.board[idx]
                if value == self.EMPTY:
                    cell = f"{idx:2}"
                else:
                    icon = "?"
                    for player in self.player_turn_list:
                        if value == player.get_my_team_value():
                            icon = player.get_player_icon()
                            break
                    cell = f"{icon:2}"
                row_data.append(cell)
            rendered_board.append(row_data)
        return rendered_board

    def get_rendered_board_str(self) -> str:
        """
        現在の盤面を整形された文字列として取得する。

        `get_rendered_board_data()` を元に、見やすい形で改行付きの文字列に整形する。

        Returns:
            str: 見やすく整形された盤面の文字列（改行区切り）。
        """
        board_data = self.get_rendered_board_data()
        return "\n" + "\n".join([" | ".join(row) for row in board_data]) + "\n"

    # エージェント達を保存する関数
    def save_player_list(self):
        for player in self.PLAYER_LIST:
            player.save_agent()

    # ゲームに参加しているプレイヤーのリストを取得する関数
    def get_player_list(self):
        return self.PLAYER_LIST

    # ゲームレートをリセットする関数
    def reset_game_rate(self):
        for game_player in self.PLAYER_LIST:
            game_player.reset_rate()
