import numpy as np
import random
from enum import Enum

import copy
import math
import pickle
import random
import agent.model.N_Mark_Alignment_agent_model as agent_model


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
        self.reset_board()
        self.reset_prev()
        self.reset_turn()

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

    # 順番をシャッフルする関数
    def shuffle_turn_list(self):
        self.player_turn_list = random.sample(self.PLAYER_LIST, len(self.PLAYER_LIST))

    # ターンをリセットする関数
    def reset_turn(self):
        self.this_turn = 0
        self.elapsed_turn = 0

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

    # 最後まで自動でプレイする関数
    def auto_play(self):
        self.shuffle_turn_list()
        self.reset_board()
        self.reset_prev()
        self.reset_turn()
        finish_flag = False

        for player in self.player_turn_list:
            player.game_init()

        while not finish_flag:
            this_turn = self.advance_turn()
            this_player = self.player_turn_list[this_turn]
            this_player_team_value = this_player.get_my_team_value()
            this_action = this_player.get_action(self)
            self.prev_board = copy.copy(self.board)
            self.board[this_action] = this_player_team_value
            self.board_icon[this_action] = this_player.get_agent_id()
            finish_flag, result_value = self.judge_game_finish(self.board)
            for player in self.player_turn_list:
                player.append_continue_result(
                    this_action,
                    self.prev_board,
                    this_player_team_value,
                )

        for player in self.player_turn_list:
            player.append_finish_result(this_action, result_value, self.board)
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

    # エージェント達を保存する関数
    def save_player_list(self):
        for player in self.PLAYER_LIST:
            player.save_agent()

    # ゲームに参加しているプレイヤーのリストを取得する関数
    def get_player_list(self):
        return self.PLAYER_LIST

    #
    def reset_game_rate(self):
        for game_player in self.PLAYER_LIST:
            game_player.reset_rate()
