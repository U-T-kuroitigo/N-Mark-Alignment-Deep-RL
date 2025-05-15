import abc
from collections import defaultdict
import datetime
import os
import pickle
import numpy as np
from uuid6 import uuid7


class Agent_Model(metaclass=abc.ABCMeta):
    def __init__(self, player_icon, player_value):
        self.set_player_icon(player_icon)
        self.set_player_value(player_value)
        self.board_side = 0
        self.my_team_value = 0
        self.empty_value = -1
        self.team_value_list = []
        self.N = None
        self.Q = None
        self.win = 0
        self.lose = 0
        self.draw = 0
        self.game_count = 0
        self.AGENT_ID = uuid7()

    def get_agent_id(self):
        return self.AGENT_ID

    def init_after_env(self, env):
        def default_q():
            return np.full(self.board_side**2, 0.0, dtype=float)

        self.board_side = env.BOARD_SIDE
        self.reward_line = env.REWARD_LINE
        self.empty_value = env.EMPTY
        self.set_team_info(env.team_list)
        self.prev_state = [self.empty_value] * self.board_side
        self.prev_action = -1
        if self.N is None:
            self.N = {}
            for team_value in self.team_value_list:
                self.N[team_value] = defaultdict(default_q)
        if self.Q is None:
            self.Q = {}
            for team_value in self.team_value_list:
                self.Q[team_value] = defaultdict(default_q)
        self.game_init()

    # プレイヤーを表す見た目(○、×など)を設定する関数
    def set_player_icon(self, player_icon):
        self.player_icon = player_icon

    # プレイヤーを表す値を設定する関数
    def set_player_value(self, player_value):
        self.player_value = player_value

    # チーム情報を設定する関数
    def set_team_info(self, team_list):
        # 自身のチームを設定する関数
        def set_my_team_value(team_list):
            self.my_team_value = team_list[self.player_value]

        # 盤面に存在するチームを設定する関数
        def set_team_list(team_list):
            self.team_value_list = list(team_list.values())

        set_my_team_value(team_list)
        set_team_list(team_list)

    # プレイヤーを表す見た目(○、×など)を取得する関数
    def get_player_icon(self):
        return self.player_icon

    # プレイヤーを表す値を取得する関数
    def get_player_value(self):
        return self.player_value

    # 盤面のサイズを取得する関数
    def get_board_side(self):
        return self.board_side

    # 盤面のEMPTYの値を取得する関数
    def get_empty_value(self):
        return self.empty_value

    # 自身のチームを取得する関数
    def get_my_team_value(self):
        return self.my_team_value

    # 盤面に存在するチームを取得する関数
    def get_team_list(self):
        return self.team_value_list

    def game_init(self):
        self.rewards = {}
        self.experience = {}
        for team_value in self.team_value_list:
            self.rewards[team_value] = []
            self.experience[team_value] = []

    @abc.abstractmethod
    # 学習するかどうかを設定する関数
    def set_learning(self, learning):
        raise NotImplementedError()

    # 渡されたenvから次の行動を決める関数
    @abc.abstractmethod
    def get_action(self, env):
        raise NotImplementedError()

    # 渡されたfinish_flag, result_valueをもとに結果を追加する関数
    @abc.abstractmethod
    def append_continue_result(self, action, state, action_team_value):
        raise NotImplementedError()

    # 渡されたresult_valueをもとに結果を追加する関数
    @abc.abstractmethod
    def append_finish_result(self, action, result_value, state):
        raise NotImplementedError()

    # 盤面を文字列に変換する関数
    def get_state_string(self, state):
        result = ""
        for square in state:
            if square in self.team_value_list:
                result += str(square)
            elif square == self.empty_value:
                result += " "
            # 例外 どれにも属さない値
            else:
                result += "?"
        return result

    # 勝率等を取得する関数
    def get_rate(self):
        if not self.game_count == 0:
            win_rate = self.win / self.game_count * 100
            lose_rate = self.lose / self.game_count * 100
            draw_rate = self.draw / self.game_count * 100
        else:
            win_rate = 0
            lose_rate = 0
            draw_rate = 0
        return win_rate, lose_rate, draw_rate

    # 勝率等を初期化する関数
    def reset_rate(self):
        self.win = 0
        self.lose = 0
        self.draw = 0
        self.game_count = 0

    def __getstate__(self):
        state = self.__dict__.copy()
        state["N"] = {k: dict(v) for k, v in self.N.items()}
        state["Q"] = {k: dict(v) for k, v in self.Q.items()}
        return state

    def __setstate__(self, state):
        def default_q():
            return np.full(self.board_side**2, 0.0, dtype=float)

        self.__dict__.update(state)
        self.N = {k: defaultdict(default_q, v) for k, v in self.N.items()}
        self.Q = {k: defaultdict(default_q, v) for k, v in self.Q.items()}

    # エージェントを保存する関数
    def save_agent(self):
        write_model_agent_path = ""
        if self.learning:
            dt_now = (datetime.datetime.now()).strftime("%Y%m%d%H%M%S")
            write_model_directory_path = f"agent_model/{self.AGENT_NAME}_{self.board_side}_{self.reward_line}_{len(self.team_value_list)}"
            write_model_agent_path = f"{write_model_directory_path}/{self.AGENT_NAME}_{self.board_side}_{self.reward_line}_{len(self.team_value_list)}_{self.learning_count}_{dt_now}.pkl"
            # ディレクトリがない場合、作成する
            if not os.path.exists(write_model_directory_path):
                os.makedirs(write_model_directory_path)
            ### pickleで保存（書き出し）
            with open(write_model_agent_path, mode="wb") as fo:
                pickle.dump(self, fo)
        return write_model_agent_path

    # 学習回数を取得する関数
    @abc.abstractmethod
    def get_learning_count(self):
        raise NotImplementedError()
