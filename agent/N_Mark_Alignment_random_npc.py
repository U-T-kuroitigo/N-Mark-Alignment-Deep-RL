import numpy as np
import random
import copy
import agent.model.N_Mark_Alignment_agent_model as model
import math
import pickle
import datetime


class N_Mark_Alignment_random_npc(model.Agent_Model):
    def __init__(self, player_icon, player_value):
        super().__init__(player_icon, player_value)

    # 学習するかどうかを設定する関数
    def set_learning(self, learning):
        # npcは学習しない
        pass

    # 渡されたenvから次の行動を取得する関数
    def get_action(self, env):
        # 行動を決定する関数
        def decide_action():
            return random.randint(0, self.board_side**2 - 1)

        # 行動が問題ないか確認する関数
        def check_next_action(state, next_action):
            next_state = copy.copy(state)
            if next_state[next_action] == self.empty_value:
                return True
            else:
                return False

        state = env.get_board()
        next_action = -1
        checked_ok = False
        while not checked_ok:
            next_action = decide_action()
            checked_ok = check_next_action(state, next_action)
        self.prev_action = next_action

        return next_action

    # 渡されたaction, result_value, stateをもとに結果を追加する関数
    def append_continue_result(self, action, state, action_team_value):
        # 特に何もしない
        pass

    def decide_finish_reward(self, team_value, result_value):
        if result_value == team_value:
            reward = self.WIN_POINT
        elif result_value == self.empty_value:
            reward = self.DRAW_POINT
        else:
            reward = self.LOSE_POINT
        return reward

    # 渡されたresult_valueをもとに結果を追加する関数
    def append_finish_result(self, action, result_value, state):
        self.game_count += 1
        if result_value == self.my_team_value:
            self.win += 1
        elif result_value == self.empty_value:
            self.draw += 1
        else:
            self.lose += 1

    # エージェントを保存する関数
    def save_agent(self):
        # npcは保存しない
        pass

    # 学習回数を取得する関数
    def get_learning_count(self):
        # npcは学習できない
        return 0
