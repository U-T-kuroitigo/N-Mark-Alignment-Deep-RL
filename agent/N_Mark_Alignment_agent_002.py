import os
import numpy as np
import random
import copy
import agent.model.N_Mark_Alignment_agent_model as model
import math
import pickle
import datetime


class N_Mark_Alignment_Agent_002(model.Agent_Model):
    AGENT_NAME = "Agent002"
    WIN_POINT = 1.5
    LOSE_POINT = -3
    DRAW_POINT = -0.3
    CONTINUE_POINT = -0.01

    EPSILON = 0.2
    MIN_ALPHA = 0.01

    def __init__(self, player_icon, player_value, learning):
        super().__init__(player_icon, player_value)
        self.set_learning(learning)
        self.continue_illegal_move_count = 0
        self.learning_count = 0

    # 学習するかどうかを設定する関数
    def set_learning(self, learning):
        self.learning = learning

    # 渡されたenvから次の行動を取得する関数
    def get_action(self, env):
        # 行動を決定する関数
        def decide_action(state):
            if self.learning and random.random() < self.EPSILON:
                return random.randint(0, self.board_side**2 - 1)
            else:
                state_string = self.get_state_string(state)
                if state_string in self.Q[self.my_team_value]:
                    if self.continue_illegal_move_count >= self.board_side**2:
                        return random.randint(0, self.board_side**2 - 1)
                    else:
                        q_values = self.Q[self.my_team_value][state_string]
                        sorted_indices = np.argsort(-q_values)
                        next_action = sorted_indices[self.continue_illegal_move_count]
                        return next_action
                else:
                    return random.randint(0, self.board_side**2 - 1)

        # 行動が問題ないか確認する関数
        def check_next_action(state, next_action):
            next_state = copy.copy(state)
            if next_state[next_action] == self.empty_value:
                self.continue_illegal_move_count = 0
                return True
            else:
                self.continue_illegal_move_count += 1
                return False

        state = env.get_board()
        next_action = -1
        checked_ok = False
        while not checked_ok:
            next_action = decide_action(state)
            checked_ok = check_next_action(state, next_action)
        self.prev_action = next_action

        return next_action

    # 渡されたaction, result_value, stateをもとに結果を追加する関数
    def append_continue_result(self, action, state, action_team_value):
        for team_value in self.team_value_list:
            if team_value == action_team_value:
                reward = self.CONTINUE_POINT
                self.experience[team_value].append(
                    {
                        "state": self.get_state_string(state),
                        "action": action,
                        "reward": reward,
                    }
                )
                self.rewards[team_value].append(
                    {"team_value": team_value, "reward": reward}
                )

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
        # 蓄積されたresultから学習する関数
        def learn():
            for team_value in self.team_value_list:
                # 価値を計算して、価値関数を更新
                for i, x in enumerate(self.experience[team_value]):
                    s, a = x["state"], x["action"]
                    G, t = 0, 0
                    for j in range(i, len(self.experience[team_value])):
                        G += math.pow(0.9, t) * self.experience[team_value][j]["reward"]
                        t += 1
                    self.N[team_value][s][a] += 1
                    alpha = 1 / self.N[team_value][s][a]
                    alpha = max(alpha, self.MIN_ALPHA)
                    self.Q[team_value][s][a] += alpha * (G - self.Q[team_value][s][a])
                self.learning_count += 1

        for team_value in self.team_value_list:
            reward = self.decide_finish_reward(team_value, result_value)
            self.experience[team_value].append(
                {
                    "state": self.get_state_string(state),
                    "action": action,
                    "reward": reward,
                }
            )
            self.rewards[team_value].append(
                {"team_value": team_value, "reward": reward}
            )

        if self.learning:
            learn()
        self.game_count += 1
        if result_value == self.my_team_value:
            self.win += 1
        elif result_value == self.empty_value:
            self.draw += 1
        else:
            self.lose += 1

    # 学習回数を取得する関数
    def get_learning_count(self):
        return self.learning_count

    # 自身のもつfieldを表示する関数(デバッグ用)
    def show_self(self):
        print(f"self.AGENT_NAME = {self.AGENT_NAME}")
        print(f"self.WIN_POINT = {self.WIN_POINT}")
        print(f"self.LOSE_POINT = {self.LOSE_POINT}")
        print(f"self.DRAW_POINT = {self.DRAW_POINT}")
        print(f"self.CONTINUE_POINT = {self.CONTINUE_POINT}")
        print(f"self.EPSILON = {self.EPSILON}")
        print(f"self.MIN_ALPHA = {self.MIN_ALPHA}")
        print(f"self.AGENT_ID = {self.AGENT_ID}")
        print(f"self.player_icon = {self.player_icon}")
        print(f"self.player_value = {self.player_value}")
        print(f"self.learning = {self.learning}")
