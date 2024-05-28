import numpy as np
import random
from enum import Enum
from collections import defaultdict
import copy
import math
import pickle


class TicTacToeVariable:
    def __init__(self, BOARD_SIDE, REWARD_LINE):
        self.board_side = BOARD_SIDE
        self.reward_line = REWARD_LINE

        self.win_point = (1 + self.board_side - self.reward_line) * 1.5
        self.lose_point = -((1 + self.board_side - self.reward_line) * 1)
        self.draw_point = -((1 + self.board_side - self.reward_line) * 0.3)
        self.continue_point = -((1 + self.board_side - self.reward_line) * 0.01)


class TURN(Enum):
    O = 1
    X = -1


class TicTacToeEnv:

    # 0 is empty
    # 1 is "O"
    # -1 is "X"
    #
    # board
    # 0 1 2
    # 3 4 5
    # 6 7 8

    def __init__(self, tic_tac_toe_variables):
        self.tic_tac_toe_variables = tic_tac_toe_variables
        self.reset()
        self.penalty = -(
            (
                1
                + self.tic_tac_toe_variables.board_side
                - self.tic_tac_toe_variables.reward_line
            )
            * 0.5
        )
        self.continue_illegal_move_count = 0

    def board_to_string(self, state=None):
        if state == None:
            state = self.board

        result = ""
        for a in state:
            if a == TURN.O:
                result += "O"
            elif a == TURN.X:
                result += "X"
            else:
                result += " "
        return result

    def reset(self):
        self.board = [0] * (self.tic_tac_toe_variables.board_side**2)

    def __len__(self):
        return len(self.board)

    def calc_reward(self, state):
        if self.board == state and state.count(0) != 0:
            return self.penalty
        # 縦
        square_state_flag = False
        for col in range(
            0,
            self.tic_tac_toe_variables.board_side
            * (
                1
                + (
                    self.tic_tac_toe_variables.board_side
                    - self.tic_tac_toe_variables.reward_line
                )
            ),
        ):
            if 0 != state[col]:
                for r_point in range(1, self.tic_tac_toe_variables.reward_line):
                    if (
                        state[col]
                        == state[col + self.tic_tac_toe_variables.board_side * r_point]
                    ):
                        square_state_flag = True
                    else:
                        square_state_flag = False
                        break
                if square_state_flag:
                    return state[col]

        # 横
        square_state_flag = False
        for c_r in range(
            0,
            1
            + (
                self.tic_tac_toe_variables.board_side
                - self.tic_tac_toe_variables.reward_line
            ),
        ):
            for row in range(0, self.tic_tac_toe_variables.board_side):
                if 0 != state[(row * self.tic_tac_toe_variables.board_side) + c_r]:
                    for c_point in range(1, self.tic_tac_toe_variables.reward_line):
                        if (
                            state[(row * self.tic_tac_toe_variables.board_side) + c_r]
                            == state[
                                ((row * self.tic_tac_toe_variables.board_side) + c_r)
                                + c_point
                            ]
                        ):
                            square_state_flag = True
                        else:
                            square_state_flag = False
                            break
                    if square_state_flag:
                        return state[
                            (row * self.tic_tac_toe_variables.board_side) + c_r
                        ]

        # 斜め
        # 左上から右下に
        square_state_flag = False
        for start_point_col in range(
            0,
            1
            + (
                self.tic_tac_toe_variables.board_side
                - self.tic_tac_toe_variables.reward_line
            ),
        ):
            for start_point_row in range(
                0,
                1
                + (
                    self.tic_tac_toe_variables.board_side
                    - self.tic_tac_toe_variables.reward_line
                ),
            ):
                if (
                    0
                    != state[
                        start_point_col
                        + (start_point_row * self.tic_tac_toe_variables.board_side)
                    ]
                ):
                    for oblique_point in range(
                        1, self.tic_tac_toe_variables.reward_line
                    ):
                        if (
                            state[
                                start_point_col
                                + (
                                    start_point_row
                                    * self.tic_tac_toe_variables.board_side
                                )
                            ]
                            == state[
                                (
                                    start_point_col
                                    + (
                                        start_point_row
                                        * self.tic_tac_toe_variables.board_side
                                    )
                                )
                                + (
                                    (self.tic_tac_toe_variables.board_side + 1)
                                    * oblique_point
                                )
                            ]
                        ):
                            square_state_flag = True
                        else:
                            square_state_flag = False
                            break
                    if square_state_flag:
                        return state[
                            start_point_col
                            + (start_point_row * self.tic_tac_toe_variables.board_side)
                        ]

        # 右上から左下に
        square_state_flag = False
        for start_point_col in range(
            (self.tic_tac_toe_variables.board_side - 1)
            - (
                self.tic_tac_toe_variables.board_side
                - self.tic_tac_toe_variables.reward_line
            ),
            self.tic_tac_toe_variables.board_side,
        ):
            for start_point_row in range(
                0,
                1
                + (
                    self.tic_tac_toe_variables.board_side
                    - self.tic_tac_toe_variables.reward_line
                ),
            ):
                if (
                    0
                    != state[
                        start_point_col
                        + (start_point_row * self.tic_tac_toe_variables.board_side)
                    ]
                ):
                    for oblique_point in range(
                        1, self.tic_tac_toe_variables.reward_line
                    ):
                        if (
                            state[
                                start_point_col
                                + (
                                    start_point_row
                                    * self.tic_tac_toe_variables.board_side
                                )
                            ]
                            == state[
                                (
                                    start_point_col
                                    + (
                                        start_point_row
                                        * self.tic_tac_toe_variables.board_side
                                    )
                                )
                                + (
                                    (self.tic_tac_toe_variables.board_side - 1)
                                    * oblique_point
                                )
                            ]
                        ):
                            square_state_flag = True
                        else:
                            square_state_flag = False
                            break
                    if square_state_flag:
                        return state[
                            start_point_col
                            + (start_point_row * self.tic_tac_toe_variables.board_side)
                        ]

        return self.tic_tac_toe_variables.continue_point

    def step(self, action):
        next_state, reward, done = self.T(self.board, action)
        self.board = next_state
        return next_state, reward, done

    def T(self, state, action):
        reward, done = self.R(state)
        if done == True:
            return state, reward, done

        next_state = copy.copy(state)
        if next_state[action] == 0:
            self.continue_illegal_move_count = 0
            next_state[action] = self.check_turn()
        else:
            self.continue_illegal_move_count += 1

        reward, done = self.R(next_state)
        return next_state, reward, done

    def R(self, state):
        reward = self.calc_reward(state)

        # 勝ち
        if reward == TURN.O:
            done = True
            return self.tic_tac_toe_variables.win_point, done

        # 負け
        if reward == TURN.X:
            done = True
            return self.tic_tac_toe_variables.lose_point, done

        # 引き分け
        elif state.count(0) == 0:
            done = True
            return self.tic_tac_toe_variables.draw_point, done

        # ゲーム続行
        else:
            done = False
            return reward, done

    def check_turn(self):
        O = self.board.count(TURN.O)
        X = self.board.count(TURN.X)
        if O == X:
            return TURN.O
        else:
            return TURN.X


class TicTacToeAgent:
    def __init__(self, env, tic_tac_toe_variables, epsilon, min_alpha, learning=True):
        self.epsilon = epsilon
        self.min_alpha = min_alpha
        self.tic_tac_toe_variables = tic_tac_toe_variables

        self.N = defaultdict(self.default_q)
        self.Q = defaultdict(self.default_q)

        # self.N = defaultdict(lambda: [0] * len(env))
        # self.Q = defaultdict(lambda: [0] * len(env))
        self.env = env

        self.prev_state = [0] * len(self.env)
        self.prev_action = -1

        self.turn = TURN.O
        self.learning = learning

    def default_q(self):
        return [0] * (self.tic_tac_toe_variables.board_side**2)

    def policy(self, continue_illegal_move_count):
        if self.learning and (random.random() < self.epsilon):
            return random.randint(0, self.tic_tac_toe_variables.board_side**2 - 1)
        else:
            board_string = self.env.board_to_string()
            if board_string in self.Q:

                if (
                    continue_illegal_move_count
                    >= self.tic_tac_toe_variables.board_side**2
                ):
                    return random.randint(
                        0, self.tic_tac_toe_variables.board_side**2 - 1
                    )
                else:
                    next_action = self.Q[board_string].index(
                        sorted(self.Q[board_string], reverse=True)[
                            continue_illegal_move_count
                        ]
                    )
                    if self.prev_action == next_action:
                        next_action = random.randint(
                            0, self.tic_tac_toe_variables.board_side**2 - 1
                        )
                    return next_action
            else:
                return random.randint(0, self.tic_tac_toe_variables.board_side**2 - 1)

    def play(self):
        self.env.reset()
        if random.randint(0, 1) == 0:
            self.turn = TURN.O
        else:
            self.turn = TURN.X

        done = False
        next_state = -1
        reward = -1

        rewards = []
        experience = []

        # ゲームを最後までプレイ
        while not done:
            if self.env.check_turn() == self.turn:

                if self.prev_action != -1:
                    experience.append(
                        {
                            "state": self.env.board_to_string(self.prev_state),
                            "action": self.prev_action,
                            "reward": reward,
                        }
                    )
                    rewards.append(reward)

                selected_action = self.policy(self.env.continue_illegal_move_count)
                self.prev_state = copy.copy(self.env.board)
                self.prev_action = selected_action

                next_state, reward, done = self.env.step(selected_action)

                if done == True:
                    experience.append(
                        {
                            "state": self.env.board_to_string(self.prev_state),
                            "action": self.prev_action,
                            "reward": reward,
                        }
                    )
                    rewards.append(reward)

            else:
                selected_action = random.randint(
                    0, self.tic_tac_toe_variables.board_side**2 - 1
                )
                self.env.step(selected_action)

        # 価値を計算して、価値関数を更新
        for i, x in enumerate(experience):
            s, a = x["state"], x["action"]
            G, t = 0, 0
            for j in range(i, len(experience)):
                G += math.pow(0.9, t) * experience[j]["reward"]
                t += 1
            self.N[s][a] += 1
            alpha = 1 / self.N[s][a]
            alpha = max(alpha, self.min_alpha)
            self.Q[s][a] += alpha * (G - self.Q[s][a])
        self.prev_action = -1

        return rewards


def print_tictactoe_board(env):
    count = 0
    for b in env.board:
        if b == TURN.O:
            print("○", end="")
        if b == TURN.X:
            print("×", end="")
        if b == 0:
            print(" ", end="")
        if (
            count % env.tic_tac_toe_variables.board_side
            == env.tic_tac_toe_variables.board_side - 1
        ):
            print("")
        count += 1
