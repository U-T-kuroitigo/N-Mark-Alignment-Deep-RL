import numpy as np
import random
from enum import Enum
from collections import defaultdict
import copy
import math
import pickle
import N_Mark_Alignment_setting as setting

BOARD_SIDE = 3
REWARD_LINE = 3
LEARNING_COUNT = 1000
TEST_COUNT = 100

write_model_agent_path = (
    f"agent_model/{BOARD_SIDE}_{REWARD_LINE}_{LEARNING_COUNT}_{TEST_COUNT}.pkl"
)

if __name__ == "__main__":
    tic_tac_toe_variables = setting.TicTacToeVariable(
        BOARD_SIDE, REWARD_LINE, LEARNING_COUNT, TEST_COUNT
    )
    env = setting.TicTacToeEnv(tic_tac_toe_variables)
    agent = setting.TicTacToeAgent(
        env,
        tic_tac_toe_variables,
        epsilon=0.1,
        min_alpha=0.01,
    )

    win = 0
    lose = 0
    draw = 0
    game = 0
    rewards = 0

    for i in range(1, LEARNING_COUNT + 1):
        rewards = agent.play()
        if rewards[-1] == tic_tac_toe_variables.win_point:
            win += 1
        elif rewards[-1] == tic_tac_toe_variables.lose_point:
            lose += 1
        elif rewards[-1] == tic_tac_toe_variables.draw_point:
            draw += 1
        game += 1

        if i % round(LEARNING_COUNT * 0.1) == 0:
            setting.print_tictactoe_board(env)
            print(str(i) + "th play", rewards)
            print("win %:", win / game * 100)
            print("lose %:", lose / game * 100)
            print("draw %:", draw / game * 100)
            win = 0
            lose = 0
            draw = 0
            game = 0

            Qs = agent.Q[env.board_to_string([0] * (BOARD_SIDE**2))]
            for Qs_col in range(0, BOARD_SIDE):
                for Qs_row in range(0, BOARD_SIDE):
                    print(Qs[BOARD_SIDE * Qs_col + Qs_row], end=" ")
                print("")

    # check learned
    agent.learning = False
    win = 0
    lose = 0
    draw = 0
    game = 0

    for i in range(1, TEST_COUNT + 1):
        rewards = agent.play()
        if rewards[-1] == tic_tac_toe_variables.win_point:
            win += 1
        elif rewards[-1] == tic_tac_toe_variables.lose_point:
            lose += 1
        elif rewards[-1] == tic_tac_toe_variables.draw_point:
            draw += 1
        game += 1

        if i % round(TEST_COUNT * 0.2) == 0:
            setting.print_tictactoe_board(env)
            print(str(i) + "th play(test)", rewards)
            print("win %:", win / game * 100)
            print("lose %:", lose / game * 100)
            print("draw %:", draw / game * 100)

            Qs = agent.Q[env.board_to_string([0] * (BOARD_SIDE**2))]
            for Qs_col in range(0, BOARD_SIDE):
                for Qs_row in range(0, BOARD_SIDE):
                    print(Qs[BOARD_SIDE * Qs_col + Qs_row], end=" ")
                print("")
    setting.print_tictactoe_board(env)
    print("Last play", rewards)

    ### pickleで保存（書き出し）
    with open(write_model_agent_path, mode="wb") as fo:
        pickle.dump(agent, fo)
