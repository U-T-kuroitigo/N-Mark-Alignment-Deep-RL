import numpy as np
import random
from enum import Enum
from collections import defaultdict
import copy
import math
import pickle
import N_Mark_Alignment_setting as setting

# USE_AGENT = f"{BOARD_SIDE}_{REWARD_LINE}_{LEARNING_COUNT}"
USE_AGENT = f"3_3_100000"

load_model_agent_path = f"agent_model/{USE_AGENT}.pkl"

SETTING_LIST = USE_AGENT.split("_")

BOARD_SIDE = int(SETTING_LIST[0])
REWARD_LINE = int(SETTING_LIST[1])
LEARNING_COUNT = int(SETTING_LIST[2])
PLAY_COUNT = 100000

if __name__ == "__main__":
    tic_tac_toe_variables = setting.TicTacToeVariable(BOARD_SIDE, REWARD_LINE)
    env = setting.TicTacToeEnv(tic_tac_toe_variables)
    ### pickleで保存（書き出し）
    with open(load_model_agent_path, mode="br") as fi:
        agent = pickle.load(fi)

    agent.learning = False
    win = 0
    lose = 0
    draw = 0
    game = 0

    for i in range(1, PLAY_COUNT + 1):
        rewards = agent.play()
        if rewards[-1] == tic_tac_toe_variables.win_point:
            win += 1
        elif rewards[-1] == tic_tac_toe_variables.lose_point:
            lose += 1
        elif rewards[-1] == tic_tac_toe_variables.draw_point:
            draw += 1
        game += 1

        if i % round(PLAY_COUNT * 0.2) == 0:
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
    print("")
    print(f"BOARD:{BOARD_SIDE} × {BOARD_SIDE}")
    print(f"REWARD_LINE:{REWARD_LINE}")
    print(f"LEARNING_COUNT:{LEARNING_COUNT}")
    print(f"PLAY_COUNT:{PLAY_COUNT}")
    print("win %:", win / game * 100)
    print("lose %:", lose / game * 100)
    print("draw %:", draw / game * 100)
    setting.print_tictactoe_board(env)
    print("Last play", rewards)
