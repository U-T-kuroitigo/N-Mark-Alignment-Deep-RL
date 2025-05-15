import sys
import N_Mark_Alignment_env as env
import agent.N_Mark_Alignment_agent_001 as agent_001
import agent.N_Mark_Alignment_agent_002 as agent_002
import agent.N_Mark_Alignment_random_npc as random_npc
import pickle

BOARD_SIDE = 9
REWARD_LINE = 5
LEARNING_COUNT = 1000
LEARNING_CONTINUOUS = 5
TEST_COUNT = 100

PLAY_NUM_OF_PEOPLE = 2
MODEL_PATH = "agent_model"

AGENT_A_DIRECTORY_PATH = f"Agent001_{BOARD_SIDE}_{REWARD_LINE}_{PLAY_NUM_OF_PEOPLE}"
AGENT_A_PATH = (
    f"{MODEL_PATH}/{AGENT_A_DIRECTORY_PATH}/Agent001_9_5_2_100000_20250415150343.pkl"
)
with open(AGENT_A_PATH, mode="br") as fi:
    agent_A = pickle.load(fi)
agent_A.set_learning(True)

# agent_A = agent_001.N_Mark_Alignment_Agent_001("A", 0, True)
# agent_A = random_npc.N_Mark_Alignment_random_npc("A", 0)
# agent_A.set_player_value(0)
# agent_A.set_player_icon("A")
# agent_A.show_self()

AGENT_B_DIRECTORY_PATH = f"Agent002_{BOARD_SIDE}_{REWARD_LINE}_{PLAY_NUM_OF_PEOPLE}"
AGENT_B_PATH = (
    f"{MODEL_PATH}/{AGENT_B_DIRECTORY_PATH}/Agent002_9_5_2_10000_20250415145858.pkl"
)
with open(AGENT_B_PATH, mode="br") as fi:
    agent_B = pickle.load(fi)
agent_B.set_learning(True)

# agent_B = agent_002.N_Mark_Alignment_Agent_002("B", 1, True)
# agent_B = random_npc.N_Mark_Alignment_random_npc("B", 1)
# agent_B.set_player_value(1)
# agent_B.set_player_icon("B")
# agent_B.show_self()

player_list = [agent_A, agent_B]
if not PLAY_NUM_OF_PEOPLE == len(player_list):
    print("Error: プレイヤーの人数が異常値です。", file=sys.stderr)
    sys.exit(1)

print_play_rate = round(LEARNING_COUNT * 0.2)
print_test_rate = round(TEST_COUNT * 0.2)

for i in range(LEARNING_CONTINUOUS):
    game_env = env.N_Mark_Alignment_Env(BOARD_SIDE, REWARD_LINE, player_list)
    win_rate = 0
    lose_rate = 0
    draw_rate = 0
    game_env.reset_game_rate()

    board, board_icon = None, None

    for j in range(1, LEARNING_COUNT + 1):
        board, board_icon = game_env.auto_play()
        if not print_play_rate == 0:
            if j % print_play_rate == 0:
                game_env.print_board(board_icon)
                win_rate, lose_rate, draw_rate = agent_A.get_rate()
                print(str((i * LEARNING_COUNT) + j) + "th play")
                print("win %:", win_rate)
                print("lose %:", lose_rate)
                print("draw %:", draw_rate)
                print("\n")
    for agent in player_list:
        agent.save_agent()

win_rate, lose_rate, draw_rate = agent_A.get_rate()
print("win %:", win_rate)
print("lose %:", lose_rate)
print("draw %:", draw_rate)

for agent in player_list:
    agent.set_learning(False)

game_env.reset_game_rate()

for i in range(1, TEST_COUNT + 1):
    board, board_icon = game_env.auto_play()
    if not print_test_rate == 0:
        if i % print_test_rate == 0:
            game_env.print_board(board_icon)
            game_player_list = game_env.get_player_list()
            for game_player in game_player_list:
                if game_player.get_agent_id() == agent_A.get_agent_id():
                    win_rate, lose_rate, draw_rate = game_player.get_rate()
            print("win %:", win_rate)
            print("lose %:", lose_rate)
            print("draw %:", draw_rate)
            print(f"{i:,}th play(learning)")
            print("\n\n")

print(f"盤面:{BOARD_SIDE} × {BOARD_SIDE}")
print(f"N目並べ:{REWARD_LINE}目並べ")
print(f"学習回数:{agent_A.learning_count} 回")
print(f"テスト回数:{TEST_COUNT} 回")
print(f"勝ち:{round(win_rate,7)} %")
print(f"負け:{round(lose_rate,7)} %")
print(f"引き分け:{round(draw_rate,7)} %")
