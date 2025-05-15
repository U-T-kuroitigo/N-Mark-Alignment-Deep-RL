import N_Mark_Alignment_env as env
import agent.N_Mark_Alignment_agent_001 as agent_001
import agent.N_Mark_Alignment_agent_002 as agent_002
import agent.N_Mark_Alignment_random_npc as random_npc

BOARD_SIDE = 9
REWARD_LINE = 5
LEARNING_COUNT = 100
PLAY_COUNT = 100

agent_A = agent_001.N_Mark_Alignment_Agent_001("A", 0, True)
# agent_A = random_npc.N_Mark_Alignment_random_npc("A", 0)
agent_B = agent_002.N_Mark_Alignment_Agent_002("B", 1, True)
# agent_B = random_npc.N_Mark_Alignment_random_npc("B", 1)
# agent_C = random_npc.N_Mark_Alignment_random_npc("C", 26)
# agent_D = random_npc.N_Mark_Alignment_random_npc("D", 300)

player_list = [agent_A, agent_B]

# player_list = [agent_A, agent_B, agent_C, agent_D]

game_env = env.N_Mark_Alignment_Env(BOARD_SIDE, REWARD_LINE, player_list)
win_rate = 0
lose_rate = 0
draw_rate = 0
game_env.reset_game_rate()
print_play_rate = round(LEARNING_COUNT * 0.2 / 2)
print_test_rate = round(PLAY_COUNT * 0.2)

board, board_icon = None, None

for i in range(1, LEARNING_COUNT + 1):
    board, board_icon = game_env.auto_play()
    if not print_play_rate == 0:
        if i % print_play_rate == 0:
            game_env.print_board(board_icon)
            win_rate, lose_rate, draw_rate = agent_A.get_rate()
            print("win %:", win_rate)
            print("lose %:", lose_rate)
            print("draw %:", draw_rate)
            print(str(i) + "th play(learning) :", end=" ")
            print(agent_A.rewards[agent_A.get_my_team_value()])

print("Last play\n")
if not board_icon is None:
    game_env.print_board(board_icon)
print("win %:", win_rate)
print("lose %:", lose_rate)
print("draw %:", draw_rate)

game_env.save_player_list()

for agent in player_list:
    agent.set_learning(False)

game_env.reset_game_rate()

for i in range(1, PLAY_COUNT + 1):
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
            print(str(i) + "th play(test) :", end=" ")
            print(agent_A.rewards[agent_A.get_my_team_value()])
            print("\n\n")

print(f"盤面:{BOARD_SIDE} × {BOARD_SIDE}")
print(f"N目並べ:{REWARD_LINE}目並べ")
print(f"学習回数:{agent_A.learning_count} 回")
print(f"テスト回数:{PLAY_COUNT} 回")
print(f"勝ち:{round(win_rate,7)} %")
print(f"負け:{round(lose_rate,7)} %")
print(f"引き分け:{round(draw_rate,7)} %")
