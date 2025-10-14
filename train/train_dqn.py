"""
train_dqn.py

五目並べDQNエージェントの学習・テストループを実行するスクリプト。
Trainerクラスを利用して学習部分を一元化する。
"""

# 定数定義
BOARD_SIDE: int = 3  # 盤面サイズ（3×3）
REWARD_LINE: int = 3  # N目並べのN
LEARNING_COUNT: int = 100  # 学習エピソード数
TEST_COUNT: int = 100  # 最終テスト数
PRINT_TEST_RATE: int = max(1, TEST_COUNT // 5)  # テスト中表示頻度
EVAL_EPISODES: int = 100  # 評価時のテストゲーム数
SAVE_FREQUENCY: int = LEARNING_COUNT // 20  # モデル保存頻度
LOG_FREQUENCY: int = LEARNING_COUNT // 20  # ログ出力頻度

# Trainer設定
TRAINER_CONFIG = {
    "total_episodes": LEARNING_COUNT,
    "learn_iterations_per_episode": 1,
    "save_frequency": SAVE_FREQUENCY,
    "log_frequency": LOG_FREQUENCY,
}

# モジュールインポート
import torch
from agent.network.q_network import QNetwork  # Qネットワーク定義
from agent.network.q_network import set_network  # ネットワーク初期化関数
from N_Mark_Alignment_env import N_Mark_Alignment_Env  # 環境クラス
from agent.dqn.dqn_agent import DQN_Agent  # DQNエージェント
from saver.dqn_agent_saver.model_saver import ModelSaver  # モデル保存クラス
from train.trainer import Trainer  # 学習ループ管理クラス
from agent.N_Mark_Alignment_random_npc import (
    N_Mark_Alignment_random_npc as Random_NPC,
)  # ランダムNPC


def main() -> None:
    """
    DQNエージェントを学習およびテストする。
    Trainerを用いて学習フェーズを実行後、テストフェーズではランダムNPCと対戦する。
    """
    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_team_values = 2  # チーム値の数（例：2チーム）

    # ネットワーク定義と初期化
    policy_net, target_net = set_network(BOARD_SIDE, num_team_values, device)

    # エージェントA, Bの生成（学習モード）
    agent_A = DQN_Agent(
        player_icon="A",
        player_value=0,
        learning=True,
        policy_net=policy_net,
        target_net=target_net,
        device=device,
    )
    # agent_A = Random_NPC(player_icon="A", player_value=0)  # エージェントAのランダムNPC
    agent_B = DQN_Agent(
        player_icon="B",
        player_value=1,
        learning=True,
        policy_net=policy_net,
        target_net=target_net,
        device=device,
    )
    # agent_B = Random_NPC(player_icon="B", player_value=1)  # エージェントBのランダムNPC
    player_list = [agent_A, agent_B]
    if len(player_list) != num_team_values:
        raise ValueError(
            f"プレイヤー数({len(player_list)})とチーム値の数({num_team_values})が一致しません。"
        )

    # 環境初期化
    env = N_Mark_Alignment_Env(
        board_side=BOARD_SIDE,
        reward_line=REWARD_LINE,
        player_list=player_list,
    )
    env.reset_game_rate()

    # モデルセーバー初期化
    saver = ModelSaver()

    # Trainer初期化および学習実行
    trainer = Trainer(
        env=env,
        agent=agent_A,
        model_saver=saver,
        config=TRAINER_CONFIG,
        # eval_interval=LEARNING_COUNT // 5,
        eval_episodes=EVAL_EPISODES,
    )
    trainer.train()

    # テストフェーズ準備：新たにランダムNPCと対戦する環境を構築
    agent_A.set_learning(False)
    random_npc = Random_NPC(player_icon="B", player_value=1)
    player_list = [agent_A, random_npc]
    env.set_player(player_list=player_list)
    agent_A.reset_rate()
    env.reset_game_rate()

    # テストフェーズ: auto_play() を使って1エピソードずつ実行
    for i in range(1, TEST_COUNT + 1):
        _, board_icon = env.auto_play()
        if i % PRINT_TEST_RATE == 0 or i == TEST_COUNT:
            env.print_board(board_icon)
            w, l, d = agent_A.get_rate()
            print(f"テスト{i}回目 → win:{w:.3f}%, lose:{l:.3f}%, draw:{d:.3f}%")

    # 最終結果表示
    w, l, d = agent_A.get_rate()
    print(f"最終テスト → 勝率{w:.3f}%  負率{l:.3f}%  引き分け率{d:.3f}%")


if __name__ == "__main__":
    main()
