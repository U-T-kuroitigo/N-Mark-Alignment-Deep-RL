"""
train_dqn.py

五目並べ DQN エージェントの学習・テストループを実行するスクリプト。
YAML 設定を読み込んで Trainer を初期化し、学習後にランダム NPC との
テスト対戦を行う。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from N_Mark_Alignment_env import N_Mark_Alignment_Env
from agent.N_Mark_Alignment_random_npc import (
    N_Mark_Alignment_random_npc as Random_NPC,
)
from agent.dqn.dqn_agent import DQN_Agent
from agent.network.q_network import set_network
from saver.dqn_agent_saver.model_saver import ModelSaver
from train.trainer import Trainer
from train.training_config import TrainingConfig
from utils.logging_utils import LoggingConfig, build_logger


def parse_args() -> argparse.Namespace:
    """
    train_dqn スクリプト用の CLI 引数を定義する。

    Returns:
        argparse.Namespace: 解析済み引数。
    """
    parser = argparse.ArgumentParser(
        description="DQN エージェントを学習し、ランダム NPC とテストする。"
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="トレーニング設定を記載した YAML ファイルのパス。",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        help="ログをファイルへ保存する場合のパス。",
    )
    return parser.parse_args()


def main() -> None:
    """
    YAML 設定を読み込み、Trainer を用いた学習とテスト評価を実行する。
    """
    args = parse_args()
    if args.config:
        training_config = TrainingConfig.from_yaml(args.config)
    else:
        training_config = TrainingConfig()

    logger = build_logger(
        LoggingConfig(
            name=__name__ + ".train_dqn",
            level=training_config.log_level,
            log_file=args.log_file,
        )
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_team_values = training_config.num_team_values

    policy_net, target_net = set_network(
        training_config.board_side, num_team_values, device
    )

    agent_a = DQN_Agent(
        player_icon="A",
        player_value=0,
        learning=True,
        policy_net=policy_net,
        target_net=target_net,
        device=device,
    )
    agent_b = DQN_Agent(
        player_icon="B",
        player_value=1,
        learning=True,
        policy_net=policy_net,
        target_net=target_net,
        device=device,
    )
    player_list = [agent_a, agent_b]
    if len(player_list) != num_team_values:
        raise ValueError(
            f"プレイヤー数({len(player_list)})とチーム数({num_team_values})が一致していません。"
        )

    env = N_Mark_Alignment_Env(
        board_side=training_config.board_side,
        reward_line=training_config.reward_line,
        player_list=player_list,
    )
    env.reset_game_rate()

    trainer = Trainer(
        env=env,
        agent=agent_a,
        model_saver=ModelSaver(),
        config=training_config.to_trainer_dict(),
        eval_episodes=training_config.eval_episodes,
        logger=logger,
    )
    trainer.train()

    agent_a.set_learning(False)
    random_npc = Random_NPC(player_icon="B", player_value=1)
    env.set_player(player_list=[agent_a, random_npc])
    agent_a.reset_rate()
    env.reset_game_rate()

    test_count = training_config.eval_episodes
    print_interval = max(1, test_count // 5)

    for i in range(1, test_count + 1):
        _, board_icon = env.auto_play()
        if i % print_interval == 0 or i == test_count:
            env.print_board(board_icon)
            win, lose, draw = agent_a.get_rate()
            print(f"テスト{i}回目 → win:{win:.3f}%, lose:{lose:.3f}%, draw:{draw:.3f}%")

    win, lose, draw = agent_a.get_rate()
    print(f"最終テスト → 勝率{win:.3f}%  負率{lose:.3f}%  引き分け率{draw:.3f}%")


if __name__ == "__main__":
    main()
