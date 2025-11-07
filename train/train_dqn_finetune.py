"""
train_dqn_finetune.py

既存モデルを読み込んで追加学習を行うスクリプト。
YAML 設定を利用し、Trainer による追加学習とランダム NPC との評価を実施する。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from N_Mark_Alignment_env import N_Mark_Alignment_Env
from agent.N_Mark_Alignment_random_npc import (
    N_Mark_Alignment_random_npc as RandomNPC,
)
from agent.dqn.dqn_agent import DQN_Agent
from agent.network.q_network import set_network
from saver.dqn_agent_saver.model_saver import ModelSaver
from train.finetune_config import FinetuneConfig, FinetunePlayerSetting
from train.trainer import Trainer
from utils.logging_utils import LoggingConfig, build_logger


def parse_args() -> argparse.Namespace:
    """
    追加学習スクリプト用の CLI 引数を定義する。

    Returns:
        argparse.Namespace: 解析済み CLI 引数。
    """
    parser = argparse.ArgumentParser(
        description="既存モデルを読み込んで追加学習を行い、ランダム NPC とテストします。"
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="追加学習設定を記載した YAML ファイルのパス。",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        help="ログをファイルへ出力する場合のパス。",
    )
    return parser.parse_args()


def load_metadata_from_model(model_path: Path) -> Dict[str, Any]:
    """
    モデルと同階層の meta ディレクトリからメタ情報を読み込む。

    Args:
        model_path (Path): モデルファイルのパス。

    Returns:
        Dict[str, Any]: 読み込んだメタデータ。
    """
    version_dir = model_path.parent
    meta_path = version_dir / "meta" / f"{model_path.stem}.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"メタデータファイルが見つかりません: {meta_path}")
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_agent_from_model(
    model_path: Path,
    player_setting: FinetunePlayerSetting,
    player_value: int,
    device: torch.device,
    metadata: Dict[str, Any] | None = None,
) -> Tuple[DQN_Agent, Dict[str, Any]]:
    """
    モデルファイルとメタ情報を基に DQN_Agent を生成する。

    Args:
        model_path (Path): モデルファイルのパス。
        player_setting (FinetunePlayerSetting): プレイヤー設定。
        player_value (int): チーム値。
        device (torch.device): 使用デバイス。
        metadata (Optional[Dict[str, Any]]): 既に読み込んでいるメタ情報。

    Returns:
        Tuple[DQN_Agent, Dict[str, Any]]: 復元したエージェントとメタ情報。
    """
    model_metadata = metadata or load_metadata_from_model(model_path)
    board_side = model_metadata.get("board_side")
    team_count = model_metadata.get("team_count", 2)
    if board_side is None:
        raise ValueError("メタデータに board_side が含まれていません。")

    policy_net, target_net = set_network(board_side, team_count, device)
    agent = DQN_Agent(
        player_icon=player_setting.icon,
        player_value=player_value,
        learning=player_setting.learning,
        policy_net=policy_net,
        target_net=target_net,
        device=device,
    )

    saver = ModelSaver()
    if not hasattr(agent, "reward_line"):
        agent.reward_line = 0
    saver.load(str(model_path), agent, load_replay=player_setting.load_replay)
    agent.set_learning(player_setting.learning)

    return agent, model_metadata


def resolve_device(device_type: str) -> torch.device:
    """
    設定値に応じて使用デバイスを決定する。
    """
    if device_type == "cpu":
        return torch.device("cpu")
    if device_type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA が利用できません。device_type を変更してください。"
            )
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_players(
    config: FinetuneConfig, device: torch.device
) -> Tuple[List[DQN_Agent], int, int, Dict[str, Any]]:
    """
    プレイヤー設定を基にエージェント一覧を生成する。
    """
    base_model_path = config.resolve_path(config.base_model_path)
    base_metadata = load_metadata_from_model(base_model_path)
    board_side = base_metadata.get("board_side")
    reward_line = base_metadata.get("reward_line")
    team_count = base_metadata.get("team_count", len(config.player_settings))
    if board_side is None or reward_line is None:
        raise ValueError(
            "メタデータに board_side または reward_line が含まれていません。"
        )
    if len(config.player_settings) != team_count:
        raise ValueError(
            f"player_settings の人数({len(config.player_settings)})とチーム数({team_count})が一致しません。"
        )

    players: List[DQN_Agent] = []
    for idx, setting in enumerate(config.player_settings):
        player_value = setting.player_value if setting.player_value is not None else idx

        if setting.type == "base":
            model_path = config.resolve_path(setting.path or config.base_model_path)
            agent, metadata = build_agent_from_model(
                model_path=model_path,
                player_setting=setting,
                player_value=player_value,
                device=device,
                metadata=base_metadata if model_path == base_model_path else None,
            )
            board_side = metadata.get("board_side", board_side)
            reward_line = metadata.get("reward_line", reward_line)
            players.append(agent)
        elif setting.type == "model":
            if not setting.path:
                raise ValueError(
                    "type='model' のプレイヤーには path を指定してください。"
                )
            model_path = config.resolve_path(setting.path)
            agent, metadata = build_agent_from_model(
                model_path=model_path,
                player_setting=setting,
                player_value=player_value,
                device=device,
            )
            if (
                metadata.get("board_side") != board_side
                or metadata.get("reward_line") != reward_line
            ):
                raise ValueError("対戦相手モデルと盤面サイズ／勝利条件が一致しません。")
            players.append(agent)
        elif setting.type == "self":
            model_path = config.resolve_path(setting.path or config.base_model_path)
            agent, _ = build_agent_from_model(
                model_path=model_path,
                player_setting=setting,
                player_value=player_value,
                device=device,
                metadata=base_metadata,
            )
            players.append(agent)
        elif setting.type == "npc":
            npc = RandomNPC(setting.icon, player_value)
            npc.set_learning(setting.learning)
            players.append(npc)
        else:
            raise ValueError(f"未知のプレイヤー種別です: {setting.type}")

    return players, board_side, reward_line, base_metadata


def main() -> None:
    """
    追加学習設定を読み込み、Trainer を用いて再学習と評価を行う。
    """
    args = parse_args()
    config = FinetuneConfig.from_yaml(args.config) if args.config else FinetuneConfig()

    logger = build_logger(
        LoggingConfig(
            name=__name__ + ".finetune",
            level=config.log_level,
            log_file=args.log_file,
        )
    )

    device = resolve_device(config.device_type)
    logger.info("使用デバイス: %s", device)

    players, board_side, reward_line, base_metadata = prepare_players(config, device)

    env = N_Mark_Alignment_Env(
        board_side=board_side,
        reward_line=reward_line,
        player_list=players,
    )
    env.reset_game_rate()

    trainer_config = config.trainer_config()
    agent_under_training = players[0]

    if config.reset_epsilon and isinstance(agent_under_training, DQN_Agent):
        agent_under_training.epsilon_reset()

    logger.info(
        "追加学習を開始します: episodes=%d, save_every=%d, log_every=%d",
        trainer_config["total_episodes"],
        trainer_config["save_frequency"],
        trainer_config["log_frequency"],
    )

    trainer = Trainer(
        env=env,
        agent=agent_under_training,
        model_saver=ModelSaver(),
        config=trainer_config,
        eval_episodes=config.eval_episodes,
        logger=logger,
    )
    trainer.train()
    logger.info("追加学習が完了しました。")

    # 学習済みモデルをランダム NPC とテストする
    agent_under_training.set_learning(False)
    test_npc = RandomNPC("B", 1)
    test_npc.set_learning(False)
    env.set_player(player_list=[agent_under_training, test_npc])
    agent_under_training.reset_rate()
    env.reset_game_rate()

    test_count = config.eval_episodes
    print_interval = max(1, test_count // 5)

    for i in range(1, test_count + 1):
        _, board_icon = env.auto_play()
        if i % print_interval == 0 or i == test_count:
            env.print_board(board_icon)
            win, lose, draw = agent_under_training.get_rate()
            print(f"テスト{i}回目 → win:{win:.3f}%, lose:{lose:.3f}%, draw:{draw:.3f}%")

    win, lose, draw = agent_under_training.get_rate()
    print(f"最終テスト → 勝率{win:.3f}%  負率{lose:.3f}%  引き分け率{draw:.3f}%")


if __name__ == "__main__":
    main()
