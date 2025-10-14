"""
train_dqn_finetune.py

既存モデルを読み込んで追加学習を行うスクリプト。
train_dqn.py の学習フローを流用しつつ、ModelSaver でモデルを復元する。
"""

from __future__ import annotations

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
from train.trainer import Trainer

# ====== 定数定義 ======
# 追加学習対象モデル
TARGET_MODEL_PATH: Path = Path(
    r"agent_model\DQN-Agent_3_3_2\DQN-Agent_3_3_2_10_202510141639\DQN-Agent_3_3_2_10_202510141639.pt"
)

# プレイヤー構成
# type: "base"（追加学習を行うモデル）, "model"（別モデル）, "self"（同モデルを独立インスタンスで使用）, "npc"（ランダムNPC）
PLAYER_SETTINGS: List[Dict[str, Any]] = [
    {"type": "base", "icon": "A", "learning": True, "load_replay": False},
    {"type": "npc", "icon": "B"},
    # {"type": "model", "path": "agent_model/DQN-Agent_3_3_2/.../xxxxx.pt", "icon": "C", "learning": False},
    # {"type": "self", "icon": "D", "learning": False},
]

LEARNING_COUNT: int = 100  # 追加学習で実行するエピソード数
LEARN_ITERATIONS_PER_EPISODE: int = 1
SAVE_FREQUENCY: int | None = None  # None の場合は LEARNING_COUNT // 20
LOG_FREQUENCY: int | None = None  # None の場合は LEARNING_COUNT // 20
EVAL_EPISODES: int = 100

DEVICE_TYPE: str = "auto"  # "auto" / "cpu" / "cuda"
RESET_EPSILON: bool = True


def load_metadata_from_model(model_path: Path) -> Dict[str, Any]:
    """
    モデルと同階層の meta ディレクトリに保存されたメタ情報を読み込む。
    """
    version_dir = model_path.parent
    base_filename = model_path.stem
    meta_path = version_dir / "meta" / f"{base_filename}.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"メタデータファイルが見つかりません: {meta_path}")
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_agent_from_model(
    model_path: Path,
    player_icon: str,
    player_value: int,
    device: torch.device,
    learning: bool,
    load_replay: bool,
    metadata: Dict[str, Any] | None = None,
) -> Tuple[DQN_Agent, Dict[str, Any]]:
    """
    モデルファイルとメタ情報を基に DQN_Agent を復元する。
    """
    model_metadata = metadata or load_metadata_from_model(model_path)
    board_side = model_metadata.get("board_side")
    team_count = model_metadata.get("team_count", 2)
    if board_side is None:
        raise ValueError("メタデータに board_side が含まれていません。")

    policy_net, target_net = set_network(board_side, team_count, device)
    agent = DQN_Agent(
        player_icon=player_icon,
        player_value=player_value,
        learning=learning,
        policy_net=policy_net,
        target_net=target_net,
        device=device,
    )

    saver = ModelSaver()
    if not hasattr(agent, "reward_line"):
        agent.reward_line = 0
    saver.load(str(model_path), agent, load_replay=load_replay)
    agent.set_learning(learning)

    return agent, model_metadata


def resolve_device(device_type: str) -> torch.device:
    """
    デバイス設定を解決する。
    """
    if device_type == "cpu":
        return torch.device("cpu")
    if device_type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA が利用できません。DEVICE_TYPE を cpu か auto に変更してください。")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    if not PLAYER_SETTINGS:
        raise ValueError("PLAYER_SETTINGS が空です。")
    if PLAYER_SETTINGS[0].get("type") != "base":
        raise ValueError("PLAYER_SETTINGS の先頭は type='base' にしてください。")

    base_path_setting = PLAYER_SETTINGS[0].get("path", TARGET_MODEL_PATH)
    base_model_path = Path(base_path_setting).expanduser().resolve()
    target_metadata = load_metadata_from_model(base_model_path)
    board_side = target_metadata.get("board_side")
    reward_line = target_metadata.get("reward_line")
    team_count = target_metadata.get("team_count", len(PLAYER_SETTINGS))
    if board_side is None or reward_line is None:
        raise ValueError("メタデータに board_side または reward_line が含まれていません。")
    if len(PLAYER_SETTINGS) != team_count:
        raise ValueError(
            f"PLAYER_SETTINGS の人数({len(PLAYER_SETTINGS)})がモデルのチーム数({team_count})と一致しません。"
        )

    device = resolve_device(DEVICE_TYPE)
    print(f"[INFO] 使用デバイス: {device}")

    player_list = []
    for idx, setting in enumerate(PLAYER_SETTINGS):
        player_type = setting.get("type")
        icon = setting.get("icon", chr(ord("A") + idx))
        player_value = setting.get("player_value", idx)
        learning = setting.get("learning", idx == 0)
        load_replay = setting.get("load_replay", False)

        if player_type == "base":
            model_path = (
                Path(setting.get("path", base_model_path)).expanduser().resolve()
            )
            agent, metadata = build_agent_from_model(
                model_path=model_path,
                player_icon=icon,
                player_value=player_value,
                device=device,
                learning=learning,
                load_replay=load_replay,
                metadata=target_metadata if model_path == base_model_path else None,
            )
            board_side = metadata.get("board_side", board_side)
            reward_line = metadata.get("reward_line", reward_line)
        elif player_type == "model":
            model_path_setting = setting.get("path")
            if model_path_setting is None:
                raise ValueError("type='model' では 'path' を指定してください。")
            model_path = Path(model_path_setting).expanduser().resolve()
            agent, metadata = build_agent_from_model(
                model_path=model_path,
                player_icon=icon,
                player_value=player_value,
                device=device,
                learning=learning,
                load_replay=load_replay,
            )
            if (
                metadata.get("board_side") != board_side
                or metadata.get("reward_line") != reward_line
            ):
                raise ValueError("対戦相手モデルと盤面サイズ／勝利条件が一致しません。")
        elif player_type == "self":
            agent, metadata = build_agent_from_model(
                model_path=base_model_path,
                player_icon=icon,
                player_value=player_value,
                device=device,
                learning=learning,
                load_replay=load_replay,
                metadata=target_metadata,
            )
        elif player_type == "npc":
            agent = RandomNPC(icon, player_value)
            agent.set_learning(learning)
        else:
            raise ValueError(f"未知のプレイヤー種別です: {player_type}")

        player_list.append(agent)

    env = N_Mark_Alignment_Env(
        board_side=board_side,
        reward_line=reward_line,
        player_list=player_list,
    )
    env.reset_game_rate()

    total_episodes = LEARNING_COUNT
    save_frequency = SAVE_FREQUENCY or max(1, total_episodes // 20)
    log_frequency = LOG_FREQUENCY or max(1, total_episodes // 20)

    trainer_config = {
        "total_episodes": total_episodes,
        "learn_iterations_per_episode": LEARN_ITERATIONS_PER_EPISODE,
        "save_frequency": save_frequency,
        "log_frequency": log_frequency,
    }

    if RESET_EPSILON and isinstance(player_list[0], DQN_Agent):
        player_list[0].epsilon_reset()

    print(
        "[INFO] 追加学習を開始します: "
        f"episodes={total_episodes}, save_every={save_frequency}, log_every={log_frequency}"
    )
    trainer = Trainer(
        env=env,
        agent=player_list[0],
        model_saver=ModelSaver(),
        config=trainer_config,
        eval_episodes=EVAL_EPISODES,
    )
    trainer.train()
    print("[INFO] 追加学習が完了しました。")


if __name__ == "__main__":
    main()
