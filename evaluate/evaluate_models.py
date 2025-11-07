"""
evaluate_models.py

保存済みモデル同士を総当たりで対戦させ、評価結果を表示・保存するスクリプト。
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import csv

import torch
import yaml

from utils.logging_utils import LoggingConfig, build_logger

from agent.dqn.dqn_agent import DQN_Agent
from saver.dqn_agent_saver.model_saver import ModelSaver
from agent.model.N_Mark_Alignment_agent_model import Agent_Model
from N_Mark_Alignment_env import N_Mark_Alignment_Env
from evaluate.round_robin_match_runner import RoundRobinMatchRunner
from agent.network.q_network import QNetwork
from agent.network.q_network import set_network  # ネットワーク初期化関数
from agent.playable_agent import PlayableAgent

# ========= 定数定義 ========= #
DEFAULT_BOARD_SIDE: int = 3  # 盤面サイズ
DEFAULT_REWARD_LINE: int = 3  # N目並べのN
DEFAULT_EVAL_EPISODES: int = 200  # 各対戦の試行回数


@dataclass
class ModelEntry:
    """
    評価対象モデルの設定。
    """

    path: str
    icon: str
    player_name: str


@dataclass
class EvaluationConfig:
    """
    評価実行時の設定をまとめたデータクラス。
    """

    board_side: int
    reward_line: int
    eval_episodes: int
    models: List[ModelEntry]
    num_team_values: int = 2
    output_dir: Path = Path("evaluate") / "result"
    record_boards: bool = False
    outputs: List[str] = field(default_factory=lambda: ["csv"])

    @classmethod
    def from_yaml(cls, path: Path) -> "EvaluationConfig":
        """
        YAML ファイルから EvaluationConfig を生成する。

        Args:
            path (Path): 設定ファイルのパス。

        Returns:
            EvaluationConfig: 生成された設定インスタンス。
        """
        if path.suffix.lower() not in {".yaml", ".yml"}:
            raise ValueError("評価設定ファイルは YAML (.yaml / .yml) としてください。")

        raw_text = path.read_text(encoding="utf-8")
        loaded = yaml.safe_load(raw_text)
        if not isinstance(loaded, dict):
            raise ValueError("設定ファイルの形式が不正です（辞書ではありません）。")
        data: Dict[str, Any] = loaded

        models_data = data.get("models")
        if not isinstance(models_data, list):
            raise ValueError("設定ファイルに 'models' リストが含まれていません。")

        normalized_models: List[Dict[str, str]] = []
        for entry in models_data:
            if not isinstance(entry, dict):
                raise ValueError("models 配列の要素は辞書である必要があります。")
            normalized_models.append(
                {**entry, "path": entry["path"].replace("\\", "/")}
            )
        models = [ModelEntry(**entry) for entry in normalized_models]
        return cls(
            board_side=data["board_side"],
            reward_line=data["reward_line"],
            eval_episodes=data["eval_episodes"],
            num_team_values=data.get("num_team_values", 2),
            models=models,
            output_dir=Path(data.get("output_dir", "evaluate/result")),
            record_boards=data.get("record_boards", False),
            outputs=data.get("outputs", ["csv"]),
        )


def get_model_list() -> list:
    """
    評価対象となるモデルファイルとプレイヤー記号を指定。
    """

    most_performing_model = [
        {
            "path": "agent_model\\DQN-Agent_3_3_2\\DQN-Agent_3_3_2_45000_202507191701\\DQN-Agent_3_3_2_45000_202507191701.pt",
            "icon": "IA",
            "player_name": "DQN3x3x2_45k_A",
        },
        {
            "path": "agent_model\\DQN-Agent_3_3_2\\DQN-Agent_3_3_2_45000_202507191701\\DQN-Agent_3_3_2_45000_202507191701.pt",
            "icon": "IB",
            "player_name": "DQN3x3x2_45k_B",
        },
    ]

    return most_performing_model


def build_default_config() -> EvaluationConfig:
    """
    従来の定数・モデル一覧から EvaluationConfig を生成する。

    Returns:
        EvaluationConfig: 既定値を用いた設定。
    """
    models = [ModelEntry(**entry) for entry in get_model_list()]
    return EvaluationConfig(
        board_side=DEFAULT_BOARD_SIDE,
        reward_line=DEFAULT_REWARD_LINE,
        eval_episodes=DEFAULT_EVAL_EPISODES,
        num_team_values=2,
        models=models,
    )


def _load_metadata_from_model_path(model_path: Path) -> Dict[str, Any]:
    """
    モデルファイルと同じディレクトリに保存されたメタ情報を読み込む。

    Args:
        model_path (Path): モデルファイル (.pt) のパス。

    Returns:
        Dict[str, Any]: メタデータ。存在しない場合は空 dict。
    """
    meta_path = model_path.parent / "meta" / f"{model_path.stem}.json"
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {}


def load_agent_model(
    filepath: str, board_side: int, reward_line: int, num_team_values: int
) -> DQN_Agent:
    """
    モデルファイルから DQN エージェントを復元する。

    Args:
        filepath (str): モデルファイルのパス。
        board_side (int): 盤面サイズ。
        reward_line (int): 勝利条件。
        num_team_values (int): 登場チーム数。

    Returns:
        DQN_Agent: 復元されたエージェント。
    """
    filepath_path = Path(filepath)
    metadata = _load_metadata_from_model_path(filepath_path)
    board_side = metadata.get("board_side", board_side)
    reward_line = metadata.get("reward_line", reward_line)
    if "team_value_list" in metadata and isinstance(metadata["team_value_list"], list):
        num_team_values = len(metadata["team_value_list"])
    elif "team_list" in metadata and isinstance(metadata["team_list"], list):
        num_team_values = len(metadata["team_list"])
    elif "team_count" in metadata:
        num_team_values = int(metadata["team_count"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        state_dict = torch.load(filepath, map_location="cpu")
    except Exception:  # pragma: no cover - 読み込み失敗時は設定値を使用
        state_dict = None
    else:
        embedding = state_dict.get("team_embedding.weight")
        if embedding is not None:
            inferred = max(embedding.shape[0] - 1, 1)
            num_team_values = inferred

    policy_net, target_net = set_network(board_side, num_team_values, device)

    dummy_agent = DQN_Agent(
        player_icon="?",
        player_value=0,
        learning=False,
        policy_net=policy_net,
        target_net=target_net,
        device=device,
    )

    dummy_agent.reward_line = reward_line
    dummy_agent.board_side = board_side
    if "team_value_list" in metadata and isinstance(metadata["team_value_list"], list):
        dummy_agent.team_value_list = metadata["team_value_list"]
    elif "team_list" in metadata and isinstance(metadata["team_list"], list):
        dummy_agent.team_value_list = metadata["team_list"]
    else:
        dummy_agent.team_value_list = list(range(num_team_values))

    agent = ModelSaver().load(filepath, dummy_agent, load_replay=False)
    agent.set_learning(False)
    return agent


def create_argument_parser() -> argparse.ArgumentParser:
    """
    コマンドライン引数のパーサを生成する。

    Returns:
        argparse.ArgumentParser: 構成済みのパーサ。
    """
    parser = argparse.ArgumentParser(description="Run round-robin evaluation.")
    parser.add_argument(
        "--config",
        type=Path,
        help="評価設定を記載した YAML ファイルのパス。",
    )
    parser.add_argument(
        "--output",
        action="append",
        choices=["csv", "json"],
        help="出力フォーマット（複数指定可）。指定がなければ設定ファイルの値を使用。",
    )
    parser.add_argument(
        "--record-boards",
        action="store_true",
        help="各試合の盤面ログを出力する場合に指定。",
    )
    parser.add_argument(
        "--no-record-boards",
        action="store_true",
        help="設定ファイルより盤面ログ出力を無効化したい場合に指定。",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="評価用ロガーのログレベル（INFO/DEBUG など）。",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        help="ログをファイルへ保存する場合のパス。",
    )
    return parser


def write_outputs(
    summary: List[Dict[str, float]],
    config: EvaluationConfig,
    player_count: int,
    outputs: List[str],
    logger: logging.Logger,
) -> None:
    """
    評価結果を指定されたフォーマットで保存する。

    Args:
        summary (List[Dict[str, float]]): 集計結果。
        config (EvaluationConfig): 評価設定。
        player_count (int): 評価対象エージェント数。
        outputs (List[str]): 出力フォーマット一覧。
        logger (logging.Logger): ロガー。
    """
    base_dir = (
        config.output_dir / f"{config.board_side}_{config.reward_line}_{player_count}"
    )
    base_dir.mkdir(parents=True, exist_ok=True)

    if "csv" in outputs:
        csv_path = base_dir / "evaluation_history.csv"
        with csv_path.open(mode="w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                ["agent_name", "episode", "win_rate", "lose_rate", "draw_rate"]
            )
            for entry in summary:
                writer.writerow(
                    [
                        entry["agent_name"],
                        entry["episode"],
                        f"{entry['win_rate']:.1f}",
                        f"{entry['lose_rate']:.1f}",
                        f"{entry['draw_rate']:.1f}",
                    ]
                )
        logger.info("評価結果を CSV として保存しました: %s", csv_path)

    if "json" in outputs:
        json_path = base_dir / "evaluation_history.json"
        json_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("評価結果を JSON として保存しました: %s", json_path)


def main() -> None:
    parser = create_argument_parser()
    args = parser.parse_args()

    if args.config:
        config = EvaluationConfig.from_yaml(args.config)
    else:
        config = build_default_config()

    outputs = args.output if args.output else config.outputs
    record_boards = config.record_boards
    if args.record_boards:
        record_boards = True
    if args.no_record_boards:
        record_boards = False

    logger = build_logger(
        LoggingConfig(
            name=__name__ + ".evaluate",
            level=args.log_level,
            log_file=args.log_file,
        )
    )

    model_list: List[Agent_Model] = []
    for idx, entry in enumerate(config.models):
        agent = load_agent_model(
            entry.path, config.board_side, config.reward_line, config.num_team_values
        )
        agent.set_player_icon(entry.icon)
        agent.set_player_value(idx)
        agent.player_name = entry.player_name
        model_list.append(agent)

    if len(model_list) < 2:
        logger.warning("評価対象モデルが 2 体未満のため、評価を実行できません。")
        return

    env = N_Mark_Alignment_Env(
        board_side=config.board_side,
        reward_line=config.reward_line,
        player_list=model_list[:2],
    )

    runner = RoundRobinMatchRunner(
        env=env,
        eval_episodes=config.eval_episodes,
        logger=logger,
        result_dir=config.output_dir,
        record_boards=record_boards,
        write_summary=False,
    )

    summary = runner.evaluate(model_list)
    player_count = len(env.get_player_list())

    ranking = sorted(
        summary, key=lambda x: x["win_rate"] - x["lose_rate"], reverse=True
    )
    for idx, entry in enumerate(ranking, start=1):
        logger.info(
            "%d. %s(%s): win=%.1f%% lose=%.1f%% draw=%.1f%%",
            idx,
            entry["agent_name"],
            entry["agent_icon"],
            entry["win_rate"],
            entry["lose_rate"],
            entry["draw_rate"],
        )

    write_outputs(summary, config, player_count, outputs, logger)


if __name__ == "__main__":
    main()
