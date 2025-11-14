"""
MetaInfoSaver の保存・ロード処理を検証するテスト。
"""

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

torch = pytest.importorskip("torch")

from agent.dqn.dqn_agent import DQN_Agent  # noqa: E402
from agent.network.q_network import set_network  # noqa: E402
from saver.dqn_agent_saver.meta_info_saver import MetaInfoSaver  # noqa: E402
from agent.utils.model_version_utils import (
    make_version_dir_and_filename,
)  # noqa: E402


def _create_agent(board_side: int = 3) -> DQN_Agent:
    """メタ情報保存のテストに利用する DQN エージェントを生成する。"""
    device = torch.device("cpu")
    policy_net, target_net = set_network(
        board_side=board_side, num_team_values=2, device=device
    )
    agent = DQN_Agent("A", 0, True, policy_net, target_net, device)
    agent.board_side = board_side
    agent.reward_line = 3
    agent.team_value_list = [0, 1]
    agent.learning_count = 7
    agent.player_name = "MetaTest"
    return agent


def test_meta_info_saver_roundtrip(tmp_path: Path) -> None:
    """メタ情報の保存・ロードで値が復元されることを確認する。"""
    agent = _create_agent()
    saver = MetaInfoSaver()

    version_paths = make_version_dir_and_filename(
        agent.get_metadata(), root_dir=str(tmp_path)
    )

    # 保存
    meta_path = saver.save(agent, version_paths)
    assert Path(meta_path).exists()

    # ファイル内容を直接チェック
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    assert metadata["board_side"] == agent.board_side
    assert metadata["reward_line"] == agent.reward_line
    assert metadata["learning_count"] == agent.learning_count

    # 新しいエージェントへロード
    loaded_agent = _create_agent()
    # 初期値を変えておいてからロードする
    loaded_agent.board_side = 0
    loaded_agent.reward_line = 0
    loaded_agent.learning_count = 0

    version_dir = Path(meta_path).parent.parent
    base_filename = Path(meta_path).stem
    saver.load(loaded_agent, str(version_dir), base_filename)

    assert loaded_agent.board_side == agent.board_side
    assert loaded_agent.reward_line == agent.reward_line
    assert loaded_agent.learning_count == agent.learning_count


def test_meta_info_saver_with_extra_metadata(tmp_path: Path) -> None:
    """追加メタ情報を指定した場合に JSON に追記されることを検証する。"""
    agent = _create_agent()
    saver = MetaInfoSaver()

    version_paths = make_version_dir_and_filename(
        agent.get_metadata(), root_dir=str(tmp_path)
    )
    extra = {"xai_summary": {"saliency_mean": 0.12}}

    meta_path = saver.save(agent, version_paths, extra_metadata=extra)
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    assert metadata["xai_summary"]["saliency_mean"] == 0.12
